"""
play_rl.py  --  Phase 3: Play Mode with trained RL NPC.

Loads the saved PPO model and runs the full Pygame game with the
RL policy directly controlling the NPC each frame.

The NPC's internal state machine (decide_action) is bypassed: instead,
build_observation() constructs the same 15-dim obs the model was
trained on, model.predict() returns an action, and we apply it to
the NPC's physics directly.

Usage
-----
  python play_rl.py                   # uses models/best_model.zip
  python play_rl.py --model models/npc_ppo_final
"""

import os
import sys
import math
import argparse

import numpy as np
import pygame

# Ensure real display (not dummy) -- import before game/env
from stable_baselines3 import PPO

from game      import Game, GameState, SCREEN_W, SCREEN_H
from env       import build_observation, MOVE_DIRS, NPC_RL_SPEED, N_ATK_RANGE, N_ATK_CD
from npc       import ATTACK_DAMAGE as N_ATK_DMG
from obstacles import resolve_circle_rect, Particle, PLAYER_COL, NPC_COL

# Priority order: best checkpoint -> final model -> smoke test (demo only)
MODEL_PRIORITY = [
    "models/best_model",
    "models/npc_ppo_final",
    "models/smoke_test",
]


# ---------------------------------------------------------------------------
# RL-controlled Game subclass
# ---------------------------------------------------------------------------

class RLGame(Game):
    """
    Extends Game.  NPC's decide_action() is replaced each frame by
    model.predict(obs).  Everything else (player input, HUD, win-logic)
    is inherited unchanged.
    """

    def __init__(self, model_path: str, deterministic: bool = False):
        super().__init__()
        print(f"Loading RL model: {model_path}.zip")
        self.rl_model  = PPO.load(model_path)
        self.deterministic = deterministic
        self._npc_atk_cd = 0.0
        pygame.display.set_caption(
            "AI NPC  [RL Policy]  --  WASD/Arrows=move  SPACE=attack  R=reset"
        )

    # --- override per-round reset so RL state is clean
    def reset_round(self):
        super().reset_round()
        self._npc_atk_cd = 0.0

    # ------------------------------------------------------------------
    def _update(self, dt: float) -> None:
        if self.state != GameState.PLAYING:
            return

        keys = pygame.key.get_pressed()

        # --- Human player (unchanged from Game) ---
        self.player.update(dt, keys, self.obstacles)

        # --- RL NPC ---
        obs             = build_observation(self.npc, self.player, self.target)
        action, _state  = self.rl_model.predict(obs, deterministic=self.deterministic)
        move_idx        = int(action[0])
        combat_idx      = int(action[1])
        do_attack       = (combat_idx == 1)
        do_defend       = (combat_idx == 2)

        dist_to_player = self.npc.pos.distance_to(self.player.pos)
        # Guardrail: only block if player is alive and within FOV range (220px)
        if dist_to_player > 220 or not self.player.alive:
            do_defend = False

        self.npc.is_defending = do_defend

        # Apply movement directly (bypass state machine)
        dx, dy = MOVE_DIRS[move_idx]
        mv = pygame.math.Vector2(dx, dy)
        if mv.length() > 0:
            mv.normalize_ip()

        # Face target angle: face player when close/combatting, face movement direction otherwise
        if dist_to_player <= 220 and self.player.alive:
            target_angle = math.atan2(self.player.pos.y - self.npc.pos.y,
                                      self.player.pos.x - self.npc.pos.x)
        elif mv.length() > 0:
            target_angle = math.atan2(mv.y, mv.x)
        else:
            target_angle = self.npc.angle

        # Smooth turn interpolation (lerp) to prevent rapid spinning/jittering
        diff_angle = (target_angle - self.npc.angle + math.pi) % (2 * math.pi) - math.pi
        self.npc.angle += diff_angle * 0.15

        # Speed is reduced by 50% when defending
        speed = NPC_RL_SPEED * 0.5 if do_defend else NPC_RL_SPEED
        self.npc.pos += mv * speed * dt
        self.npc.pos.x = max(self.npc.radius,
                             min(self.npc.pos.x, SCREEN_W - self.npc.radius))
        self.npc.pos.y = max(self.npc.radius,
                             min(self.npc.pos.y, SCREEN_H - self.npc.radius))
        self.npc.pos = resolve_circle_rect(self.npc.pos, self.npc.radius,
                                          self.obstacles)

        # NPC attack cooldown tick
        if self._npc_atk_cd > 0:
            self._npc_atk_cd -= dt
        if self.npc._atk_vis > 0:
            self.npc._atk_vis -= dt

        dist_to_player = self.npc.pos.distance_to(self.player.pos)
        if (do_attack and self._npc_atk_cd <= 0
                and dist_to_player <= N_ATK_RANGE
                and self.player.alive):
            self.player.take_damage(N_ATK_DMG)
            self._npc_atk_cd  = N_ATK_CD
            self.npc._atk_vis = 0.15
            self.spawn_hit_sparks(self.player.pos, NPC_COL)

        # Update displayed NPC state label (approximate from context)
        if not self.npc.alive:
            pass
        elif self.npc.is_defending:
            self.npc.state = "DEFEND"
        elif self.npc.hp / self.npc.max_hp < 0.30:
            self.npc.state = "RETREAT"
        elif do_attack and dist_to_player <= N_ATK_RANGE:
            self.npc.state = "ATTACK"
        elif dist_to_player < 220:
            self.npc.state = "CHASE"
        else:
            self.npc.state = "PATROL"

        # --- Target & timer ---
        self.target.update(dt)
        if self.target.active:
            if self.target.check_collect(self.player.pos):
                self.target.active = False
                self.player.score += 1
            elif self.target.check_collect(self.npc.pos):
                self.target.active = False
                self.npc.hp = min(self.npc.max_hp, self.npc.hp + 25)

        self.round_timer -= dt

        # --- Win conditions ---
        if not self.player.alive:
            self.npc_score += 1
            self.state = GameState.NPC_WIN
        elif not self.npc.alive:
            self.player_score += 1
            self.state = GameState.PLAYER_WIN
        elif self.round_timer <= 0:
            self.state = GameState.DRAW

        # update particles
        self.particles = [p for p in self.particles if p.update(dt)]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Play against the trained RL NPC")
    parser.add_argument(
        "--model", default=None,
        help="Path to SB3 model file (without .zip). "
             "Auto-detects best available if omitted."
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic prediction (defaults to stochastic for more active combat)"
    )
    args = parser.parse_args()

    # Resolve model path
    if args.model:
        candidates = [args.model]
    else:
        candidates = MODEL_PRIORITY

    chosen = None
    for path in candidates:
        if os.path.exists(path + ".zip"):
            chosen = path
            break

    if chosen is None:
        print("ERROR: No trained model found. Checked:")
        for p in candidates:
            print(f"  - {p}.zip")
        print("\nRun 'python train.py' to train the model first.")
        sys.exit(1)

    # Warn if using smoke-test (untrained) model
    if "smoke_test" in chosen:
        print("=" * 55)
        print("  WARNING: Using smoke-test model (3 000 steps).")
        print("  NPC behaviour will look random/untrained.")
        print("  Run 'python train.py' for a properly trained NPC.")
        print("=" * 55)
    else:
        print(f"Model: {chosen}.zip")

    game = RLGame(model_path=chosen, deterministic=args.deterministic)
    try:
        game.run()
    except KeyboardInterrupt:
        pass  # Ctrl+C exits cleanly without traceback



if __name__ == "__main__":
    main()
