"""
env.py  --  Gymnasium environment wrapping the NPC tactical game.

The AGENT controls the NPC.
The PLAYER runs a simple scripted AI (chase + attack + retreat when low HP).

Observation (15-dim Box, all in [0,1] or [-1,1]):
  [0]  npc_x  / WORLD_W
  [1]  npc_y  / WORLD_H
  [2]  player_x / WORLD_W
  [3]  player_y / WORLD_H
  [4]  npc_hp  fraction
  [5]  player_hp fraction
  [6]  dist_to_player (normalised)
  [7]  sin(angle_to_player)
  [8]  cos(angle_to_player)
  [9]  dist_to_target (normalised)
  [10] sin(angle_to_target)
  [11] cos(angle_to_target)
  [12] player_alive  (0 or 1)
  [13] target_active (0 or 1)
  [14] player_attacking (0 or 1)

Action  MultiDiscrete([9, 3]):
  axis-0  movement: 0=stay, 1=N, 2=S, 3=W, 4=E, 5=NW, 6=NE, 7=SW, 8=SE
  axis-1  combat:   0=no,   1=attack, 2=defend

Reward shaping:
  +damage_dealt * 0.10  per HP dealt to player
  -damage_taken * 0.05  per HP taken from player
  +0.002 * dist_delta   proximity shaping (encourages approach)
  +3.0   collect target before player
  -0.5   player collects target first
  +15.0  terminal: NPC kills player
  -15.0  terminal: NPC dies
  +0.002 per-step survival bonus
"""

import os
import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from obstacles import world_obstacles, Target, resolve_circle_rect, SafeZone
from player   import Player
from npc      import NPC, ATTACK_RANGE as N_ATK_RANGE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORLD_W     = 1200
WORLD_H     = 720
MAX_DIST    = math.hypot(WORLD_W, WORLD_H)
SIM_DT      = 1.0 / 60.0        # fixed physics timestep (s)
STEP_LIMIT  = 5_400             # 90 s * 60 fps

NPC_RL_SPEED   = 190   # px/s when RL controls NPC
NPC_ATK_DMG_RL = 20    # NPC damage in RL env (boosted vs rule-based 12)
PLAYER_AI_SPD  = 190   # scripted player speed (slightly slower)
P_ATK_DMG_ENV  = 15    # player damage in env (reduced from 20)
N_ATK_CD       = 0.6   # NPC attack cooldown: 20 dmg / 0.6s = 33 DPS
P_ATK_CD_S     = 0.8   # player attack cooldown: 15 dmg / 0.8s = 18.75 DPS
P_ATK_RANGE    = 70    # scripted player attack range (px)
ATK_RANGE_BONUS = 0.15 # reward per step NPC spends within attack range
OBS_DIM        = 15

# Nine movement directions (index 0 = stay still)
MOVE_DIRS = [
    ( 0,  0),  # 0 stay
    ( 0, -1),  # 1 N
    ( 0,  1),  # 2 S
    (-1,  0),  # 3 W
    ( 1,  0),  # 4 E
    (-1, -1),  # 5 NW
    ( 1, -1),  # 6 NE
    (-1,  1),  # 7 SW
    ( 1,  1),  # 8 SE
]


# ---------------------------------------------------------------------------
# Shared observation builder (importable in play_rl.py)
# ---------------------------------------------------------------------------

def build_observation(npc, player, target) -> np.ndarray:
    """
    Build a 15-dim float32 observation vector.
    All values normalised; importable for play_rl.py consistency.
    """
    npc_x  = npc.pos.x   / WORLD_W
    npc_y  = npc.pos.y   / WORLD_H
    p_x    = player.pos.x / WORLD_W
    p_y    = player.pos.y / WORLD_H
    npc_hp = npc.hp    / npc.max_hp
    p_hp   = player.hp / player.max_hp

    dp      = player.pos - npc.pos
    dist_p  = dp.length()
    dn_p    = min(dist_p / MAX_DIST, 1.0)
    sin_p, cos_p = (dp.y / dist_p, dp.x / dist_p) if dist_p > 0 else (0.0, 0.0)

    dt_     = target.pos - npc.pos
    dist_t  = dt_.length()
    dn_t    = min(dist_t / MAX_DIST, 1.0)
    sin_t, cos_t = (dt_.y / dist_t, dt_.x / dist_t) if dist_t > 0 else (0.0, 0.0)

    p_attacking = 1.0 if getattr(player, "_atk_vis", 0) > 0 else 0.0

    return np.array([
        npc_x, npc_y,
        p_x,   p_y,
        npc_hp, p_hp,
        dn_p, sin_p, cos_p,
        dn_t, sin_t, cos_t,
        float(player.alive),
        float(target.active),
        p_attacking,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------

class NPCEnv(gym.Env):
    """Gymnasium environment: RL agent plays as NPC vs scripted player."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Set headless SDL before pygame.init() for non-render envs
        if render_mode is None:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        if not pygame.get_init():
            pygame.init()

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([9, 3])

        # Rendering state
        self._screen   = None
        self._clock    = None
        self._bg_surf  = None

        # Create entities
        self._make_entities()

    # ------------------------------------------------------------------
    def _make_entities(self):
        self.obstacles   = world_obstacles()
        self.target      = Target(600, 360)
        self.player      = Player(160, 360)
        self.npc         = NPC(1040, 360, WORLD_W, WORLD_H)
        self._step_count = 0
        self._last_p_hp  = self.player.max_hp
        self._last_n_hp  = self.npc.max_hp
        self._n_atk_cd   = 0.0
        self._p_atk_cd   = 0.0
        self._prev_dist  = self.npc.pos.distance_to(self.player.pos)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # Randomise spawns for training diversity
        self.npc.reset(
            float(rng.integers(850, 1150)),
            float(rng.integers(80, 640)),
        )
        self.player.reset(
            float(rng.integers(50, 350)),
            float(rng.integers(80, 640)),
        )
        self.target.reset(
            float(rng.integers(400, 800)),
            float(rng.integers(150, 570)),
        )

        self._step_count = 0
        self._last_p_hp  = self.player.max_hp
        self._last_n_hp  = self.npc.max_hp
        self._n_atk_cd   = 0.0
        self._p_atk_cd   = 0.0
        self._prev_dist  = self.npc.pos.distance_to(self.player.pos)

        obs  = build_observation(self.npc, self.player, self.target)
        info = {}
        return obs, info

    def step(self, action):
        move_idx  = int(action[0])
        combat_idx = int(action[1])
        do_attack = (combat_idx == 1)
        do_defend = (combat_idx == 2)

        # Guardrail: only block if player is alive and within FOV range (220px)
        dist_now = self.npc.pos.distance_to(self.player.pos)
        if dist_now > 220 or not self.player.alive:
            do_defend = False

        self.npc.is_defending = do_defend

        # --- Apply NPC movement ---
        dx, dy = MOVE_DIRS[move_idx]
        mv = pygame.math.Vector2(dx, dy)
        if mv.length() > 0:
            mv.normalize_ip()
            self.npc.angle = math.atan2(mv.y, mv.x)

        # Speed is reduced by 50% when defending
        speed = NPC_RL_SPEED * 0.5 if do_defend else NPC_RL_SPEED
        self.npc.pos += mv * speed * SIM_DT
        self.npc.pos.x = max(self.npc.radius,
                             min(self.npc.pos.x, WORLD_W - self.npc.radius))
        self.npc.pos.y = max(self.npc.radius,
                             min(self.npc.pos.y, WORLD_H - self.npc.radius))
        self.npc.pos = resolve_circle_rect(self.npc.pos, self.npc.radius,
                                          self.obstacles)

        # --- NPC attack (with cooldown) ---
        if self._n_atk_cd > 0:
            self._n_atk_cd -= SIM_DT
        npc_hit = False
        dist_now = self.npc.pos.distance_to(self.player.pos)
        if (do_attack and self._n_atk_cd <= 0
                and dist_now <= N_ATK_RANGE):
            self.player.take_damage(NPC_ATK_DMG_RL)
            self._n_atk_cd = N_ATK_CD
            npc_hit = True

        # --- Scripted player AI ---
        self._n_blocked = False
        self._step_player_ai()

        # --- Target collection ---
        tgt_npc = tgt_player = False
        if self.target.active:
            if self.target.check_collect(self.npc.pos):
                self.target.active = False
                self.npc.hp = min(self.npc.max_hp, self.npc.hp + 25)
                tgt_npc = True
            elif self.target.check_collect(self.player.pos):
                self.target.active = False
                self.player.hp = min(self.player.max_hp, self.player.hp + 15)
                tgt_player = True
        self.target.update(SIM_DT)

        # --- Reward ---
        reward = self._compute_reward(npc_hit, tgt_npc, tgt_player, do_defend)

        # --- Termination ---
        self._step_count += 1
        terminated = not self.npc.alive or not self.player.alive
        truncated  = self._step_count >= STEP_LIMIT

        # Update trackers
        self._last_p_hp  = self.player.hp
        self._last_n_hp  = self.npc.hp
        self._prev_dist  = (self.npc.pos.distance_to(self.player.pos)
                            if self.player.alive else MAX_DIST)

        obs  = build_observation(self.npc, self.player, self.target)
        info = {
            "npc_alive":    self.npc.alive,
            "player_alive": self.player.alive,
            "npc_hp":       self.npc.hp,
            "step":         self._step_count,
        }
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Scripted player
    # ------------------------------------------------------------------

    def _step_player_ai(self):
        """Simple chase-and-attack player with low-HP retreat."""
        if not self.player.alive:
            return
        if self._p_atk_cd > 0:
            self._p_atk_cd -= SIM_DT
        if not self.npc.alive:
            return

        to_npc = self.npc.pos - self.player.pos
        dist   = to_npc.length()
        hp_frac = self.player.hp / self.player.max_hp

        # Movement: chase, but retreat when low HP
        if hp_frac < 0.30 and dist < 250:
            mv = -to_npc.normalize() if dist > 0 else pygame.math.Vector2(0, 0)
        elif dist > 35:
            mv = to_npc.normalize()
        else:
            mv = pygame.math.Vector2(0, 0)

        self.player.pos += mv * PLAYER_AI_SPD * SIM_DT
        self.player.pos.x = max(self.player.radius,
                                min(self.player.pos.x, WORLD_W - self.player.radius))
        self.player.pos.y = max(self.player.radius,
                                min(self.player.pos.y, WORLD_H - self.player.radius))
        self.player.pos = resolve_circle_rect(self.player.pos, self.player.radius,
                                              self.obstacles)

        # Attack
        if dist <= P_ATK_RANGE and self._p_atk_cd <= 0:
            # Set player attack flash visual for observation state detection
            self.player._atk_vis = 0.15
            dmg = P_ATK_DMG_ENV
            if getattr(self.npc, "is_defending", False):
                dmg = int(dmg * 0.2)  # 80% damage reduction when defending
                self._n_blocked = True
            self.npc.take_damage(dmg)
            self._p_atk_cd = P_ATK_CD_S

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, npc_hit: bool, tgt_npc: bool,
                        tgt_player: bool, do_defend: bool = False) -> float:
        r = 0.0

        # --- Damage dealt / received ---
        dmg_dealt = self._last_p_hp - self.player.hp
        dmg_taken = self._last_n_hp - self.npc.hp
        if dmg_dealt > 0:
            r += dmg_dealt * 0.30   # tripled: strongly reward hurting the player
        if dmg_taken > 0:
            r -= dmg_taken * 0.05

        # --- Proximity shaping ---
        curr_dist = (self.npc.pos.distance_to(self.player.pos)
                     if self.player.alive else MAX_DIST)
        r += (self._prev_dist - curr_dist) * 0.003  # stronger pull toward player

        # --- In-range attack bonus (stay close and fight!) ---
        if self.player.alive and curr_dist <= N_ATK_RANGE:
            r += ATK_RANGE_BONUS

        # --- Defend reward shaping ---
        if getattr(self, "_n_blocked", False):
            r += 0.5  # Reward NPC for successfully blocking attacks (reduced from 2.0 to avoid turtling)
        elif do_defend:
            r -= 0.04 # Penalty for blocking when player is not attacking (increased from 0.02 to prevent turtling)

        # --- Target collection ---
        if tgt_npc:
            r += 3.0
        if tgt_player:
            r -= 0.5

        # --- Terminal outcomes ---
        if not self.player.alive:
            r += 20.0   # NPC wins: large bonus
        if not self.npc.alive:
            r -= 12.0   # slightly reduced so NPC isn't overly risk-averse

        # --- Per-step survival bonus (reward lasting longer) ---
        r += 0.002      # small but counters the previous avoidance strategy

        return r

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode != "human":
            return

        if self._screen is None:
            from obstacles import DARK_BG, GRID_LINE
            self._screen  = pygame.display.set_mode((WORLD_W, WORLD_H))
            self._clock   = pygame.time.Clock()
            pygame.display.set_caption("NPC RL Training")
            self._bg_surf = pygame.Surface((WORLD_W, WORLD_H))
            self._bg_surf.fill(DARK_BG)
            for gx in range(0, WORLD_W, 40):
                for gy in range(0, WORLD_H, 40):
                    pygame.draw.circle(self._bg_surf, GRID_LINE, (gx, gy), 1)
            SafeZone(40, 580, 160, 100).draw(self._bg_surf)
            for obs in self.obstacles:
                obs.draw(self._bg_surf)

        self._screen.blit(self._bg_surf, (0, 0))
        self.target.draw(self._screen)
        self.npc.draw(self._screen)
        self.player.draw(self._screen)

        # Step / HP overlay
        font = pygame.font.SysFont("consolas", 15)
        info = (f"Step {self._step_count}  "
                f"NPC HP {self.npc.hp}  Player HP {self.player.hp}  "
                f"Reward shaping active")
        surf = font.render(info, True, (180, 210, 240))
        self._screen.blit(surf, (8, 8))

        pygame.display.flip()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.close()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if pygame.get_init():
            pygame.quit()
