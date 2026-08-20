"""
game.py  --  Main game loop, round management, and event dispatch.

Round lifecycle
---------------
  PLAYING  -> player dies   -> NPC_WINS  -> wait R key
  PLAYING  -> NPC dies      -> PLAYER_WINS-> wait R key
  PLAYING  -> timer expires -> DRAW       -> wait R key
  Any end state + R key     -> reset_round()
"""

import sys
import math
import pygame

from player    import Player, ATTACK_DAMAGE as P_ATTACK_DMG
from npc       import NPC,    ATTACK_DAMAGE as N_ATTACK_DMG
from obstacles import (SafeZone, Target, Particle, world_obstacles,
                       DARK_BG, GRID_LINE, PLAYER_COL, NPC_COL)
from hud       import HUD

# ---------------------------------------------------------------------------
# World / window constants
# ---------------------------------------------------------------------------
SCREEN_W  = 1200
SCREEN_H  = 720
FPS       = 60
ROUND_SEC = 90   # seconds per round before draw

# Spawn points
PLAYER_SPAWN = (160,  360)
NPC_SPAWN    = (1040, 360)
TARGET_POS   = (600,  360)
SAFE_RECT    = (40, 580, 160, 100)


class GameState:
    PLAYING   = "playing"
    PLAYER_WIN = "player_wins"
    NPC_WIN    = "npc_wins"
    DRAW       = "draw"


class Game:
    """Encapsulates everything needed to run one session."""

    def __init__(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("AI-Based Autonomous NPC Agent - Tactical Arena")
        self.clock   = pygame.time.Clock()

        # World entities
        self.obstacles = world_obstacles()
        self.safe_zone = SafeZone(*SAFE_RECT)
        self.target    = Target(*TARGET_POS)

        # Characters
        self.player = Player(*PLAYER_SPAWN)
        self.npc    = NPC(*NPC_SPAWN, SCREEN_W, SCREEN_H)

        # HUD
        self.hud = HUD(SCREEN_W, SCREEN_H, SCREEN_W, SCREEN_H)
        self.particles = []

        # Session state
        self.player_score = 0
        self.npc_score    = 0
        self.state        = GameState.PLAYING
        self.round_timer  = float(ROUND_SEC)
        self._bg_surf     = None  # cached background

        self._build_bg()

    # ------------------------------------------------------------------
    # Pre-render static background
    # ------------------------------------------------------------------

    def _build_bg(self) -> None:
        self._bg_surf = pygame.Surface((SCREEN_W, SCREEN_H))
        self._bg_surf.fill(DARK_BG)

        # subtle dot-grid
        for gx in range(0, SCREEN_W, 40):
            for gy in range(0, SCREEN_H, 40):
                pygame.draw.circle(self._bg_surf, GRID_LINE, (gx, gy), 1)

        # draw static elements
        self.safe_zone.draw(self._bg_surf)
        for obs in self.obstacles:
            obs.draw(self._bg_surf)

    # ------------------------------------------------------------------
    # Round reset
    # ------------------------------------------------------------------

    def reset_round(self) -> None:
        self.player.reset(*PLAYER_SPAWN)
        self.npc.reset(*NPC_SPAWN)
        self.target.reset(*TARGET_POS)
        self.state       = GameState.PLAYING
        self.round_timer = float(ROUND_SEC)
        self.particles.clear()

    def spawn_hit_sparks(self, pos: pygame.math.Vector2, color) -> None:
        import random
        for _ in range(12):
            angle = random.uniform(0, 2 * 3.14159)
            speed = random.uniform(80, 200)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = random.uniform(0.15, 0.4)
            self.particles.append(Particle(pos.x, pos.y, vx, vy, color, life))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 0.05)   # cap dt to avoid spiral of death

            running = self._handle_events()
            self._update(dt)
            self._draw()

        pygame.quit()
        sys.exit()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self.reset_round()
                if event.key in (pygame.K_SPACE,) and self.state == GameState.PLAYING:
                    self._player_attack()
        return True

    def _player_attack(self) -> None:
        if self.player.try_attack() and self.npc.alive:
            # Auto-aim: face the player towards the NPC when attacking
            to_npc = self.npc.pos - self.player.pos
            if to_npc.length() > 0:
                self.player.angle = math.atan2(to_npc.y, to_npc.x)

            if self.player.attack_hits(self.npc.pos):
                dmg = P_ATTACK_DMG
                is_defending = getattr(self.npc, "is_defending", False)
                if is_defending:
                    dmg = int(dmg * 0.2)  # 80% damage reduction if blocking/defending
                self.npc.take_damage(dmg)
                spark_col = (0, 180, 255) if is_defending else PLAYER_COL
                self.spawn_hit_sparks(self.npc.pos, spark_col)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _update(self, dt: float) -> None:
        if self.state != GameState.PLAYING:
            return

        keys = pygame.key.get_pressed()

        # update entities
        self.player.update(dt, keys, self.obstacles)
        self.npc.update(dt, self.player.pos, self.player.alive, self.obstacles, (self.player._atk_vis > 0))
        self.target.update(dt)

        # NPC auto-attack player
        if self.npc.attack_hits(self.player.pos) and self.player.alive:
            self.player.take_damage(N_ATTACK_DMG)
            self.spawn_hit_sparks(self.player.pos, NPC_COL)

        # update particles
        self.particles = [p for p in self.particles if p.update(dt)]

        # target collection
        if self.target.active:
            if self.target.check_collect(self.player.pos):
                self.target.active = False
                self.player.score += 1
            elif self.target.check_collect(self.npc.pos):
                self.target.active = False
                self.npc.hp = min(self.npc.max_hp, self.npc.hp + 20)  # NPC heals on collect

        # round timer
        self.round_timer -= dt

        # check win conditions
        if not self.player.alive:
            self.npc_score   += 1
            self.state = GameState.NPC_WIN
        elif not self.npc.alive:
            self.player_score += 1
            self.state = GameState.PLAYER_WIN
        elif self.round_timer <= 0:
            self.state = GameState.DRAW

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        # static background
        self.screen.blit(self._bg_surf, (0, 0))

        # dynamic world objects
        self.target.draw(self.screen)

        # characters (NPC FOV drawn first so body is on top)
        self.npc.draw(self.screen)
        self.player.draw(self.screen)

        # particles
        for p in self.particles:
            p.draw(self.screen)

        # message for non-playing states
        msg = ""
        if self.state == GameState.PLAYER_WIN:
            msg = "PLAYER WINS!"
        elif self.state == GameState.NPC_WIN:
            msg = "NPC WINS!"
        elif self.state == GameState.DRAW:
            msg = "DRAW - Time Up"

        is_rl = hasattr(self, "rl_model")
        self.hud.draw(
            self.screen, self.player, self.npc,
            self.round_timer, self.player_score, self.npc_score,
            message=msg,
            is_rl=is_rl,
            obstacles=self.obstacles,
            target=self.target,
        )

        pygame.display.flip()

