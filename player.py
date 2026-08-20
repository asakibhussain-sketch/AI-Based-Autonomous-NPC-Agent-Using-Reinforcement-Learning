"""
player.py  --  Human-controlled player character.
"""

import pygame
import math
from obstacles import resolve_circle_rect, PLAYER_COL, HP_BG, HP_FG_P


PLAYER_SPEED  = 220          # px / s
PLAYER_RADIUS = 14
ATTACK_RANGE  = 70           # px
ATTACK_DAMAGE = 20
ATTACK_COOLDOWN = 0.5        # s
MAX_HP        = 100


class Player:
    """
    Handles input, movement, collision, attack and rendering for the
    human-controlled character.  Rendering is visually separate from
    decide/update logic so it can later be stripped for headless training.
    """

    def __init__(self, x: float, y: float):
        self.pos      = pygame.math.Vector2(x, y)
        self.vel      = pygame.math.Vector2(0, 0)
        self.angle    = 0.0          # radians, facing direction
        self.hp       = MAX_HP
        self.max_hp   = MAX_HP
        self._atk_cd  = 0.0         # attack cooldown timer
        self._atk_vis = 0.0         # visual flash timer
        self.alive    = True
        self.radius   = PLAYER_RADIUS
        self.score    = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt: float, keys, obstacles) -> None:
        if not self.alive:
            return

        # --- movement ---
        move = pygame.math.Vector2(0, 0)
        if keys[pygame.K_w] or keys[pygame.K_UP]:    move.y -= 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:  move.y += 1
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  move.x -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: move.x += 1

        if move.length() > 0:
            move.normalize_ip()
            self.angle = math.atan2(move.y, move.x)

        self.pos += move * PLAYER_SPEED * dt

        # clamp to world bounds
        self.pos.x = max(self.radius, min(self.pos.x, 1200 - self.radius))
        self.pos.y = max(self.radius, min(self.pos.y, 720  - self.radius))

        # obstacle collision
        self.pos = resolve_circle_rect(self.pos, self.radius, obstacles)

        # cooldown timers
        if self._atk_cd  > 0: self._atk_cd  -= dt
        if self._atk_vis > 0: self._atk_vis -= dt

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def try_attack(self) -> bool:
        """Return True and trigger attack if cooldown allows."""
        if self._atk_cd <= 0:
            self._atk_cd  = ATTACK_COOLDOWN
            self._atk_vis = 0.15
            return True
        return False

    def attack_hit_point(self) -> pygame.math.Vector2:
        """Centre of the attack arc - a point ahead of the player."""
        return self.pos + pygame.math.Vector2(
            math.cos(self.angle) * ATTACK_RANGE * 0.7,
            math.sin(self.angle) * ATTACK_RANGE * 0.7,
        )

    def attack_hits(self, target_pos: pygame.math.Vector2) -> bool:
        """True if target is within attack arc."""
        return self.pos.distance_to(target_pos) <= ATTACK_RANGE

    def take_damage(self, dmg: int) -> None:
        self.hp = max(0, self.hp - dmg)
        if self.hp == 0:
            self.alive = False

    def reset(self, x: float, y: float) -> None:
        self.pos     = pygame.math.Vector2(x, y)
        self.vel     = pygame.math.Vector2(0, 0)
        self.angle   = 0.0
        self.hp      = MAX_HP
        self._atk_cd = 0.0
        self._atk_vis= 0.0
        self.alive   = True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        if not self.alive:
            return
        px, py = int(self.pos.x), int(self.pos.y)

        # glow halo
        halo = pygame.Surface((80, 80), pygame.SRCALPHA)
        pygame.draw.circle(halo, (*PLAYER_COL, 30), (40, 40), 36)
        surface.blit(halo, (px - 40, py - 40))

        # attack flash arc
        if self._atk_vis > 0:
            arc_surf = pygame.Surface((ATTACK_RANGE*2+4, ATTACK_RANGE*2+4), pygame.SRCALPHA)
            pygame.draw.circle(arc_surf, (*PLAYER_COL, 90),
                               (ATTACK_RANGE+2, ATTACK_RANGE+2), ATTACK_RANGE)
            surface.blit(arc_surf, (px - ATTACK_RANGE - 2, py - ATTACK_RANGE - 2))

        # body
        pygame.draw.circle(surface, PLAYER_COL, (px, py), self.radius)
        # direction pip
        tip = self.pos + pygame.math.Vector2(
            math.cos(self.angle) * self.radius,
            math.sin(self.angle) * self.radius,
        )
        pygame.draw.line(surface, (255, 255, 255), (px, py),
                         (int(tip.x), int(tip.y)), 3)
        # outline
        pygame.draw.circle(surface, (255, 255, 255), (px, py), self.radius, 1)

        # health bar
        self._draw_hp_bar(surface, px, py)

    def _draw_hp_bar(self, surface, px, py):
        bw, bh = 40, 5
        bx, by = px - bw // 2, py - self.radius - 10
        pygame.draw.rect(surface, HP_BG,   (bx, by, bw, bh))
        fill = int(bw * self.hp / self.max_hp)
        if fill > 0:
            pygame.draw.rect(surface, HP_FG_P, (bx, by, fill, bh))
        pygame.draw.rect(surface, (200, 200, 200), (bx, by, bw, bh), 1)
