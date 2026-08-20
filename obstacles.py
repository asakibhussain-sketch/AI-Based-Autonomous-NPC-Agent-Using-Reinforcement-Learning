"""
obstacles.py  --  Static obstacle definitions and AABB collision helpers.
"""

import pygame
from typing import List


# ---------------------------------------------------------------------------
# Colour palette (shared across modules via import)
# ---------------------------------------------------------------------------
DARK_BG      = (10,  14,  22)
GRID_LINE    = (20,  28,  40)
OBSTACLE_COL = (35,  50,  70)
OBSTACLE_EDG = (60,  90, 120)
SAFE_COL     = (20,  60,  30)
SAFE_EDG     = (40, 180,  60)
TARGET_COL   = (180, 140,  20)
TARGET_EDG   = (255, 210,  50)
PLAYER_COL   = (0,  220, 255)   # cyan
NPC_COL      = (255,  50, 180)  # magenta
HP_BG        = (60,  20,  20)
HP_FG_P      = (50, 230, 120)
HP_FG_N      = (230,  80,  50)
TEXT_COL     = (200, 220, 240)
TEXT_DIM     = (100, 130, 160)
MINIMAP_BG   = (10,  16,  26, 200)


class Obstacle:
    """Axis-aligned bounding-box obstacle."""

    def __init__(self, x: int, y: int, w: int, h: int):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, OBSTACLE_COL, self.rect)
        pygame.draw.rect(surface, OBSTACLE_EDG, self.rect, 2)


class SafeZone:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surface: pygame.Surface) -> None:
        s = pygame.Surface((self.rect.w, self.rect.h), pygame.SRCALPHA)
        s.fill((40, 180, 60, 40))
        surface.blit(s, (self.rect.x, self.rect.y))
        pygame.draw.rect(surface, SAFE_EDG, self.rect, 2)


class Target:
    """Collectible objective that player or NPC can reach."""
    RADIUS = 14

    def __init__(self, x: int, y: int):
        self.pos  = pygame.math.Vector2(x, y)
        self.active = True
        self._pulse = 0.0

    def update(self, dt: float) -> None:
        self._pulse = (self._pulse + dt * 3.0) % (2 * 3.14159)

    def draw(self, surface: pygame.Surface) -> None:
        if not self.active:
            return
        import math
        r = int(self.RADIUS + 3 * abs(math.sin(self._pulse)))
        pygame.draw.circle(surface, TARGET_COL, (int(self.pos.x), int(self.pos.y)), r)
        pygame.draw.circle(surface, TARGET_EDG, (int(self.pos.x), int(self.pos.y)), r, 2)
        # inner glint
        pygame.draw.circle(surface, (255, 255, 200), (int(self.pos.x)-4, int(self.pos.y)-4), 4)

    def check_collect(self, entity_pos: pygame.math.Vector2) -> bool:
        if not self.active:
            return False
        return entity_pos.distance_to(self.pos) < self.RADIUS + 16

    def reset(self, x: int, y: int) -> None:
        self.pos   = pygame.math.Vector2(x, y)
        self.active = True


# ---------------------------------------------------------------------------
# Collision helpers
# ---------------------------------------------------------------------------

def circle_rect_collision(cx: float, cy: float, radius: float,
                           rect: pygame.Rect) -> bool:
    """True if circle overlaps AABB rect."""
    # nearest point on rect to circle centre
    nx = max(rect.left, min(cx, rect.right))
    ny = max(rect.top,  min(cy, rect.bottom))
    dx, dy = cx - nx, cy - ny
    return dx * dx + dy * dy < radius * radius


def resolve_circle_rect(pos: pygame.math.Vector2, radius: float,
                         obstacles: List[Obstacle]) -> pygame.math.Vector2:
    """Push pos out of any overlapping obstacle rects, return corrected pos."""
    for obs in obstacles:
        rect = obs.rect
        if circle_rect_collision(pos.x, pos.y, radius, rect):
            # find nearest point on rect
            nx = max(rect.left, min(pos.x, rect.right))
            ny = max(rect.top,  min(pos.y, rect.bottom))
            diff = pygame.math.Vector2(pos.x - nx, pos.y - ny)
            dist = diff.length()
            if dist < 0.001:
                diff = pygame.math.Vector2(1, 0)
                dist = 1.0
            overlap = radius - dist
            pos = pos + diff.normalize() * (overlap + 1)
    return pos


def world_obstacles() -> List[Obstacle]:
    """Return the fixed set of obstacles used for every round."""
    return [
        Obstacle(180, 160, 140,  30),   # top-left horizontal wall
        Obstacle(520, 110,  30, 200),   # upper-centre vertical pillar
        Obstacle(800, 250, 160,  30),   # right horizontal bar
        Obstacle(280, 440,  30, 180),   # lower-left vertical
        Obstacle(650, 480, 200,  30),   # lower-right horizontal
    ]


class Particle:
    def __init__(self, x: float, y: float, vx: float, vy: float, color, life: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.life = life
        self.max_life = life

    def update(self, dt: float) -> bool:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        return self.life > 0

    def draw(self, surface: pygame.Surface):
        alpha = int(255 * (self.life / self.max_life))
        color_with_alpha = (*self.color, alpha)
        size = max(1, int(4 * (self.life / self.max_life)))
        s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.circle(s, color_with_alpha, (size, size), size)
        surface.blit(s, (int(self.x - size), int(self.y - size)))

