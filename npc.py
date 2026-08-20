"""
npc.py  --  Rule-based NPC with a clean decide_action() interface.

States
------
  PATROL  : walk random waypoints
  CHASE   : move toward player when detected inside FOV
  ATTACK  : melee attack when within range
  RETREAT : move away from player when HP < RETREAT_THRESHOLD

The NPC's decision logic is fully isolated in decide_action() so it can
be swapped for a trained RL policy in Phase 3 without touching the rest
of the class.
"""

import pygame
import math
import random

from obstacles import resolve_circle_rect, NPC_COL, HP_BG, HP_FG_N


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
NPC_SPEED         = 160     # px/s patrol / retreat
NPC_CHASE_SPEED   = 200     # px/s while chasing
NPC_RADIUS        = 14
FOV_RADIUS        = 220     # detection distance
ATTACK_RANGE      = 75      # px - close enough to swing
ATTACK_STOP_DIST  = 42      # px - avoid jittering on top of the player
ATTACK_DAMAGE     = 8
ATTACK_COOLDOWN   = 1.0     # s
RETREAT_THRESHOLD = 0.30    # retreat when HP fraction falls below this
WAYPOINT_RADIUS   = 20      # how close counts as "reached waypoint"
TURN_LERP         = 0.22
MAX_HP            = 100

# State identifiers (also shown on HUD)
PATROL  = "PATROL"
CHASE   = "CHASE"
ATTACK  = "ATTACK"
RETREAT = "RETREAT"
DEFEND  = "DEFEND"


class NPC:
    """
    Rule-based NPC.  All AI decisions flow through decide_action();
    the rest of the class is physics, rendering and bookkeeping.
    """

    def __init__(self, x: float, y: float, world_w: int, world_h: int):
        self.pos        = pygame.math.Vector2(x, y)
        self.angle      = 0.0
        self.hp         = MAX_HP
        self.max_hp     = MAX_HP
        self.alive      = True
        self.radius     = NPC_RADIUS
        self.state      = PATROL
        self.world_w    = world_w
        self.world_h    = world_h
        self._atk_cd    = 0.0
        self._atk_vis   = 0.0
        self._atk_ready_to_hit = False
        self._waypoint  = self._random_waypoint()
        self._retreat_target = pygame.math.Vector2(x, y)
        self.is_defending = False
        self._patrol_timer = 0.0

    # ------------------------------------------------------------------
    # Core AI interface - swap this out for an RL policy in Phase 3
    # ------------------------------------------------------------------

    def decide_action(
        self,
        player_pos: pygame.math.Vector2,
        player_alive: bool,
        player_is_attacking: bool = False,
    ) -> tuple[pygame.math.Vector2, bool, bool]:
        """
        Pure decision function.  Returns:
          move_dir  : normalised Vector2 (or zero) - desired movement direction
          do_attack : bool - whether to attempt an attack this frame
          do_defend : bool - whether to block this frame
        """
        hp_frac = self.hp / self.max_hp
        dist    = self.pos.distance_to(player_pos) if player_alive else float('inf')

        # --- State transitions ---
        if not player_alive:
            self.state = PATROL

        elif hp_frac < RETREAT_THRESHOLD:
            self.state = RETREAT

        elif player_is_attacking and dist <= ATTACK_RANGE + 40:
            self.state = DEFEND

        elif dist <= ATTACK_RANGE:
            self.state = ATTACK

        elif dist <= FOV_RADIUS:
            self.state = CHASE

        else:
            self.state = PATROL

        # --- Compute desired direction ---
        move_dir   = pygame.math.Vector2(0, 0)
        do_attack  = False
        do_defend  = False

        if self.state == PATROL:
            move_dir = self._patrol_direction()

        elif self.state == CHASE:
            toward = player_pos - self.pos
            if toward.length_squared() > 0.001:
                move_dir = toward.normalize()

        elif self.state == ATTACK:
            do_attack = (self._atk_cd <= 0)
            toward = player_pos - self.pos
            if toward.length() > ATTACK_STOP_DIST:
                move_dir = toward.normalize()

        elif self.state == DEFEND:
            do_defend = True
            # retreat slightly when defending to create space
            away = self.pos - player_pos
            if away.length_squared() > 0.001:
                move_dir = away.normalize()

        elif self.state == RETREAT:
            # update retreat target only when close to it
            away = self.pos - player_pos
            if away.length_squared() > 0.001:
                move_dir = away.normalize()
            # also try to reach safe corner
            retreat_goal = pygame.math.Vector2(
                self.world_w - 100 if player_pos.x < self.world_w / 2 else 100,
                self.world_h - 100 if player_pos.y < self.world_h / 2 else 100,
            )
            blend = (retreat_goal - self.pos)
            if blend.length_squared() > 0.001:
                blended = move_dir + blend.normalize() * 0.4
                if blended.length_squared() > 0.001:
                    move_dir = blended.normalize()

        return move_dir, do_attack, do_defend

    # ------------------------------------------------------------------
    # Update (physics + timers, calls decide_action)
    # ------------------------------------------------------------------

    def update(self, dt: float, player_pos: pygame.math.Vector2,
               player_alive: bool, obstacles, player_is_attacking: bool = False) -> None:
        if not self.alive:
            return

        move_dir, do_attack, do_defend = self.decide_action(player_pos, player_alive, player_is_attacking)
        self.is_defending = do_defend

        # Choose speed based on state
        if self.state == DEFEND:
            speed = NPC_SPEED * 0.5  # slower movement while defending
        elif self.state in (CHASE, ATTACK):
            speed = NPC_CHASE_SPEED
        else:
            speed = NPC_SPEED

        # Stuck detection: reset patrol waypoint if stuck on a wall/obstacle for 5 seconds
        if self.state == PATROL:
            self._patrol_timer += dt
            if self._patrol_timer > 5.0:
                self._waypoint = self._random_waypoint()
                self._patrol_timer = 0.0
        else:
            self._patrol_timer = 0.0

        if move_dir.length_squared() > 0.001:
            self.pos  += move_dir * speed * dt

        # Face target angle: face player when close/combatting, face movement direction otherwise
        dist_to_player = self.pos.distance_to(player_pos) if player_alive else float('inf')
        if dist_to_player <= FOV_RADIUS and player_alive and dist_to_player > 1.0:
            target_angle = math.atan2(player_pos.y - self.pos.y,
                                      player_pos.x - self.pos.x)
        elif move_dir.length_squared() > 0.001:
            target_angle = math.atan2(move_dir.y, move_dir.x)
        else:
            target_angle = self.angle

        # Smooth turn interpolation (lerp) to prevent jittering/instant snaps
        diff_angle = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        self.angle += diff_angle * TURN_LERP

        # clamp to world bounds
        self.pos.x = max(self.radius, min(self.pos.x, self.world_w - self.radius))
        self.pos.y = max(self.radius, min(self.pos.y, self.world_h - self.radius))

        # obstacle collision
        self.pos = resolve_circle_rect(self.pos, self.radius, obstacles)

        # attack timer
        if self._atk_cd  > 0: self._atk_cd  -= dt
        if self._atk_vis > 0: self._atk_vis -= dt

        if do_attack:
            self._atk_cd  = ATTACK_COOLDOWN
            self._atk_vis = 0.15
            self._atk_ready_to_hit = True

    # ------------------------------------------------------------------
    # Helper: patrol waypoint logic
    # ------------------------------------------------------------------

    def _patrol_direction(self) -> pygame.math.Vector2:
        """Move toward current waypoint; pick a new one when reached."""
        to_wp = self._waypoint - self.pos
        if to_wp.length() < WAYPOINT_RADIUS:
            self._waypoint = self._random_waypoint()
            self._patrol_timer = 0.0
            to_wp = self._waypoint - self.pos
        if to_wp.length() > 0:
            return to_wp.normalize()
        return pygame.math.Vector2(0, 0)

    def _random_waypoint(self) -> pygame.math.Vector2:
        margin = 80
        return pygame.math.Vector2(
            random.randint(margin, self.world_w - margin),
            random.randint(margin, self.world_h - margin),
        )

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    def attack_hits(self, target_pos: pygame.math.Vector2) -> bool:
        if not self._atk_ready_to_hit or self._atk_vis <= 0:
            return False
        if self.pos.distance_to(target_pos) <= ATTACK_RANGE:
            self._atk_ready_to_hit = False
            return True
        return False

    def take_damage(self, dmg: int) -> None:
        self.hp = max(0, self.hp - dmg)
        if self.hp == 0:
            self.alive = False

    def reset(self, x: float, y: float) -> None:
        self.pos       = pygame.math.Vector2(x, y)
        self.angle     = 0.0
        self.hp        = MAX_HP
        self._atk_cd   = 0.0
        self._atk_vis  = 0.0
        self._atk_ready_to_hit = False
        self.alive     = True
        self.state     = PATROL
        self._waypoint = self._random_waypoint()
        self.is_defending = False
        self._patrol_timer = 0.0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        if not self.alive:
            return
        px, py = int(self.pos.x), int(self.pos.y)

        # Draw visual block/shield if defending
        if self.is_defending:
            shield_surf = pygame.Surface((self.radius*4, self.radius*4), pygame.SRCALPHA)
            pygame.draw.circle(shield_surf, (0, 180, 255, 110), (self.radius*2, self.radius*2), int(self.radius * 1.5))
            pygame.draw.circle(shield_surf, (0, 220, 255, 255), (self.radius*2, self.radius*2), int(self.radius * 1.5), 2)
            surface.blit(shield_surf, (px - self.radius*2, py - self.radius*2))

        # FOV circle (translucent)
        self._draw_fov(surface, px, py)

        # attack flash
        if self._atk_vis > 0:
            arc_surf = pygame.Surface((ATTACK_RANGE*2+4, ATTACK_RANGE*2+4), pygame.SRCALPHA)
            pygame.draw.circle(arc_surf, (*NPC_COL, 100),
                               (ATTACK_RANGE+2, ATTACK_RANGE+2), ATTACK_RANGE)
            surface.blit(arc_surf, (px - ATTACK_RANGE - 2, py - ATTACK_RANGE - 2))

        # glow halo
        halo = pygame.Surface((80, 80), pygame.SRCALPHA)
        pygame.draw.circle(halo, (*NPC_COL, 25), (40, 40), 36)
        surface.blit(halo, (px - 40, py - 40))

        # body
        pygame.draw.circle(surface, NPC_COL, (px, py), self.radius)

        # direction pip
        tip = self.pos + pygame.math.Vector2(
            math.cos(self.angle) * self.radius,
            math.sin(self.angle) * self.radius,
        )
        pygame.draw.line(surface, (255, 255, 255), (px, py),
                         (int(tip.x), int(tip.y)), 3)
        pygame.draw.circle(surface, (255, 255, 255), (px, py), self.radius, 1)

        # health bar
        self._draw_hp_bar(surface, px, py)

    def _draw_fov(self, surface: pygame.Surface, px: int, py: int) -> None:
        """Draw translucent FOV circle."""
        fov_surf = pygame.Surface((FOV_RADIUS*2, FOV_RADIUS*2), pygame.SRCALPHA)
        # colour shifts with state
        col = {
            PATROL:  (100, 100, 255,  22),
            CHASE:   (255, 200,  50,  35),
            ATTACK:  (255,  80,  80,  50),
            RETREAT: (50,  255, 100,  28),
            DEFEND:  (0,   180, 255,  38),
        }.get(self.state, (100, 100, 255, 22))
        pygame.draw.circle(fov_surf, col, (FOV_RADIUS, FOV_RADIUS), FOV_RADIUS)
        surface.blit(fov_surf, (px - FOV_RADIUS, py - FOV_RADIUS))
        # ring
        edge_col = {
            PATROL:  (100, 100, 255),
            CHASE:   (255, 200,  50),
            ATTACK:  (255,  80,  80),
            RETREAT: (50,  255, 100),
            DEFEND:  (0,   180, 255),
        }.get(self.state, (100, 100, 255))
        pygame.draw.circle(surface, edge_col, (px, py), FOV_RADIUS, 1)

    def _draw_hp_bar(self, surface, px, py):
        bw, bh = 40, 5
        bx, by = px - bw // 2, py - self.radius - 10
        pygame.draw.rect(surface, HP_BG,   (bx, by, bw, bh))
        fill = int(bw * self.hp / self.max_hp)
        if fill > 0:
            pygame.draw.rect(surface, HP_FG_N, (bx, by, fill, bh))
        pygame.draw.rect(surface, (200, 200, 200), (bx, by, bw, bh), 1)

