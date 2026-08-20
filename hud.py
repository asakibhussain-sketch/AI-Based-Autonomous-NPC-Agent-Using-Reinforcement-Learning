"""
hud.py  --  HUD overlay and minimap rendering.
All drawing here reads state but never mutates it.
"""

import pygame
from obstacles import TEXT_COL, TEXT_DIM, PLAYER_COL, NPC_COL, TARGET_COL

# Minimap geometry
MM_W, MM_H = 190, 126
MM_MARGIN = 16
MM_ALPHA = 225

PANEL_BG = (9, 15, 25, 218)
PANEL_EDGE = (54, 86, 118, 210)
PANEL_EDGE_SOFT = (33, 52, 76, 180)
GOOD_COL = (60, 235, 150)
WARN_COL = (255, 195, 75)
DANGER_COL = (255, 82, 96)
SHIELD_COL = (0, 190, 255)


class HUD:
    """
    Renders:
      - Top-left player HP bar + label
      - Top-right NPC HP bar + state label
      - Centre-top round timer
      - Top-centre round score
      - Bottom minimap
      - Message overlay (round-end result)
    """

    def __init__(self, screen_w: int, screen_h: int,
                 world_w: int, world_h: int):
        self.sw      = screen_w
        self.sh      = screen_h
        self.ww      = world_w
        self.wh      = world_h
        self._fonts_ready = False

    # ------------------------------------------------------------------
    # Font initialisation (must happen after pygame.init())
    # ------------------------------------------------------------------

    def _init_fonts(self) -> None:
        if self._fonts_ready:
            return
        self.font_xs = pygame.font.SysFont("consolas", 12, bold=False)
        self.font_sm = pygame.font.SysFont("consolas", 14, bold=False)
        self.font_md = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_lg = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_xl = pygame.font.SysFont("consolas", 48, bold=True)
        self._fonts_ready = True

    # ------------------------------------------------------------------
    # Main draw call
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface, player, npc,
             round_timer: float, player_score: int, npc_score: int,
             message: str = "", is_rl: bool = False, obstacles=None,
             target=None) -> None:
        self._init_fonts()

        # Glassy top command bar
        bar = pygame.Surface((self.sw, 60), pygame.SRCALPHA)
        bar.fill((7, 11, 19, 222))
        surface.blit(bar, (0, 0))
        pygame.draw.line(surface, (56, 92, 126), (0, 60), (self.sw, 60), 1)
        pygame.draw.line(surface, (20, 32, 48), (0, 61), (self.sw, 61), 1)

        self._draw_player_hud(surface, player)
        self._draw_npc_hud(surface, npc)
        self._draw_timer(surface, round_timer)
        self._draw_score(surface, player_score, npc_score)
        self._draw_minimap(surface, player, npc, obstacles, target)
        self._draw_ai_panel(surface, npc, is_rl)
        self._draw_controls(surface)

        if message:
            self._draw_message(surface, message)

    # ------------------------------------------------------------------
    # Sub-renders
    # ------------------------------------------------------------------

    def _draw_player_hud(self, surface, player):
        x, y = 18, 10
        lbl = self.font_sm.render("PLAYER", True, PLAYER_COL)
        surface.blit(lbl, (x, y))
        self._draw_health_bar(surface, x, y + 22, 190, player.hp, player.max_hp)
        hp_txt = self.font_sm.render(f"{player.hp:03d}/{player.max_hp}", True, TEXT_COL)
        surface.blit(hp_txt, (x + 202, y + 20))

    def _draw_npc_hud(self, surface, npc):
        bw = 190
        x  = self.sw - 16 - bw
        y  = 10
        state_col = {
            "PATROL":  (100, 140, 255),
            "CHASE":   (255, 200,  50),
            "ATTACK":  (255,  60,  60),
            "RETREAT": (50,  220, 100),
            "DEFEND":  (0,   180, 255),
        }.get(npc.state, TEXT_DIM)
        lbl = self.font_sm.render(f"NPC  {npc.state}", True, state_col)
        surface.blit(lbl, (x, y))
        self._draw_health_bar(surface, x, y + 22, bw, npc.hp, npc.max_hp)
        hp_txt = self.font_sm.render(f"{npc.hp:03d}/{npc.max_hp}", True, TEXT_COL)
        surface.blit(hp_txt, (x - 70, y + 20))

    def _draw_timer(self, surface, t: float):
        mins = int(t) // 60
        secs = int(t) % 60
        color = DANGER_COL if t <= 10 else (WARN_COL if t <= 25 else TEXT_COL)
        txt = self.font_lg.render(f"{mins:02d}:{secs:02d}", True, color)
        x = self.sw // 2 - txt.get_width() // 2
        surface.blit(txt, (x, 6))

    def _draw_score(self, surface, ps: int, ns: int):
        txt = self.font_sm.render(f"PLAYER {ps}  /  NPC {ns}", True, TEXT_DIM)
        surface.blit(txt, (self.sw // 2 - txt.get_width() // 2, 40))

    def _draw_minimap(self, surface, player, npc, obstacles=None, target=None):
        mm_x = self.sw  - MM_W - MM_MARGIN
        mm_y = self.sh  - MM_H - MM_MARGIN

        mm_surf = pygame.Surface((MM_W, MM_H), pygame.SRCALPHA)
        mm_surf.fill((8, 14, 24, MM_ALPHA))
        pygame.draw.rect(mm_surf, PANEL_EDGE, (0, 0, MM_W, MM_H), 1, border_radius=6)
        pygame.draw.rect(mm_surf, PANEL_EDGE_SOFT, (5, 5, MM_W - 10, MM_H - 10), 1)

        sx = MM_W / self.ww
        sy = MM_H / self.wh

        # draw obstacles on minimap
        if obstacles:
            for obs in obstacles:
                ox = int(obs.rect.x * sx)
                oy = int(obs.rect.y * sy)
                ow = int(obs.rect.w * sx)
                oh = int(obs.rect.h * sy)
                pygame.draw.rect(mm_surf, (30, 45, 60), (ox, oy, ow, oh))
                pygame.draw.rect(mm_surf, (50, 75, 100), (ox, oy, ow, oh), 1)

        # draw safe zone boundary on minimap
        sz_x = int(40 * sx)
        sz_y = int(580 * sy)
        sz_w = int(160 * sx)
        sz_h = int(100 * sy)
        pygame.draw.rect(mm_surf, (40, 180, 60, 40), (sz_x, sz_y, sz_w, sz_h))
        pygame.draw.rect(mm_surf, (40, 180, 60, 150), (sz_x, sz_y, sz_w, sz_h), 1)

        if target is not None and getattr(target, "active", False):
            tx = int(target.pos.x * sx)
            ty = int(target.pos.y * sy)
            pygame.draw.circle(mm_surf, TARGET_COL, (tx, ty), 4)
            pygame.draw.circle(mm_surf, (255, 230, 120), (tx, ty), 6, 1)

        # player dot
        if player.alive:
            mx = int(player.pos.x * sx)
            my = int(player.pos.y * sy)
            pygame.draw.circle(mm_surf, PLAYER_COL, (mx, my), 4)

        # npc dot
        if npc.alive:
            mx = int(npc.pos.x * sx)
            my = int(npc.pos.y * sy)
            pygame.draw.circle(mm_surf, NPC_COL, (mx, my), 4)

        surface.blit(mm_surf, (mm_x, mm_y))

        lbl = self.font_sm.render("TACTICAL MAP", True, TEXT_DIM)
        surface.blit(lbl, (mm_x + 2, mm_y - 18))

    def _draw_ai_panel(self, surface, npc, is_rl: bool = False):
        # Position: bottom-left
        panel_x = MM_MARGIN
        panel_y = self.sh - MM_H - MM_MARGIN
        panel_w = 288
        panel_h = MM_H

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        pygame.draw.rect(panel, PANEL_EDGE, (0, 0, panel_w, panel_h), 1, border_radius=6)
        pygame.draw.rect(panel, PANEL_EDGE_SOFT, (5, 5, panel_w - 10, panel_h - 10), 1)

        mode_str = "RL POLICY AGENT" if is_rl else "RULE-BASED AGENT"
        title_col = SHIELD_COL if is_rl else WARN_COL
        lbl_title = self.font_md.render(mode_str, True, title_col)
        panel.blit(lbl_title, (12, 10))

        # Divider line
        pygame.draw.line(panel, (40, 70, 100, 100), (12, 32), (panel_w - 12, 32), 1)

        state_col = {
            "PATROL":  (100, 140, 255),
            "CHASE":   (255, 200,  50),
            "ATTACK":  (255,  60,  60),
            "RETREAT": (50,  220, 100),
            "DEFEND":  (0,   180, 255),
        }.get(npc.state, TEXT_DIM)
        lbl_state = self.font_sm.render(f"State    {npc.state}", True, state_col)
        panel.blit(lbl_state, (12, 40))

        atk_status = "SWINGING" if npc._atk_vis > 0 else "READY"
        atk_col = DANGER_COL if npc._atk_vis > 0 else TEXT_DIM
        lbl_atk = self.font_sm.render(f"Attack   {atk_status}", True, atk_col)
        panel.blit(lbl_atk, (12, 60))

        def_status = "BLOCKING" if getattr(npc, "is_defending", False) else "READY"
        def_col = SHIELD_COL if getattr(npc, "is_defending", False) else TEXT_DIM
        lbl_def = self.font_sm.render(f"Defend   {def_status}", True, def_col)
        panel.blit(lbl_def, (12, 80))

        hp_frac = max(0.0, npc.hp / npc.max_hp)
        threat = "CRITICAL" if hp_frac < 0.25 else ("WATCH" if hp_frac < 0.55 else "STABLE")
        threat_col = DANGER_COL if hp_frac < 0.25 else (WARN_COL if hp_frac < 0.55 else GOOD_COL)
        lbl_threat = self.font_sm.render(f"Status   {threat}", True, threat_col)
        panel.blit(lbl_threat, (12, 100))

        surface.blit(panel, (panel_x, panel_y))

        lbl_hud = self.font_sm.render("AI TELEMETRY", True, TEXT_DIM)
        surface.blit(lbl_hud, (panel_x + 4, panel_y - 16))

    def _draw_controls(self, surface):
        hints = [("WASD/ARROWS", "MOVE"), ("SPACE", "ATTACK"), ("R", "RESET"), ("ESC", "QUIT")]
        x = self.sw // 2
        y = self.sh - 30
        rendered = []
        total_w = 0
        for key, action in hints:
            key_surf = self.font_xs.render(key, True, TEXT_COL)
            action_surf = self.font_xs.render(action, True, TEXT_DIM)
            width = key_surf.get_width() + action_surf.get_width() + 22
            rendered.append((key_surf, action_surf, width))
            total_w += width + 8
        total_w -= 8
        cursor = x - total_w // 2
        for key_surf, action_surf, width in rendered:
            rect = pygame.Rect(cursor, y, width, 20)
            pygame.draw.rect(surface, (8, 14, 24, 190), rect, border_radius=5)
            pygame.draw.rect(surface, (42, 68, 96), rect, 1, border_radius=5)
            surface.blit(key_surf, (cursor + 8, y + 4))
            surface.blit(action_surf, (cursor + width - action_surf.get_width() - 8, y + 4))
            cursor += width + 8

    def _draw_message(self, surface, message: str):
        """Big centred message (round result)."""
        overlay = pygame.Surface((self.sw, self.sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 165))
        surface.blit(overlay, (0, 0))

        # parse colour hint from leading token
        col = TEXT_COL
        if message.startswith("PLAYER"):
            col = PLAYER_COL
        elif message.startswith("NPC"):
            col = NPC_COL

        panel_w, panel_h = 470, 170
        panel_x = self.sw // 2 - panel_w // 2
        panel_y = self.sh // 2 - panel_h // 2
        pygame.draw.rect(surface, (9, 15, 25), (panel_x, panel_y, panel_w, panel_h), border_radius=8)
        pygame.draw.rect(surface, col, (panel_x, panel_y, panel_w, panel_h), 2, border_radius=8)

        txt = self.font_xl.render(message, True, col)
        surface.blit(txt, (self.sw // 2 - txt.get_width() // 2, panel_y + 34))

        hint = self.font_md.render("Press R to restart", True, TEXT_DIM)
        surface.blit(hint, (self.sw // 2 - hint.get_width() // 2, panel_y + 112))

    def _draw_health_bar(self, surface, x, y, w, hp, max_hp):
        h = 14
        frac = 0.0 if max_hp <= 0 else max(0.0, min(hp / max_hp, 1.0))
        pygame.draw.rect(surface, (34, 16, 20), (x, y, w, h), border_radius=5)
        if frac > 0:
            fill = max(2, int(w * frac))
            pygame.draw.rect(surface, _hp_colour(frac), (x, y, fill, h), border_radius=5)
            shine = pygame.Surface((fill, h // 2), pygame.SRCALPHA)
            shine.fill((255, 255, 255, 34))
            surface.blit(shine, (x, y))
        pygame.draw.rect(surface, (105, 132, 154), (x, y, w, h), 1, border_radius=5)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _hp_colour(frac: float):
    """Interpolate red, yellow and green based on HP fraction."""
    if frac > 0.5:
        t = (frac - 0.5) / 0.5
        return (int(255 * (1 - t)), 200, int(30 * (1 - t)))
    else:
        t = frac / 0.5
        return (255, int(200 * t), 0)
