#!/usr/bin/env python3
"""
PX100 Robot Arm Demo Controller
================================
3D visualization · expressive gestures · on-screen joystick · gripper/rotation

Controls
--------
  Mouse drag on 3D view : orbit camera
  Scroll                : zoom
  Left joystick pad     : waist / shoulder
  Right joystick pad    : elbow / wrist rotation
  Grip slider           : open / close gripper
  A / D                 : waist
  W / S                 : shoulder
  Up / Down             : elbow
  Left / Right          : wrist rotate
  Space                 : toggle gripper
  H                     : home position
  R                     : toggle auto-rotate camera
  1-9                   : trigger gestures
  Esc                   : quit
"""

import os, sys, math, time, json
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import numpy as np

from robot import (
    PX100Arm, GesturePlayer, HardwareLink,
    GESTURES, LIMITS, JOINT_NAMES, L_BASE,
    PRESETS, PRESET_ORDER, FLOOR_Z,
    COMPOUND_ORDER,
)
try:
    from mesh_render import PX100MeshModel
except ImportError:
    PX100MeshModel = None

# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════
WIN_W, WIN_H = 1280, 1080
VIEW_W  = 880
PANEL_X = VIEW_W
PANEL_W = WIN_W - VIEW_W
FPS = 60

SAVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_positions.json')

# Colors
C_BG       = (18, 18, 32)
C_PANEL    = (26, 26, 44)
C_DIVIDER  = (42, 42, 62)
C_GRID     = (38, 38, 58)
C_TEXT     = (210, 210, 225)
C_DIM      = (110, 110, 135)
C_ACCENT   = (70, 130, 255)
C_WHITE    = (240, 240, 248)
C_STOP     = (220, 60, 60)
C_GREEN    = (60, 200, 90)
C_ORANGE   = (240, 160, 40)
C_YELLOW   = (255, 210, 50)
C_PURPLE   = (160, 100, 220)
C_TEAL     = (60, 190, 170)

# Arm segment colors
C_BASE_LINK = (110, 110, 130)
C_UPPER     = (70, 140, 255)
C_FORE      = (50, 195, 195)
C_HAND      = (75, 200, 120)
C_GRIP_C    = (255, 200, 60)
C_JOINT_DOT = (230, 230, 245)

GESTURE_ORDER = ['happy', 'surprise', 'chirpy', 'funny', 'waving', 'bite', 'curious', 'sad', 'excited']
COMPOUND_GESTURE_ORDER = COMPOUND_ORDER


# ═══════════════════════════════════════════════════════════
#  Camera
# ═══════════════════════════════════════════════════════════
class Camera:
    def __init__(self):
        self.az   = 32.0
        self.el   = 28.0
        self.zoom = 1.0
        self.cx   = VIEW_W // 2
        self.cy   = WIN_H // 2 + 30
        self.fov  = 2000.0
        self._drag = False
        self._ms   = (0, 0)
        self._az0  = 0.0
        self._el0  = 0.0

    def event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and ev.pos[0] < VIEW_W:
            self._drag = True
            self._ms   = ev.pos
            self._az0  = self.az
            self._el0  = self.el
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._drag = False
        elif ev.type == pygame.MOUSEMOTION and self._drag:
            dx = ev.pos[0] - self._ms[0]
            dy = ev.pos[1] - self._ms[1]
            self.az = self._az0 + dx * 0.4
            self.el = max(-80, min(80, self._el0 - dy * 0.4))
        elif ev.type == pygame.MOUSEWHEEL:
            self.zoom = max(0.3, min(3.0, self.zoom + ev.y * 0.1))

    def project(self, p):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        a = math.radians(self.az)
        ca, sa = math.cos(a), math.sin(a)
        x1 =  x*ca - y*sa
        y1 =  x*sa + y*ca
        e = math.radians(self.el)
        ce, se = math.cos(e), math.sin(e)
        y2 =  y1*ce - z*se
        z2 =  y1*se + z*ce
        d = 2.0 + y2
        if d < 0.05:
            d = 0.05
        f = self.fov * self.zoom / d
        return (int(self.cx + x1*f), int(self.cy - z2*f)), d


# ═══════════════════════════════════════════════════════════
#  Virtual Joystick (on-screen draggable pad)
# ═══════════════════════════════════════════════════════════
class VJoy:
    def __init__(self, cx, cy, radius, label, color=C_ACCENT):
        self.cx, self.cy, self.r = cx, cy, radius
        self.label = label
        self.color = color
        self.vx = 0.0   # -1..1
        self.vy = 0.0   # -1..1
        self._drag = False

    def draw(self, surf, font):
        # Background disc
        bg_surf = pygame.Surface((self.r*2+4, self.r*2+4), pygame.SRCALPHA)
        pygame.draw.circle(bg_surf, (30, 30, 50, 160), (self.r+2, self.r+2), self.r)
        surf.blit(bg_surf, (self.cx-self.r-2, self.cy-self.r-2))
        # Ring
        pygame.draw.circle(surf, (60, 60, 85), (self.cx, self.cy), self.r, 2)
        # Crosshair
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            end = (self.cx + int(dx*self.r*0.8), self.cy + int(dy*self.r*0.8))
            pygame.draw.aaline(surf, (45, 45, 65), (self.cx, self.cy), end)
        # Thumb
        tx = int(self.cx + self.vx * self.r * 0.72)
        ty = int(self.cy + self.vy * self.r * 0.72)
        pygame.draw.circle(surf, self.color, (tx, ty), 16)
        glow = tuple(min(255, c+80) for c in self.color)
        pygame.draw.circle(surf, glow, (tx, ty), 10)
        pygame.draw.circle(surf, C_WHITE, (tx, ty), 5)
        # Label
        ls = font.render(self.label, True, C_DIM)
        surf.blit(ls, (self.cx - ls.get_width()//2, self.cy + self.r + 6))

    def event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            d = math.hypot(ev.pos[0]-self.cx, ev.pos[1]-self.cy)
            if d <= self.r + 8:
                self._drag = True
                self._set(ev.pos)
                return True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self._drag:
                self._drag = False
                self.vx, self.vy = 0, 0
                return True
        elif ev.type == pygame.MOUSEMOTION and self._drag:
            self._set(ev.pos)
            return True
        return False

    def _set(self, pos):
        dx = (pos[0] - self.cx) / self.r
        dy = (pos[1] - self.cy) / self.r
        m = math.hypot(dx, dy)
        if m > 1:
            dx /= m; dy /= m
        self.vx, self.vy = dx, dy


# ═══════════════════════════════════════════════════════════
#  Vertical Grip Slider
# ═══════════════════════════════════════════════════════════
class GripSlider:
    """Vertical slider for gripper open/close."""
    def __init__(self, cx, y_top, height):
        self.cx = cx
        self.y0 = y_top
        self.h  = height
        self._drag = False

    def draw(self, surf, grip_val, font):
        # Track
        tx = self.cx
        pygame.draw.line(surf, (50, 50, 72), (tx, self.y0), (tx, self.y0+self.h), 4)
        # Fill from bottom (closed) to current
        t = grip_val  # 0=closed(bottom), 1=open(top)
        fy = int(self.y0 + self.h * (1 - t))
        pygame.draw.line(surf, C_GREEN, (tx, fy), (tx, self.y0+self.h), 4)
        # Thumb
        pygame.draw.circle(surf, C_WHITE, (tx, fy), 10)
        pygame.draw.circle(surf, C_GREEN, (tx, fy), 7)
        # Labels
        o = font.render("OPEN", True, C_GREEN)
        surf.blit(o, (tx - o.get_width()//2, self.y0 - 18))
        c = font.render("CLOSE", True, C_ORANGE)
        surf.blit(c, (tx - c.get_width()//2, self.y0 + self.h + 6))
        # Percentage
        pct = font.render(f"{int(grip_val*100)}%", True, C_TEXT)
        surf.blit(pct, (tx + 16, fy - 7))

    def event(self, ev, robot, gesture_player):
        hit = pygame.Rect(self.cx - 18, self.y0 - 10, 36, self.h + 20)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and hit.collidepoint(ev.pos):
            self._drag = True
            gesture_player.stop()
            self._apply(ev.pos[1], robot)
            return True
        if ev.type == pygame.MOUSEBUTTONUP:
            self._drag = False
        if ev.type == pygame.MOUSEMOTION and self._drag:
            self._apply(ev.pos[1], robot)
            return True
        return False

    def _apply(self, my, robot):
        t = 1.0 - max(0, min(1, (my - self.y0) / self.h))
        robot.angles['gripper'] = t


# ═══════════════════════════════════════════════════════════
#  Rotation Knob (wrist/gripper rotation)
# ═══════════════════════════════════════════════════════════
class RotKnob:
    """Circular knob for wrist rotation control."""
    def __init__(self, cx, cy, radius):
        self.cx, self.cy, self.r = cx, cy, radius
        self._drag = False
        self._last_angle = 0

    def draw(self, surf, wrist_val, font):
        # Background disc
        bg = pygame.Surface((self.r*2+4, self.r*2+4), pygame.SRCALPHA)
        pygame.draw.circle(bg, (30, 30, 50, 140), (self.r+2, self.r+2), self.r)
        surf.blit(bg, (self.cx-self.r-2, self.cy-self.r-2))
        # Ring
        pygame.draw.circle(surf, (60, 60, 85), (self.cx, self.cy), self.r, 2)
        # Tick marks
        for i in range(12):
            a = math.radians(i * 30)
            r1 = self.r - 6
            r2 = self.r - 2
            x1 = self.cx + int(r1 * math.cos(a))
            y1 = self.cy + int(r1 * math.sin(a))
            x2 = self.cx + int(r2 * math.cos(a))
            y2 = self.cy + int(r2 * math.sin(a))
            pygame.draw.aaline(surf, (55, 55, 75), (x1,y1), (x2,y2))
        # Indicator line — maps wrist range to full rotation visual
        lo, hi = LIMITS['wrist']
        frac = (wrist_val - lo) / (hi - lo)  # 0..1
        angle = math.radians(-220 + frac * 260)  # sweep from -220 to +40 deg
        ix = self.cx + int((self.r - 12) * math.cos(angle))
        iy = self.cy + int((self.r - 12) * math.sin(angle))
        pygame.draw.line(surf, C_ORANGE, (self.cx, self.cy), (ix, iy), 3)
        pygame.draw.circle(surf, C_ORANGE, (ix, iy), 4)
        # Center dot
        pygame.draw.circle(surf, (80, 80, 100), (self.cx, self.cy), 5)
        # Label & value
        lb = font.render("ROTATE", True, C_DIM)
        surf.blit(lb, (self.cx - lb.get_width()//2, self.cy + self.r + 6))
        vt = font.render(f"{wrist_val:+.0f}\u00b0", True, C_TEXT)
        surf.blit(vt, (self.cx - vt.get_width()//2, self.cy + self.r + 20))

    def event(self, ev, robot, gesture_player):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            d = math.hypot(ev.pos[0]-self.cx, ev.pos[1]-self.cy)
            if d <= self.r + 8:
                self._drag = True
                self._last_angle = math.atan2(ev.pos[1]-self.cy, ev.pos[0]-self.cx)
                return True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._drag = False
        elif ev.type == pygame.MOUSEMOTION and self._drag:
            a = math.atan2(ev.pos[1]-self.cy, ev.pos[0]-self.cx)
            delta = a - self._last_angle
            # Handle wrap-around
            if delta > math.pi: delta -= 2*math.pi
            if delta < -math.pi: delta += 2*math.pi
            self._last_angle = a
            gesture_player.stop()
            robot.angles['wrist'] += math.degrees(delta) * 1.5
            lo, hi = LIMITS['wrist']
            robot.angles['wrist'] = max(lo, min(hi, robot.angles['wrist']))
            return True
        return False


# ═══════════════════════════════════════════════════════════
#  Panel UI Elements
# ═══════════════════════════════════════════════════════════
class Button:
    def __init__(self, x, y, w, h, label, color, cb):
        self.rect  = pygame.Rect(x, y, w, h)
        self.label = label
        self.color = color
        self.cb    = cb
        self.hover = False
        self.active = False

    def draw(self, surf, font):
        if self.active:
            bg, tc = self.color, C_BG
        elif self.hover:
            bg, tc = tuple(min(255, c+12) for c in C_PANEL), C_TEXT
        else:
            bg, tc = C_PANEL, C_TEXT
        pygame.draw.rect(surf, bg, self.rect, border_radius=7)
        bc = self.color if (self.active or self.hover) else C_DIVIDER
        pygame.draw.rect(surf, bc, self.rect, width=2, border_radius=7)
        ts = font.render(self.label, True, tc)
        surf.blit(ts, ts.get_rect(center=self.rect.center))

    def event(self, ev):
        if ev.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                self.cb()
                return True
        return False


class Slider:
    def __init__(self, x, y, w, label, joint, limits):
        self.x, self.y, self.w = x, y, w
        self.label = label
        self.joint = joint
        self.lo, self.hi = limits
        self.drag = False
        self.track = pygame.Rect(x, y+18, w, 6)

    def draw(self, surf, robot, fn_sm, fn_val):
        v = robot.angles[self.joint]
        t = (v - self.lo) / (self.hi - self.lo)
        if self.joint == 'gripper':
            vt = f"{int(v*100)}%"
        else:
            vt = f"{v:+.0f}\u00b0"
        surf.blit(fn_sm.render(self.label, True, C_DIM), (self.x, self.y))
        vs = fn_val.render(vt, True, C_TEXT)
        surf.blit(vs, (self.x + self.w - vs.get_width(), self.y))
        pygame.draw.rect(surf, C_DIVIDER, self.track, border_radius=3)
        fw = max(0, int(t * self.track.w))
        pygame.draw.rect(surf, C_ACCENT, (self.track.x, self.track.y, fw, self.track.h), border_radius=3)
        tx = self.track.x + int(t * self.track.w)
        ty = self.track.centery
        pygame.draw.circle(surf, C_WHITE, (tx, ty), 8)
        pygame.draw.circle(surf, C_ACCENT, (tx, ty), 6)

    def event(self, ev, robot, gesture_player):
        hit = self.track.inflate(12, 24)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and hit.collidepoint(ev.pos):
            self.drag = True
            gesture_player.stop()
            self._apply(ev.pos[0], robot)
            return True
        if ev.type == pygame.MOUSEBUTTONUP:
            self.drag = False
        if ev.type == pygame.MOUSEMOTION and self.drag:
            self._apply(ev.pos[0], robot)
            return True
        return False

    def _apply(self, mx, robot):
        t = max(0, min(1, (mx - self.track.x) / self.track.w))
        robot.angles[self.joint] = self.lo + t * (self.hi - self.lo)


# ═══════════════════════════════════════════════════════════
#  Grabbable objects
# ═══════════════════════════════════════════════════════════
class GrabObject:
    def __init__(self, pos, radius, color):
        self.pos   = np.array(pos, dtype=float)
        self.rest  = np.array(pos, dtype=float)
        self.r     = radius
        self.color = color
        self.held  = False

    def update(self, grip_pos, grip_open):
        if self.held:
            if grip_open > 0.65:
                self.held = False
                self.pos[2] = max(self.r, self.pos[2])
            else:
                self.pos = grip_pos.copy()
                self.pos[2] = max(self.r, self.pos[2])
        else:
            dist = np.linalg.norm(self.pos - grip_pos)
            if dist < 0.045 and grip_open < 0.3:
                self.held = True
            if not self.held and self.pos[2] > self.r + 0.001:
                self.pos[2] = max(self.r, self.pos[2] - 0.002)


# ═══════════════════════════════════════════════════════════
#  Rendering helpers
# ═══════════════════════════════════════════════════════════
def draw_floor(surf, cam):
    span, step = 0.30, 0.05
    v = -span
    while v <= span + 0.001:
        pa, _ = cam.project((v, -span, 0))
        pb, _ = cam.project((v,  span, 0))
        pygame.draw.aaline(surf, C_GRID, pa, pb)
        pa, _ = cam.project((-span, v, 0))
        pb, _ = cam.project(( span, v, 0))
        pygame.draw.aaline(surf, C_GRID, pa, pb)
        v += step


def draw_arm(surf, cam, robot):
    """Draw robot arm with depth-based shading, highlights, and glow joints."""
    pts, lf, rf = robot.fk()
    colors = [C_BASE_LINK, C_UPPER, C_FORE, C_HAND]
    widths = [10, 9, 7, 6]

    # Floor shadow — soft offset
    shadow_c = (25, 25, 38)
    sp = [cam.project((p[0], p[1], 0))[0] for p in pts]
    for i in range(len(sp)-1):
        pygame.draw.line(surf, shadow_c, sp[i], sp[i+1], widths[i] - 1)

    # Project all points
    proj = [cam.project(p) for p in pts]

    # Segments with depth shading + highlight/shadow edges
    for i in range(len(pts)-1):
        p1, d1 = proj[i]
        p2, d2 = proj[i+1]
        avg_d = (d1 + d2) / 2
        bright = max(0.6, min(1.15, 1.3 / max(0.5, avg_d)))

        base = colors[i]
        w = widths[i]

        # Perpendicular in screen space (for edge shading)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1:
            continue
        nx = -dy / seg_len
        ny = dx / seg_len
        off = max(1, w // 4)

        # Shadow edge (shifted toward bottom-right)
        dark_c = tuple(max(0, int(c * 0.35 * bright)) for c in base)
        s1 = (p1[0] + int(nx * off) + 1, p1[1] + int(ny * off) + 1)
        s2 = (p2[0] + int(nx * off) + 1, p2[1] + int(ny * off) + 1)
        pygame.draw.line(surf, dark_c, s1, s2, w)

        # Main segment body
        main_c = tuple(int(min(255, c * 0.85 * bright)) for c in base)
        pygame.draw.line(surf, main_c, p1, p2, w)

        # Highlight edge (shifted toward top-left)
        hi_c = tuple(int(min(255, c * 1.3 * bright)) for c in base)
        h1 = (p1[0] - int(nx * off), p1[1] - int(ny * off))
        h2 = (p2[0] - int(nx * off), p2[1] - int(ny * off))
        pygame.draw.line(surf, hi_c, h1, h2, max(1, w // 3))

    # Joints with glow effect
    for i, (s, d) in enumerate(proj):
        r = 9 if i < 2 else 8
        bright = max(0.6, min(1.15, 1.3 / max(0.5, d)))
        c = colors[min(i, len(colors)-1)]

        # Soft glow ring
        glow_r = r + 6
        glow_c = tuple(int(min(255, v * 0.5 * bright)) for v in c)
        glow_surf = pygame.Surface((glow_r*2, glow_r*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*glow_c, 45), (glow_r, glow_r), glow_r)
        surf.blit(glow_surf, (s[0]-glow_r, s[1]-glow_r))

        # Outer ring
        pygame.draw.circle(surf, C_JOINT_DOT, s, r)
        # Inner colored fill
        inner_c = tuple(int(min(255, v * bright)) for v in c)
        pygame.draw.circle(surf, inner_c, s, r - 2)
        # Specular highlight dot
        hi_pos = (s[0] - 2, s[1] - 2)
        pygame.draw.circle(surf, (255, 255, 255), hi_pos, max(1, r // 3))

    # Gripper with highlights
    wp = proj[-1][0]
    lp, _ = cam.project(lf)
    rp, _ = cam.project(rf)
    pygame.draw.line(surf, C_GRIP_C, wp, lp, 6)
    pygame.draw.line(surf, C_GRIP_C, wp, rp, 6)
    hi_grip = tuple(min(255, c + 60) for c in C_GRIP_C)
    pygame.draw.line(surf, hi_grip, wp, lp, 2)
    pygame.draw.line(surf, hi_grip, wp, rp, 2)
    pygame.draw.circle(surf, C_GRIP_C, lp, 6)
    pygame.draw.circle(surf, C_GRIP_C, rp, 6)
    pygame.draw.circle(surf, hi_grip, lp, 3)
    pygame.draw.circle(surf, hi_grip, rp, 3)

    return pts


def draw_objects(surf, cam, objects):
    for obj in objects:
        sp, depth = cam.project(obj.pos)
        sz = max(5, int(14 * cam.zoom / max(0.5, depth)))
        c = obj.color if not obj.held else tuple(min(255, v+70) for v in obj.color)
        pygame.draw.circle(surf, c, sp, sz)
        pygame.draw.circle(surf, (255,255,255), sp, sz, 1)


# ═══════════════════════════════════════════════════════════
#  Layout helper
# ═══════════════════════════════════════════════════════════
def _grid_rows(n):
    """Number of rows for n items in a 2-column grid."""
    return (n + 1) // 2


# ═══════════════════════════════════════════════════════════
#  Application
# ═══════════════════════════════════════════════════════════
class App:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("PX100 Robot Arm Controller")
        self.clock = pygame.time.Clock()

        self.f_title = pygame.font.SysFont("Helvetica", 19, bold=True)
        self.f_label = pygame.font.SysFont("Helvetica", 14)
        self.f_small = pygame.font.SysFont("Helvetica", 12)
        self.f_big   = pygame.font.SysFont("Helvetica", 28, bold=True)

        self.robot   = PX100Arm()
        self.gesture = GesturePlayer(self.robot)
        self.hw      = HardwareLink()
        self.cam     = Camera()

        # ── Auto-rotate camera ────────────────────────────
        self.auto_rotate = False

        # ── 3D Mesh model ────────────────────────────────
        self.mesh_model = None
        if PX100MeshModel is not None:
            try:
                self.mesh_model = PX100MeshModel(decimate=4)
                if not self.mesh_model.ready:
                    self.mesh_model = None
            except Exception:
                self.mesh_model = None

        # ── Smooth movement ──────────────────────────────
        self.move_target = None     # dict of target angles, or None if idle
        self.move_speed  = 0.30     # 0.05 (very slow) to 1.0 (fast)

        # ── Saved positions ──────────────────────────────
        self.saved_positions = []   # list of {'name': str, 'angles': dict}
        self._load_saved()

        self.objects = [
            GrabObject([0.15,  0.00, 0.012], 0.012, (230, 75, 55)),
            GrabObject([0.10,  0.10, 0.012], 0.012, (55, 120, 230)),
            GrabObject([0.10, -0.10, 0.012], 0.012, (55, 200, 90)),
        ]

        # Physical joystick
        self.joy = None
        self.joy_name = "None"
        if pygame.joystick.get_count() > 0:
            self.joy = pygame.joystick.Joystick(0)
            self.joy.init()
            self.joy_name = self.joy.get_name()[:22]

        # ── On-screen controls (bottom of 3D viewport) ────
        ctrl_y = WIN_H - 145
        self.vjoy_left  = VJoy(140, ctrl_y, 62, "Rotate / Lift", C_ACCENT)
        self.vjoy_right = VJoy(VIEW_W - 140, ctrl_y, 62, "Reach / Elbow", (50, 195, 195))
        self.grip_slider = GripSlider(VIEW_W // 2 - 60, ctrl_y - 55, 110)
        self.rot_knob    = RotKnob(VIEW_W // 2 + 60, ctrl_y, 42)

        # ── Panel layout (dynamic Y positions) ─────────────
        bx = PANEL_X + 12
        bw = (PANEL_W - 34) // 2
        bh = 32
        row_h = bh + 5  # 37px per row
        sec_gap = 14     # gap between sections

        # --- Gesture buttons ---
        gest_label_y = 52
        by0 = gest_label_y + 15

        self.buttons = []
        for i, gname in enumerate(GESTURE_ORDER):
            g = GESTURES[gname]
            x = bx + (i % 2) * (bw + 10)
            y = by0 + (i // 2) * row_h
            self.buttons.append(Button(x, y, bw, bh, g['label'], g['color'],
                                       lambda n=gname: self._play(n)))
        stop_row = len(GESTURE_ORDER) // 2
        stop_col = len(GESTURE_ORDER) % 2
        sx = bx + stop_col * (bw + 10)
        sy = by0 + stop_row * row_h
        self.btn_stop = Button(sx, sy, bw, bh, "STOP", C_STOP, self._stop)
        self.buttons.append(self.btn_stop)
        gesture_bottom = sy + bh

        # --- Compound movement buttons ---
        self.compound_label_y = gesture_bottom + sec_gap
        compound_by0 = self.compound_label_y + 15
        self.compound_buttons = []
        for i, cname in enumerate(COMPOUND_GESTURE_ORDER):
            g = GESTURES[cname]
            x = bx + (i % 2) * (bw + 10)
            y = compound_by0 + (i // 2) * row_h
            self.compound_buttons.append(Button(x, y, bw, bh, g['label'], g['color'],
                                                lambda n=cname: self._play(n)))
        n_compound_rows = _grid_rows(len(COMPOUND_GESTURE_ORDER))
        compound_bottom = compound_by0 + n_compound_rows * row_h

        # --- Preset position buttons ---
        self.preset_label_y = compound_bottom + sec_gap - 4
        preset_by0 = self.preset_label_y + 15
        self.preset_buttons = []
        for i, pname in enumerate(PRESET_ORDER):
            p = PRESETS[pname]
            x = bx + (i % 2) * (bw + 10)
            y = preset_by0 + (i // 2) * row_h
            self.preset_buttons.append(Button(x, y, bw, bh, p['label'], p['color'],
                                              lambda n=pname: self._preset(n)))
        n_preset_rows = _grid_rows(len(PRESET_ORDER))
        preset_bottom = preset_by0 + n_preset_rows * row_h

        # --- Gripper open/close buttons ---
        self.grip_label_y = preset_bottom + sec_gap - 4
        grip_by0 = self.grip_label_y + 15
        self.btn_grip_open = Button(bx, grip_by0, bw, bh, "OPEN GRIP", C_GREEN,
                                    lambda: self._set_gripper(1.0))
        self.btn_grip_close = Button(bx + bw + 10, grip_by0, bw, bh, "CLOSE GRIP", C_ORANGE,
                                     lambda: self._set_gripper(0.0))
        grip_bottom = grip_by0 + bh

        # ── Speed slider ─────────────────────────────────
        self.speed_y = grip_bottom + sec_gap
        self.speed_track = pygame.Rect(bx, self.speed_y + 18, PANEL_W - 26, 6)
        self._speed_drag = False
        speed_bottom = self.speed_y + 30

        # ── Joint sliders ────────────────────────────────
        self.joint_label_y = speed_bottom + sec_gap - 4
        slider_y0 = self.joint_label_y + 15
        slider_h = 38
        labels = ['Waist', 'Shoulder', 'Elbow', 'Wrist Rotate', 'Gripper']
        self.sliders = []
        for i, jname in enumerate(JOINT_NAMES):
            self.sliders.append(Slider(
                bx, slider_y0 + i * slider_h, PANEL_W - 26,
                labels[i], jname, LIMITS[jname]))
        slider_bottom = slider_y0 + 5 * slider_h

        # ── Save / Load position buttons ─────────────────
        self.save_section_y = slider_bottom + 6
        save_bw_half = (PANEL_W - 34) // 2
        self.btn_save = Button(bx, self.save_section_y + 15, save_bw_half, bh,
                               "SAVE POS", C_PURPLE, self._save_position)
        self.btn_clear = Button(bx + save_bw_half + 10, self.save_section_y + 15, save_bw_half, bh,
                                "CLR ALL", C_STOP, self._clear_positions)

        # Store layout constants for _draw
        self._bh = bh
        self._row_h = row_h

        # ── Safety: collision tracking ─────────────────────
        self.last_safe_angles = dict(self.robot.angles)
        self.collision_warning = 0.0   # >0 means flash timer (seconds)

        self.running = True
        self.gripper_toggled = False

    # ── Saved positions persistence ──────────────────────
    def _load_saved(self):
        try:
            with open(SAVE_FILE, 'r') as f:
                self.saved_positions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.saved_positions = []

    def _persist_saved(self):
        with open(SAVE_FILE, 'w') as f:
            json.dump(self.saved_positions, f, indent=2)

    def _save_position(self):
        idx = len(self.saved_positions) + 1
        entry = {
            'name': f'Pos {idx}',
            'angles': {k: round(v, 1) for k, v in self.robot.angles.items()},
        }
        self.saved_positions.append(entry)
        self._persist_saved()

    def _load_position(self, idx):
        if 0 <= idx < len(self.saved_positions):
            self.gesture.stop()
            for b in self.buttons: b.active = False
            self.move_target = dict(self.saved_positions[idx]['angles'])

    def _clear_positions(self):
        self.saved_positions = []
        self._persist_saved()

    # ── Actions ──────────────────────────────────────────
    def _play(self, name):
        self.move_target = None
        self.gesture.play(name)
        for b in self.buttons:
            b.active = (b.label == GESTURES.get(name, {}).get('label'))
        for b in self.compound_buttons:
            b.active = (b.label == GESTURES.get(name, {}).get('label'))

    def _stop(self):
        self.gesture.stop()
        self.move_target = None
        for b in self.buttons: b.active = False
        for b in self.compound_buttons: b.active = False

    def _home(self):
        self._preset('forward')

    def _preset(self, name):
        self.gesture.stop()
        for b in self.buttons: b.active = False
        for b in self.compound_buttons: b.active = False
        if name in PRESETS:
            self.move_target = dict(PRESETS[name]['angles'])

    def _set_gripper(self, val):
        self.gesture.stop()
        for b in self.buttons: b.active = False
        for b in self.compound_buttons: b.active = False
        if self.move_target is None:
            self.move_target = dict(self.robot.angles)
        self.move_target['gripper'] = val

    # ── main loop ─────────────────────────────────────────
    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self._events()
            self._update(dt)
            self._draw()
        pygame.quit()

    def _events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False; return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                self.running = False; return

            # Gesture hotkeys
            if ev.type == pygame.KEYDOWN:
                if   ev.key == pygame.K_1: self._play('happy')
                elif ev.key == pygame.K_2: self._play('surprise')
                elif ev.key == pygame.K_3: self._play('chirpy')
                elif ev.key == pygame.K_4: self._play('funny')
                elif ev.key == pygame.K_5: self._play('waving')
                elif ev.key == pygame.K_6: self._play('bite')
                elif ev.key == pygame.K_7: self._play('curious')
                elif ev.key == pygame.K_8: self._play('sad')
                elif ev.key == pygame.K_9: self._play('excited')
                elif ev.key == pygame.K_h: self._home()
                elif ev.key == pygame.K_r: self.auto_rotate = not self.auto_rotate
                elif ev.key == pygame.K_SPACE:
                    if not self.gripper_toggled:
                        g = self.robot.angles['gripper']
                        self._set_gripper(0.0 if g > 0.5 else 1.0)
                        self.gripper_toggled = True
            if ev.type == pygame.KEYUP and ev.key == pygame.K_SPACE:
                self.gripper_toggled = False

            # On-screen controls have priority over camera orbit
            consumed = False
            consumed = self.vjoy_left.event(ev) or consumed
            consumed = self.vjoy_right.event(ev) or consumed
            consumed = self.grip_slider.event(ev, self.robot, self.gesture) or consumed
            consumed = self.rot_knob.event(ev, self.robot, self.gesture) or consumed
            for b in self.buttons:
                consumed = b.event(ev) or consumed
            for b in self.compound_buttons:
                consumed = b.event(ev) or consumed
            for b in self.preset_buttons:
                consumed = b.event(ev) or consumed
            consumed = self.btn_grip_open.event(ev) or consumed
            consumed = self.btn_grip_close.event(ev) or consumed

            # Speed slider events
            consumed = self._speed_event(ev) or consumed

            for s in self.sliders:
                if s.event(ev, self.robot, self.gesture):
                    self.move_target = None  # cancel smooth move on slider drag
                    consumed = True

            # Save / load / clear buttons
            consumed = self.btn_save.event(ev) or consumed
            consumed = self.btn_clear.event(ev) or consumed
            # Saved position buttons (dynamic)
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                idx = self._saved_btn_hit(ev.pos)
                if idx is not None:
                    self._load_position(idx)
                    consumed = True
            if ev.type == pygame.MOUSEMOTION:
                pass  # hover handled in draw

            if not consumed:
                self.cam.event(ev)

    def _speed_event(self, ev):
        hit = self.speed_track.inflate(12, 24)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and hit.collidepoint(ev.pos):
            self._speed_drag = True
            self._speed_apply(ev.pos[0])
            return True
        if ev.type == pygame.MOUSEBUTTONUP:
            self._speed_drag = False
        if ev.type == pygame.MOUSEMOTION and self._speed_drag:
            self._speed_apply(ev.pos[0])
            return True
        return False

    def _speed_apply(self, mx):
        t = max(0, min(1, (mx - self.speed_track.x) / self.speed_track.w))
        self.move_speed = 0.05 + t * 0.95  # range 0.05 to 1.0

    def _saved_btn_hit(self, pos):
        """Return index of saved position button at pos, or None."""
        bx = PANEL_X + 12
        bw3 = (PANEL_W - 40) // 3
        btn_y0 = self.save_section_y + 15 + self._bh + 8
        for i in range(len(self.saved_positions)):
            col = i % 3
            row = i // 3
            rx = bx + col * (bw3 + 7)
            ry = btn_y0 + row * 34
            r = pygame.Rect(rx, ry, bw3, 28)
            if r.collidepoint(pos):
                return i
        return None

    def _update(self, dt):
        # Auto-rotate camera
        if self.auto_rotate:
            self.cam.az += 25 * dt

        # Gesture animation
        if self.gesture.playing:
            self.gesture.update()
            self.move_target = None  # gesture overrides smooth move
            if not self.gesture.playing:
                for b in self.buttons: b.active = False
                for b in self.compound_buttons: b.active = False
        else:
            # On-screen joystick input — cancels smooth move
            js = 110 * dt
            if abs(self.vjoy_left.vx) > 0.05 or abs(self.vjoy_left.vy) > 0.05:
                self.move_target = None
                self.robot.angles['waist']    += self.vjoy_left.vx * js
                self.robot.angles['shoulder'] -= self.vjoy_left.vy * js
            if abs(self.vjoy_right.vx) > 0.05 or abs(self.vjoy_right.vy) > 0.05:
                self.move_target = None
                self.robot.angles['wrist'] += self.vjoy_right.vx * js
                self.robot.angles['elbow'] -= self.vjoy_right.vy * js

            # Keyboard — cancels smooth move
            keys = pygame.key.get_pressed()
            sp = 100 * dt
            any_key = False
            if keys[pygame.K_a]:     self.robot.angles['waist']    -= sp; any_key = True
            if keys[pygame.K_d]:     self.robot.angles['waist']    += sp; any_key = True
            if keys[pygame.K_w]:     self.robot.angles['shoulder'] += sp; any_key = True
            if keys[pygame.K_s]:     self.robot.angles['shoulder'] -= sp; any_key = True
            if keys[pygame.K_UP]:    self.robot.angles['elbow']    += sp; any_key = True
            if keys[pygame.K_DOWN]:  self.robot.angles['elbow']    -= sp; any_key = True
            if keys[pygame.K_LEFT]:  self.robot.angles['wrist']    -= sp; any_key = True
            if keys[pygame.K_RIGHT]: self.robot.angles['wrist']    += sp; any_key = True
            if any_key:
                self.move_target = None

            # Physical joystick
            if self.joy:
                def ax(i):
                    if i >= self.joy.get_numaxes(): return 0
                    v = self.joy.get_axis(i)
                    return v if abs(v) > 0.15 else 0
                jsp = 120 * dt
                lx, ly = ax(0), ax(1)
                rx, ry = ax(2), ax(3)
                if any([lx, ly, rx, ry]):
                    self.gesture.stop()
                    self.move_target = None
                    for b in self.buttons: b.active = False
                    for b in self.compound_buttons: b.active = False
                self.robot.angles['waist']    += lx * jsp
                self.robot.angles['shoulder'] -= ly * jsp
                self.robot.angles['elbow']    -= ry * jsp
                self.robot.angles['wrist']    += rx * jsp
                if self.joy.get_numaxes() > 5:
                    lt = (ax(4)+1)/2
                    rt = (ax(5)+1)/2
                    if lt > 0.1 or rt > 0.1:
                        self.robot.angles['gripper'] = max(0, min(1, 0.5+rt*0.5-lt*0.5))
                for bi in range(min(self.joy.get_numbuttons(), 5)):
                    if self.joy.get_button(bi):
                        self._play(GESTURE_ORDER[bi])

            # ── Smooth interpolation toward move_target ──
            if self.move_target is not None:
                max_deg  = self.move_speed * 280 * dt   # deg/frame for joints
                max_grip = self.move_speed * 3.0 * dt   # grip units/frame
                done = True
                for j in JOINT_NAMES:
                    target = self.move_target[j]
                    current = self.robot.angles[j]
                    diff = target - current
                    ms = max_grip if j == 'gripper' else max_deg
                    if abs(diff) > 0.05:
                        step = max(-ms, min(ms, diff))
                        self.robot.angles[j] = current + step
                        done = False
                    else:
                        self.robot.angles[j] = target
                if done:
                    self.move_target = None

        # Clamp
        for j in JOINT_NAMES:
            lo, hi = LIMITS[j]
            self.robot.angles[j] = max(lo, min(hi, self.robot.angles[j]))

        # ── Safety: floor collision check ─────────────────
        if self.robot.is_safe():
            self.last_safe_angles = dict(self.robot.angles)
            if self.collision_warning > 0:
                self.collision_warning -= dt
        else:
            # Revert to last known safe position
            for j in JOINT_NAMES:
                self.robot.angles[j] = self.last_safe_angles[j]
            self.move_target = None
            self.gesture.stop()
            for b in self.buttons:
                b.active = False
            for b in self.compound_buttons:
                b.active = False
            self.collision_warning = 2.0   # flash warning for 2 seconds

        # Objects
        _, lf, rf = self.robot.fk()
        grip_center = (lf + rf) / 2.0
        for obj in self.objects:
            obj.update(grip_center, self.robot.angles['gripper'])

        # Hardware
        if self.hw.connected:
            self.hw.send(self.robot.angles)

    def _draw(self):
        self.screen.fill(C_BG)

        # ── 3D viewport ──
        draw_floor(self.screen, self.cam)
        if self.mesh_model:
            self.mesh_model.render(self.screen, self.cam, self.robot.angles, pygame)
        else:
            draw_arm(self.screen, self.cam, self.robot)
        draw_objects(self.screen, self.cam, self.objects)

        # Tip readout
        _, lf, rf = self.robot.fk()
        tip = (lf + rf) / 2
        ts = self.f_small.render(
            f"Tip: ({tip[0]*100:.1f}, {tip[1]*100:.1f}, {tip[2]*100:.1f}) cm",
            True, C_DIM)
        self.screen.blit(ts, (10, 10))

        # Auto-rotate indicator
        if self.auto_rotate:
            pulse = int(180 + 55 * math.sin(time.time() * 3))
            ar_c = (pulse, pulse, 255)
            ar_t = self.f_label.render("AUTO-ROTATE [R]", True, ar_c)
            self.screen.blit(ar_t, (10, 28))

        # Moving indicator
        if self.move_target is not None:
            pulse = int(180 + 75 * math.sin(time.time() * 6))
            mc = (pulse, pulse, 50)
            mt = self.f_label.render("MOVING...", True, mc)
            self.screen.blit(mt, (VIEW_W//2 - mt.get_width()//2, 45))

        # Collision warning
        if self.collision_warning > 0:
            pulse = int(180 + 75 * math.sin(time.time() * 10))
            warn_c = (pulse, 30, 30)
            wt = self.f_big.render("COLLISION!", True, warn_c)
            self.screen.blit(wt, (VIEW_W//2 - wt.get_width()//2, 70))
            ws = self.f_small.render("Pose would go below table \u2014 reverted to safe position", True, C_STOP)
            self.screen.blit(ws, (VIEW_W//2 - ws.get_width()//2, 102))

        # Gesture overlay
        if self.gesture.playing and self.gesture.gesture_name:
            g = GESTURES[self.gesture.gesture_name]
            nt = self.f_big.render(g['label'], True, g['color'])
            self.screen.blit(nt, (VIEW_W//2 - nt.get_width()//2, 12))

        # ── On-screen controls ──
        sep_y = WIN_H - 210
        pygame.draw.line(self.screen, (35,35,52), (10, sep_y), (VIEW_W-10, sep_y), 1)
        cl = self.f_small.render("CONTROLS", True, (60,60,80))
        self.screen.blit(cl, (VIEW_W//2 - cl.get_width()//2, sep_y - 14))

        self.vjoy_left.draw(self.screen, self.f_small)
        self.vjoy_right.draw(self.screen, self.f_small)
        self.grip_slider.draw(self.screen, self.robot.angles['gripper'], self.f_small)
        self.rot_knob.draw(self.screen, self.robot.angles['wrist'], self.f_small)

        # ── Right panel ──
        pygame.draw.rect(self.screen, C_PANEL, (PANEL_X, 0, PANEL_W, WIN_H))
        pygame.draw.line(self.screen, C_DIVIDER, (PANEL_X, 0), (PANEL_X, WIN_H), 2)

        tt = self.f_title.render("PX100 ARM CONTROL", True, C_WHITE)
        self.screen.blit(tt, (PANEL_X + (PANEL_W - tt.get_width())//2, 18))

        # Gestures
        self.screen.blit(self.f_small.render("GESTURES", True, C_DIM), (PANEL_X + 12, 52))
        for b in self.buttons:
            b.draw(self.screen, self.f_label)

        # Compound
        self.screen.blit(self.f_small.render("COMPOUND", True, C_TEAL),
                         (PANEL_X + 12, self.compound_label_y))
        for b in self.compound_buttons:
            b.draw(self.screen, self.f_label)

        # Presets
        self.screen.blit(self.f_small.render("POSITIONS", True, C_DIM),
                         (PANEL_X + 12, self.preset_label_y))
        for b in self.preset_buttons:
            b.draw(self.screen, self.f_label)

        # Gripper
        self.screen.blit(self.f_small.render("GRIPPER", True, C_DIM),
                         (PANEL_X + 12, self.grip_label_y))
        self.btn_grip_open.draw(self.screen, self.f_label)
        self.btn_grip_close.draw(self.screen, self.f_label)

        # Speed slider
        self.screen.blit(self.f_small.render("MOVE SPEED", True, C_DIM), (PANEL_X+12, self.speed_y))
        pct = int(self.move_speed * 100)
        pct_s = self.f_label.render(f"{pct}%", True, C_TEXT)
        self.screen.blit(pct_s, (PANEL_X + PANEL_W - 26 - pct_s.get_width(), self.speed_y))
        pygame.draw.rect(self.screen, C_DIVIDER, self.speed_track, border_radius=3)
        t_speed = (self.move_speed - 0.05) / 0.95
        fw = max(0, int(t_speed * self.speed_track.w))
        pygame.draw.rect(self.screen, C_YELLOW,
                         (self.speed_track.x, self.speed_track.y, fw, self.speed_track.h),
                         border_radius=3)
        thumb_x = self.speed_track.x + int(t_speed * self.speed_track.w)
        thumb_y = self.speed_track.centery
        pygame.draw.circle(self.screen, C_WHITE, (thumb_x, thumb_y), 8)
        pygame.draw.circle(self.screen, C_YELLOW, (thumb_x, thumb_y), 6)

        # Joint sliders
        self.screen.blit(self.f_small.render("JOINT CONTROL", True, C_DIM),
                         (PANEL_X + 12, self.joint_label_y))
        for s in self.sliders:
            s.draw(self.screen, self.robot, self.f_small, self.f_label)

        # ── Saved positions section ──
        self.screen.blit(self.f_small.render("SAVED POSITIONS", True, C_DIM),
                         (PANEL_X+12, self.save_section_y))
        self.btn_save.draw(self.screen, self.f_label)
        self.btn_clear.draw(self.screen, self.f_label)

        # Draw saved position buttons (grid of 3 per row)
        bx = PANEL_X + 12
        bw3 = (PANEL_W - 40) // 3
        btn_y0 = self.save_section_y + 15 + self._bh + 8
        mx, my = pygame.mouse.get_pos()
        for i, sp in enumerate(self.saved_positions):
            col = i % 3
            row = i // 3
            rx = bx + col * (bw3 + 7)
            ry = btn_y0 + row * 34
            r = pygame.Rect(rx, ry, bw3, 28)
            hover = r.collidepoint(mx, my)
            bg = (40, 40, 65) if hover else C_PANEL
            pygame.draw.rect(self.screen, bg, r, border_radius=5)
            bc = C_PURPLE if hover else C_DIVIDER
            pygame.draw.rect(self.screen, bc, r, width=2, border_radius=5)
            label = sp['name']
            ls = self.f_small.render(label, True, C_TEXT)
            self.screen.blit(ls, ls.get_rect(center=r.center))

        # ── Keyboard shortcuts (compact) ──
        max_save_rows = max(1, (len(self.saved_positions) + 2) // 3)
        ky = btn_y0 + max_save_rows * 34 + 8
        pygame.draw.line(self.screen, C_DIVIDER, (PANEL_X+12, ky), (PANEL_X+PANEL_W-12, ky))
        ky += 4
        self.screen.blit(self.f_small.render("KEYBOARD", True, C_DIM), (PANEL_X+12, ky))
        ky += 14
        for key, desc in [("A/D","Waist"),("W/S","Shoulder"),
                          ("\u2191/\u2193","Elbow"),("\u2190/\u2192","Wrist"),
                          ("Space","Gripper"),("H","Home"),("R","Auto-Rotate"),
                          ("1-9","Gestures")]:
            ks = self.f_small.render(key, True, C_ACCENT)
            ds = self.f_small.render(desc, True, C_DIM)
            self.screen.blit(ks, (PANEL_X+18, ky))
            self.screen.blit(ds, (PANEL_X+80, ky))
            ky += 14

        # Status
        ky += 2
        pygame.draw.line(self.screen, C_DIVIDER, (PANEL_X+12, ky), (PANEL_X+PANEL_W-12, ky))
        ky += 4
        self.screen.blit(self.f_small.render("STATUS", True, C_DIM), (PANEL_X+12, ky))
        ky += 14
        mode = "Hardware" if self.hw.connected else "Simulation"
        mc = C_GREEN if self.hw.connected else C_DIM
        self.screen.blit(self.f_small.render(f"Mode: {mode}", True, mc), (PANEL_X+18, ky))
        ky += 14
        self.screen.blit(self.f_small.render(f"Joystick: {self.joy_name}", True, C_TEXT), (PANEL_X+18, ky))
        ky += 14
        gn = self.gesture.gesture_name or "\u2014"
        self.screen.blit(self.f_small.render(f"Gesture: {gn}", True, C_TEXT), (PANEL_X+18, ky))
        ky += 14
        rot_s = "ON" if self.auto_rotate else "OFF"
        rot_c = C_TEAL if self.auto_rotate else C_DIM
        self.screen.blit(self.f_small.render(f"Rotate: {rot_s}", True, rot_c), (PANEL_X+18, ky))

        pygame.display.flip()


# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("PX100 Robot Arm Controller")
    print("\u2500" * 36)

    import glob
    ports = glob.glob('/dev/tty.usbserial*') + glob.glob('/dev/ttyUSB*')
    if ports:
        print(f"  Hardware detected: {ports[0]}")
    else:
        print("  No U2D2 detected \u2014 running in simulation mode")
    print("  Use on-screen joysticks, gripper slider, and rotation knob")
    print("  Press R to toggle auto-rotate, ESC to quit")
    print()

    app = App()

    if ports:
        if app.hw.connect(ports[0]):
            print(f"  Connected to {ports[0]}!")
        else:
            print(f"  Could not connect to {ports[0]}, continuing in sim mode")

    app.run()
