"""PX100 robot arm kinematics, gesture definitions, and hardware interface."""

import math
import time
import numpy as np

# ── PX100 dimensions (meters) ──────────────────────────────
L_BASE   = 0.089
L_UPPER  = 0.100
L_FORE   = 0.100
L_HAND   = 0.070
L_FINGER = 0.035

# ── Joint limits (degrees, except gripper 0-1) ─────────────
# Tightened from URDF maximums for table-top safety
LIMITS = {
    'waist':    (-180, 180),
    'shoulder': (-90,  75),    # URDF: -111/107, tightened to avoid table collision
    'elbow':    (-100, 80),    # URDF: -121/92, tightened for safety
    'wrist':    (-90,  90),    # URDF: -100/123, tightened for safety
    'gripper':  (0.0, 1.0),
}
JOINT_NAMES = ['waist', 'shoulder', 'elbow', 'wrist', 'gripper']

# Minimum Z height (meters) — any FK point below this triggers safety stop
FLOOR_Z = -0.01


class PX100Arm:
    """Forward kinematics model of the PincherX-100."""

    def __init__(self):
        self.angles = {j: 0.0 for j in JOINT_NAMES}
        self.angles['gripper'] = 0.5

    def set(self, **kw):
        for k, v in kw.items():
            if k in LIMITS:
                lo, hi = LIMITS[k]
                self.angles[k] = max(lo, min(hi, float(v)))

    def fk(self):
        """Return (joint_positions[5], left_finger, right_finger) in 3-D."""
        w  = math.radians(self.angles['waist'])
        s  = math.radians(-self.angles['shoulder'])   # negate: servo direction opposite to FK
        e  = math.radians(-self.angles['elbow'])      # negate: servo direction opposite to FK
        wr = math.radians(-self.angles['wrist'])      # negate: servo direction opposite to FK
        cw, sw = math.cos(w), math.sin(w)

        pts = [np.array([0.0, 0.0, 0.0]),
               np.array([0.0, 0.0, L_BASE])]

        r1 = L_UPPER * math.cos(s)
        z1 = L_BASE + L_UPPER * math.sin(s)
        pts.append(np.array([r1*cw, r1*sw, z1]))

        se = s + e
        r2 = r1 + L_FORE * math.cos(se)
        z2 = z1 + L_FORE * math.sin(se)
        pts.append(np.array([r2*cw, r2*sw, z2]))

        sew = se + wr
        r3 = r2 + L_HAND * math.cos(sew)
        z3 = z2 + L_HAND * math.sin(sew)
        pts.append(np.array([r3*cw, r3*sw, z3]))

        grip = self.angles['gripper']
        spread = grip * 0.022
        dr, dz = math.cos(sew), math.sin(sew)
        pr, pz = -dz, dr

        def finger(sign):
            fr = r3 + L_FINGER*dr + sign*spread*pr
            fz = z3 + L_FINGER*dz + sign*spread*pz
            return np.array([fr*cw, fr*sw, fz])

        return pts, finger(1), finger(-1)

    def is_safe(self):
        """Check if current pose keeps all joints above the table surface."""
        pts, lf, rf = self.fk()
        for p in pts:
            if p[2] < FLOOR_Z:
                return False
        if lf[2] < FLOOR_Z or rf[2] < FLOOR_Z:
            return False
        return True


# ═══════════════════════════════════════════════════════════
#  GESTURE KEYFRAMES — Expressive & slow
# ═══════════════════════════════════════════════════════════
# Principles applied: anticipation, follow-through, speed
# variation, pauses for emphasis, secondary motion.
#
# Format: ({'joint': angle, ...}, duration_seconds)

GESTURES = {

    # ── HAPPY ──────────────────────────────────────────────
    # Pure joy — bouncy celebration with increasing energy
    'happy': {
        'label': 'Happy', 'color': (255, 210, 50),
        'frames': [
            # Anticipation — slight crouch before burst
            ({'waist': 0,  'shoulder':-10, 'elbow':-25, 'wrist': 5,  'gripper':.3},  .30),
            # Burst up — excited rise
            ({'waist': 0,  'shoulder': 55, 'elbow':-55, 'wrist': 25, 'gripper':1},   .25),
            # Bounce 1 — down
            ({'waist': 20, 'shoulder': 18, 'elbow':-28, 'wrist': 5,  'gripper':.9},  .22),
            # Bounce 2 — up & bigger
            ({'waist':-25, 'shoulder': 65, 'elbow':-62, 'wrist': 30, 'gripper':1},   .25),
            # Bounce 3 — down
            ({'waist': 30, 'shoulder': 12, 'elbow':-22, 'wrist': 0,  'gripper':.8},  .22),
            # Bounce 4 — biggest!
            ({'waist':-35, 'shoulder': 75, 'elbow':-68, 'wrist': 38, 'gripper':1},   .30),
            # Hold the peak moment
            ({'waist':-30, 'shoulder': 72, 'elbow':-65, 'wrist': 35, 'gripper':1},   .40),
            # Celebratory wiggle
            ({'waist': 25, 'shoulder': 60, 'elbow':-58, 'wrist': 28, 'gripper':1},   .20),
            ({'waist':-20, 'shoulder': 62, 'elbow':-60, 'wrist': 30, 'gripper':1},   .20),
            ({'waist': 15, 'shoulder': 58, 'elbow':-55, 'wrist': 25, 'gripper':.9},  .20),
            # Gentle settle
            ({'waist': 5,  'shoulder': 30, 'elbow':-35, 'wrist': 12, 'gripper':.7},  .40),
            ({'waist': 0,  'shoulder':  0, 'elbow':  0, 'wrist':  0, 'gripper':.5},  .50),
        ],
    },

    # ── SURPRISE ───────────────────────────────────────────
    # Startle → freeze → slowly realize → trembling disbelief
    'surprise': {
        'label': 'Surprise', 'color': (255, 95, 70),
        'frames': [
            # STARTLE — snap back instantly (defensive)
            ({'waist': 0,  'shoulder':-25, 'elbow':-90, 'wrist':-25, 'gripper': 0},  .10),
            # Freeze — hold in retracted position, processing
            ({'waist': 0,  'shoulder':-25, 'elbow':-88, 'wrist':-24, 'gripper': 0},  .45),
            # Slowly start to look… what was that?
            ({'waist': 8,  'shoulder': 20, 'elbow':-50, 'wrist':-15, 'gripper':.4},  .50),
            # Lean forward — discovering
            ({'waist': 0,  'shoulder': 60, 'elbow':-22, 'wrist':-35, 'gripper':.8},  .45),
            # FULL SURPRISE — mouth agape
            ({'waist': 0,  'shoulder': 75, 'elbow':-18, 'wrist':-42, 'gripper': 1},  .35),
            # Hold in shock
            ({'waist': 0,  'shoulder': 75, 'elbow':-18, 'wrist':-42, 'gripper': 1},  .50),
            # Trembles of disbelief
            ({'waist': 6,  'shoulder': 77, 'elbow':-16, 'wrist':-40, 'gripper': 1},  .12),
            ({'waist':-6,  'shoulder': 73, 'elbow':-20, 'wrist':-44, 'gripper': 1},  .12),
            ({'waist': 4,  'shoulder': 76, 'elbow':-17, 'wrist':-41, 'gripper': 1},  .12),
            ({'waist':-4,  'shoulder': 74, 'elbow':-19, 'wrist':-43, 'gripper': 1},  .12),
            ({'waist': 3,  'shoulder': 75, 'elbow':-18, 'wrist':-42, 'gripper': 1},  .20),
            # Nervous look around
            ({'waist': 35, 'shoulder': 50, 'elbow':-30, 'wrist':-25, 'gripper':.6},  .40),
            ({'waist':-30, 'shoulder': 45, 'elbow':-28, 'wrist':-22, 'gripper':.5},  .40),
            # Slowly recover
            ({'waist': 0,  'shoulder': 15, 'elbow':-15, 'wrist': -5, 'gripper':.5},  .50),
            ({'waist': 0,  'shoulder':  0, 'elbow':  0, 'wrist':  0, 'gripper':.5},  .45),
        ],
    },

    # ── CHIRPY ─────────────────────────────────────────────
    # Energetic bird — rapid pecks, head tilts, excited clicks
    'chirpy': {
        'label': 'Chirpy', 'color': (80, 210, 80),
        'frames': [
            # Alert! Head up
            ({'waist': 0,  'shoulder': 55, 'elbow':-65, 'wrist': 30, 'gripper':.5},  .25),
            # Quick look left
            ({'waist': 30, 'shoulder': 50, 'elbow':-60, 'wrist': 35, 'gripper':.3},  .12),
            # Peck down!
            ({'waist': 25, 'shoulder': 20, 'elbow':-80, 'wrist': 55, 'gripper':.1},  .10),
            # Head up — got something
            ({'waist': 15, 'shoulder': 60, 'elbow':-55, 'wrist': 20, 'gripper':.7},  .15),
            # Quick look right
            ({'waist':-28, 'shoulder': 55, 'elbow':-58, 'wrist': 32, 'gripper':.4},  .12),
            # Peck down!
            ({'waist':-22, 'shoulder': 18, 'elbow':-82, 'wrist': 58, 'gripper':.0},  .10),
            # Chirp chirp — rapid gripper clicks while bobbing
            ({'waist':-10, 'shoulder': 58, 'elbow':-52, 'wrist': 25, 'gripper':.9},  .10),
            ({'waist': 5,  'shoulder': 48, 'elbow':-58, 'wrist': 30, 'gripper':.1},  .10),
            ({'waist': 15, 'shoulder': 62, 'elbow':-50, 'wrist': 22, 'gripper':.9},  .10),
            ({'waist':-5,  'shoulder': 45, 'elbow':-60, 'wrist': 35, 'gripper':.1},  .10),
            # Excited hop
            ({'waist': 0,  'shoulder': 72, 'elbow':-40, 'wrist': 10, 'gripper': 1},  .18),
            # Head tilt — curious
            ({'waist': 20, 'shoulder': 65, 'elbow':-50, 'wrist': 45, 'gripper':.6},  .30),
            # Another tilt
            ({'waist':-20, 'shoulder': 65, 'elbow':-50, 'wrist':-15, 'gripper':.6},  .30),
            # Settle
            ({'waist': 0,  'shoulder': 25, 'elbow':-25, 'wrist': 10, 'gripper':.5},  .35),
            ({'waist': 0,  'shoulder':  0, 'elbow':  0, 'wrist':  0, 'gripper':.5},  .40),
        ],
    },

    # ── FUNNY ──────────────────────────────────────────────
    # Slapstick comedy — exaggerated setup, pratfall, dizzy wobble
    'funny': {
        'label': 'Funny', 'color': (255, 105, 180),
        'frames': [
            # Dramatic pompous setup — "look at me"
            ({'waist': 0,   'shoulder': 90, 'elbow':-85, 'wrist': 90,  'gripper': 0},  .50),
            # Hold pose
            ({'waist': 0,   'shoulder': 90, 'elbow':-85, 'wrist': 88,  'gripper': 0},  .30),
            # PRATFALL — whoops! rapid tumble
            ({'waist': 90,  'shoulder':-15, 'elbow':-10, 'wrist':-50,  'gripper': 1},  .20),
            # Overshoot the other way
            ({'waist':-100, 'shoulder': 82, 'elbow':-88, 'wrist': 85,  'gripper': 1},  .25),
            # Dizzy spin 1
            ({'waist': 130, 'shoulder': 30, 'elbow':-40, 'wrist':-30,  'gripper':.5},  .30),
            # Dizzy spin 2
            ({'waist':-130, 'shoulder': 50, 'elbow':-60, 'wrist': 40,  'gripper':.5},  .30),
            # Wobble wobble — rubber arm
            ({'waist': 60,  'shoulder': 70, 'elbow':-75, 'wrist': 65,  'gripper': 1},  .22),
            ({'waist':-50,  'shoulder': 15, 'elbow':-20, 'wrist':-30,  'gripper': 0},  .22),
            ({'waist': 40,  'shoulder': 55, 'elbow':-55, 'wrist': 45,  'gripper': 1},  .22),
            # "Laughing" — rapid gripper open-close
            ({'waist': 10,  'shoulder': 45, 'elbow':-50, 'wrist': 20,  'gripper': 0},  .12),
            ({'waist':-10,  'shoulder': 48, 'elbow':-48, 'wrist': 22,  'gripper': 1},  .12),
            ({'waist': 8,   'shoulder': 43, 'elbow':-52, 'wrist': 18,  'gripper': 0},  .12),
            ({'waist':-8,   'shoulder': 46, 'elbow':-49, 'wrist': 21,  'gripper': 1},  .12),
            ({'waist': 5,   'shoulder': 44, 'elbow':-50, 'wrist': 20,  'gripper': 0},  .12),
            # Unsteady recovery
            ({'waist': 15,  'shoulder': 20, 'elbow':-25, 'wrist': 10,  'gripper':.5},  .35),
            ({'waist':-10,  'shoulder': 10, 'elbow':-15, 'wrist': 5,   'gripper':.5},  .30),
            ({'waist': 0,   'shoulder':  0, 'elbow':  0, 'wrist': 0,   'gripper':.5},  .45),
        ],
    },

    # ── WAVING ─────────────────────────────────────────────
    # Warm, graceful greeting — pageant-style wave
    'waving': {
        'label': 'Waving', 'color': (70, 130, 255),
        'frames': [
            # Graceful arm raise
            ({'waist': 0,  'shoulder': 35, 'elbow':-30, 'wrist':-10, 'gripper':.8},  .40),
            ({'waist': 0,  'shoulder': 70, 'elbow':-50, 'wrist':-22, 'gripper': 1},  .45),
            # First wave — wide and slow
            ({'waist': 35, 'shoulder': 72, 'elbow':-48, 'wrist':-18, 'gripper': 1},  .45),
            ({'waist':-35, 'shoulder': 68, 'elbow':-52, 'wrist':-25, 'gripper': 1},  .65),
            # Second wave — add wrist wiggle
            ({'waist': 38, 'shoulder': 74, 'elbow':-46, 'wrist':-10, 'gripper': 1},  .55),
            ({'waist':-38, 'shoulder': 66, 'elbow':-54, 'wrist':-30, 'gripper': 1},  .65),
            # Third wave — full energy
            ({'waist': 40, 'shoulder': 76, 'elbow':-44, 'wrist': -5, 'gripper': 1},  .50),
            ({'waist':-40, 'shoulder': 64, 'elbow':-56, 'wrist':-35, 'gripper': 1},  .60),
            # Slow down — final small wave
            ({'waist': 20, 'shoulder': 70, 'elbow':-50, 'wrist':-20, 'gripper': 1},  .50),
            ({'waist':-15, 'shoulder': 68, 'elbow':-50, 'wrist':-22, 'gripper': 1},  .50),
            # Hold — friendly pause
            ({'waist': 0,  'shoulder': 65, 'elbow':-48, 'wrist':-20, 'gripper':.9},  .40),
            # Graceful lower
            ({'waist': 0,  'shoulder': 35, 'elbow':-30, 'wrist':-10, 'gripper':.7},  .45),
            ({'waist': 0,  'shoulder':  0, 'elbow':  0, 'wrist':  0, 'gripper':.5},  .55),
        ],
    },

    # ── BITE ──────────────────────────────────────────────
    # Aggressive snake/dog strike — coil back, rapid lunge + snap, repeat
    'bite': {
        'label': 'Bite!', 'color': (200, 50, 50),
        'frames': [
            # Coil back — tensing up, mouth open
            ({'waist': 0,   'shoulder':-15, 'elbow':-80, 'wrist': 60,  'gripper': 1},   .25),
            # Hold — locked on target
            ({'waist': 0,   'shoulder':-15, 'elbow':-78, 'wrist': 58,  'gripper': 1},   .15),
            # STRIKE 1 — fast lunge forward + SNAP
            ({'waist': 5,   'shoulder': 50, 'elbow':-15, 'wrist':-20,  'gripper': 0},   .08),
            # Pull back quick
            ({'waist':-5,   'shoulder':-10, 'elbow':-75, 'wrist': 55,  'gripper':.8},   .12),
            # STRIKE 2 — different angle
            ({'waist':-12,  'shoulder': 55, 'elbow':-10, 'wrist':-25,  'gripper': 0},   .08),
            # Yank back
            ({'waist': 8,   'shoulder':-12, 'elbow':-78, 'wrist': 58,  'gripper':.9},   .12),
            # STRIKE 3 — biggest lunge
            ({'waist': 0,   'shoulder': 65, 'elbow': -5, 'wrist':-30,  'gripper': 0},   .07),
            # Hold the bite — shaking prey
            ({'waist': 20,  'shoulder': 62, 'elbow': -8, 'wrist':-25,  'gripper': 0},   .10),
            ({'waist':-20,  'shoulder': 60, 'elbow':-10, 'wrist':-28,  'gripper': 0},   .10),
            ({'waist': 15,  'shoulder': 63, 'elbow': -7, 'wrist':-22,  'gripper': 0},   .10),
            # Rapid fire snapping — machine gun bites
            ({'waist': 5,   'shoulder': 40, 'elbow':-25, 'wrist':-10,  'gripper': 1},   .06),
            ({'waist':-5,   'shoulder': 50, 'elbow':-12, 'wrist':-18,  'gripper': 0},   .06),
            ({'waist': 8,   'shoulder': 38, 'elbow':-28, 'wrist': -8,  'gripper': 1},   .06),
            ({'waist':-8,   'shoulder': 52, 'elbow':-10, 'wrist':-20,  'gripper': 0},   .06),
            ({'waist': 3,   'shoulder': 42, 'elbow':-22, 'wrist':-12,  'gripper': 1},   .06),
            ({'waist':-3,   'shoulder': 48, 'elbow':-14, 'wrist':-16,  'gripper': 0},   .06),
            ({'waist': 6,   'shoulder': 36, 'elbow':-30, 'wrist': -5,  'gripper': 1},   .06),
            ({'waist':-6,   'shoulder': 55, 'elbow': -8, 'wrist':-22,  'gripper': 0},   .06),
            # Final big snap + hold
            ({'waist': 0,   'shoulder': 60, 'elbow': -5, 'wrist':-28,  'gripper': 0},   .10),
            ({'waist': 0,   'shoulder': 58, 'elbow': -6, 'wrist':-26,  'gripper': 0},   .30),
            # Slowly release and pull back — satisfied
            ({'waist': 0,   'shoulder': 40, 'elbow':-30, 'wrist':-10,  'gripper':.5},   .30),
            ({'waist': 0,   'shoulder': 10, 'elbow':-50, 'wrist': 20,  'gripper':.5},   .35),
            ({'waist': 0,   'shoulder':  0, 'elbow':  0, 'wrist':  0,  'gripper':.5},   .40),
        ],
    },

    # ── CURIOUS ───────────────────────────────────────────
    # Cautious investigation — head tilts, slow approach, sniffing
    'curious': {
        'label': 'Curious', 'color': (100, 180, 220),
        'frames': [
            # Perk up — something caught attention
            ({'waist': 0,  'shoulder': 40, 'elbow':-45, 'wrist': 10,  'gripper':.4},   .35),
            # Head tilt left — "what's that?"
            ({'waist': 25, 'shoulder': 45, 'elbow':-40, 'wrist': 40,  'gripper':.3},   .45),
            # Hold tilt — processing
            ({'waist': 25, 'shoulder': 45, 'elbow':-40, 'wrist': 38,  'gripper':.3},   .30),
            # Head tilt right
            ({'waist':-20, 'shoulder': 42, 'elbow':-42, 'wrist':-30,  'gripper':.3},   .45),
            # Lean forward cautiously — investigating
            ({'waist': 0,  'shoulder': 60, 'elbow':-20, 'wrist':-15,  'gripper':.6},   .50),
            # Closer… sniffing (tiny gripper pulses)
            ({'waist': 5,  'shoulder': 68, 'elbow':-15, 'wrist':-20,  'gripper':.2},   .15),
            ({'waist':-3,  'shoulder': 70, 'elbow':-12, 'wrist':-18,  'gripper':.5},   .15),
            ({'waist': 4,  'shoulder': 69, 'elbow':-14, 'wrist':-22,  'gripper':.2},   .15),
            # Pull back — startled
            ({'waist': 0,  'shoulder': 20, 'elbow':-60, 'wrist': 30,  'gripper':.1},   .20),
            # Pause… safe?
            ({'waist': 0,  'shoulder': 22, 'elbow':-58, 'wrist': 28,  'gripper':.2},   .35),
            # Approach again — braver this time
            ({'waist': 10, 'shoulder': 65, 'elbow':-18, 'wrist':-10,  'gripper':.7},   .40),
            # Big head tilt — fascinated
            ({'waist': 30, 'shoulder': 60, 'elbow':-22, 'wrist': 50,  'gripper':.5},   .50),
            # Settle back — satisfied
            ({'waist': 0,  'shoulder': 30, 'elbow':-35, 'wrist': 10,  'gripper':.5},   .40),
            ({'waist': 0,  'shoulder':  0, 'elbow':  0, 'wrist':  0,  'gripper':.5},   .45),
        ],
    },

    # ── SAD ───────────────────────────────────────────────
    # Deflated, drooping, heavy — slow and melancholic
    'sad': {
        'label': 'Sad', 'color': (80, 90, 160),
        'frames': [
            # Start normal, then slowly droop
            ({'waist': 0,  'shoulder': 20, 'elbow':-20, 'wrist':  0,  'gripper':.5},   .50),
            # Sinking down… heavy
            ({'waist': 0,  'shoulder': -5, 'elbow':-70, 'wrist': 50,  'gripper':.2},   .70),
            # Head hangs low
            ({'waist': 0,  'shoulder':-15, 'elbow':-80, 'wrist': 65,  'gripper':.1},   .60),
            # Slow sway left — listless
            ({'waist': 15, 'shoulder':-12, 'elbow':-78, 'wrist': 60,  'gripper':.1},   .55),
            # Slow sway right
            ({'waist':-15, 'shoulder':-14, 'elbow':-80, 'wrist': 62,  'gripper':.1},   .55),
            # Tiny lift — maybe hope?
            ({'waist': 0,  'shoulder':  5, 'elbow':-55, 'wrist': 30,  'gripper':.3},   .50),
            # Nope, falls back down
            ({'waist': 5,  'shoulder':-18, 'elbow':-85, 'wrist': 68,  'gripper':.0},   .60),
            # Deep sigh — slow rise and fall
            ({'waist': 0,  'shoulder': -8, 'elbow':-72, 'wrist': 55,  'gripper':.2},   .45),
            ({'waist': 0,  'shoulder':-20, 'elbow':-88, 'wrist': 70,  'gripper':.0},   .50),
            # One more sway
            ({'waist':-10, 'shoulder':-18, 'elbow':-85, 'wrist': 68,  'gripper':.1},   .50),
            ({'waist': 10, 'shoulder':-16, 'elbow':-82, 'wrist': 65,  'gripper':.1},   .50),
            # Very slowly come back up — reluctant
            ({'waist': 0,  'shoulder': -5, 'elbow':-60, 'wrist': 40,  'gripper':.3},   .60),
            ({'waist': 0,  'shoulder':  0, 'elbow':-30, 'wrist': 15,  'gripper':.4},   .55),
            ({'waist': 0,  'shoulder':  0, 'elbow':  0, 'wrist':  0,  'gripper':.5},   .50),
        ],
    },

    # ── EXCITED ───────────────────────────────────────────
    # Pure energy — bouncing, spinning, chattering, can't contain it
    'excited': {
        'label': 'Excited', 'color': (255, 130, 0),
        'frames': [
            # PERK UP — instant alert
            ({'waist': 0,   'shoulder': 60, 'elbow':-50, 'wrist': 15,  'gripper': 1},   .15),
            # Bounce 1 — up up up!
            ({'waist': 20,  'shoulder': 25, 'elbow':-30, 'wrist':  5,  'gripper':.5},   .12),
            ({'waist':-20,  'shoulder': 70, 'elbow':-55, 'wrist': 20,  'gripper': 1},   .12),
            # Bounce 2 — bigger!
            ({'waist': 30,  'shoulder': 20, 'elbow':-25, 'wrist':  0,  'gripper':.3},   .12),
            ({'waist':-30,  'shoulder': 78, 'elbow':-60, 'wrist': 25,  'gripper': 1},   .12),
            # Spin! — full waist rotation
            ({'waist': 90,  'shoulder': 50, 'elbow':-45, 'wrist': 10,  'gripper':.8},   .20),
            ({'waist':-90,  'shoulder': 55, 'elbow':-48, 'wrist': 12,  'gripper':.8},   .25),
            # Chattering — rapid gripper clicks while bobbing
            ({'waist': 15,  'shoulder': 65, 'elbow':-40, 'wrist': 18,  'gripper': 0},   .07),
            ({'waist':-10,  'shoulder': 55, 'elbow':-50, 'wrist': 12,  'gripper': 1},   .07),
            ({'waist': 12,  'shoulder': 68, 'elbow':-38, 'wrist': 20,  'gripper': 0},   .07),
            ({'waist':-8,   'shoulder': 52, 'elbow':-52, 'wrist': 10,  'gripper': 1},   .07),
            ({'waist': 10,  'shoulder': 70, 'elbow':-35, 'wrist': 22,  'gripper': 0},   .07),
            ({'waist':-5,   'shoulder': 58, 'elbow':-48, 'wrist': 14,  'gripper': 1},   .07),
            # BIG EXTENSION — reaching for the sky!
            ({'waist': 0,   'shoulder': 85, 'elbow':-75, 'wrist': 35,  'gripper': 1},   .25),
            # Hold peak — trembling with excitement
            ({'waist': 8,   'shoulder': 87, 'elbow':-73, 'wrist': 37,  'gripper': 1},   .10),
            ({'waist':-8,   'shoulder': 83, 'elbow':-77, 'wrist': 33,  'gripper': 1},   .10),
            ({'waist': 5,   'shoulder': 86, 'elbow':-74, 'wrist': 36,  'gripper': 1},   .10),
            # Quick spin celebration
            ({'waist': 120, 'shoulder': 50, 'elbow':-45, 'wrist': 15,  'gripper':.7},   .25),
            ({'waist':-60,  'shoulder': 55, 'elbow':-48, 'wrist': 12,  'gripper':.7},   .20),
            # Wind down — still buzzing
            ({'waist': 15,  'shoulder': 35, 'elbow':-30, 'wrist':  8,  'gripper':.6},   .25),
            ({'waist':-10,  'shoulder': 20, 'elbow':-20, 'wrist':  5,  'gripper':.5},   .30),
            ({'waist': 0,   'shoulder':  0, 'elbow':  0, 'wrist':  0,  'gripper':.5},   .40),
        ],
    },
}


# ═══════════════════════════════════════════════════════════
#  PRESET POSITIONS
# ═══════════════════════════════════════════════════════════
PRESETS = {
    'vertical': {
        'label': 'Vertical', 'color': (180, 180, 200),
        'angles': {'waist': 0, 'shoulder': -25, 'elbow': -55, 'wrist': 0, 'gripper': 0.5},
    },
    'forward': {
        'label': 'Forward', 'color': (130, 170, 210),
        'angles': {'waist': 0, 'shoulder': 0, 'elbow': 0, 'wrist': 0, 'gripper': 0.5},
    },
    'rest': {
        'label': 'Rest', 'color': (140, 160, 140),
        'angles': {'waist': 0, 'shoulder': 50, 'elbow': -50, 'wrist': 0, 'gripper': 0.5},
    },
    'reach': {
        'label': 'Reach Down', 'color': (170, 140, 180),
        'angles': {'waist': 0, 'shoulder': 90, 'elbow': 0, 'wrist': 0, 'gripper': 0.8},
    },
}
PRESET_ORDER = ['vertical', 'forward', 'rest', 'reach']


class GesturePlayer:
    """Interpolates through gesture keyframes with smoothstep easing."""

    def __init__(self, robot: PX100Arm):
        self.robot = robot
        self.playing = False
        self.gesture_name = None
        self._frames = []
        self._idx = 0
        self._t0 = 0.0
        self._prev = {}

    def play(self, name):
        if name not in GESTURES:
            return
        self.gesture_name = name
        self._frames = GESTURES[name]['frames']
        self._prev = dict(self.robot.angles)
        self._idx = 0
        self._t0 = time.time()
        self.playing = True

    def stop(self):
        self.playing = False
        self.gesture_name = None

    def update(self):
        if not self.playing:
            return
        if self._idx >= len(self._frames):
            self.stop()
            return

        target, dur = self._frames[self._idx]
        elapsed = time.time() - self._t0
        t = min(1.0, elapsed / dur) if dur > 0 else 1.0
        # Smooth ease-in-out
        t = t * t * (3.0 - 2.0 * t)

        for k in JOINT_NAMES:
            if k in target:
                a = self._prev.get(k, 0.0)
                b = float(target[k])
                self.robot.angles[k] = a + (b - a) * t

        if elapsed >= dur:
            self._prev = dict(self.robot.angles)
            self._idx += 1
            self._t0 = time.time()


# ── Optional hardware ──────────────────────────────────────

class HardwareLink:
    """Sends joint angles to real DYNAMIXEL servos via U2D2."""

    MOTOR_IDS = {
        'waist': 1, 'shoulder': 2, 'elbow': 3, 'wrist': 4, 'gripper': 5,
    }
    ADDR_OP_MODE = 11
    ADDR_TORQUE  = 64
    ADDR_GOAL    = 116

    # Gripper calibration (XL430 position units)
    GRIP_CLOSED = 1500
    GRIP_OPEN   = 2600

    def __init__(self):
        self.connected = False
        self._port = None
        self._pkt = None

    def connect(self, port_path, baud=1000000):
        try:
            from dynamixel_sdk import PortHandler, PacketHandler
        except ImportError:
            return False
        try:
            self._port = PortHandler(port_path)
            self._pkt  = PacketHandler(2.0)
            if not self._port.openPort():
                return False
            self._port.setBaudRate(baud)

            # Configure gripper: ensure position control mode (mode 3)
            grip_id = self.MOTOR_IDS['gripper']
            self._pkt.write1ByteTxRx(self._port, grip_id, self.ADDR_TORQUE, 0)
            import time; time.sleep(0.15)
            self._pkt.write1ByteTxRx(self._port, grip_id, self.ADDR_OP_MODE, 3)
            import time; time.sleep(0.15)

            # Enable torque on all motors
            for mid in self.MOTOR_IDS.values():
                self._pkt.write1ByteTxRx(self._port, mid, self.ADDR_TORQUE, 1)
            self.connected = True
            return True
        except Exception:
            return False

    def disconnect(self):
        if self._port and self.connected:
            for mid in self.MOTOR_IDS.values():
                self._pkt.write1ByteTxRx(self._port, mid, self.ADDR_TORQUE, 0)
            self._port.closePort()
        self.connected = False

    def send(self, angles):
        if not self.connected:
            return
        try:
            for name, mid in self.MOTOR_IDS.items():
                val = angles[name]
                if name == 'gripper':
                    pos = int(self.GRIP_CLOSED + val * (self.GRIP_OPEN - self.GRIP_CLOSED))
                else:
                    pos = int(2048 + val / 0.088)
                pos = max(0, min(4095, pos))
                self._pkt.write4ByteTxRx(self._port, mid, self.ADDR_GOAL, pos)
        except Exception:
            self.connected = False
