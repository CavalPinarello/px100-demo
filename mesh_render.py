"""3D mesh renderer for PX100 using actual Interbotix STL files + URDF kinematics."""

import os, math
import numpy as np

MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')

# ── Rotation helpers ──────────────────────────────────────

def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)

# ── Mesh loader ───────────────────────────────────────────

def _load_stl(name, decimate=1):
    """Load STL file, decimate, apply URDF rpy(0,0,π/2) and mm→m scale."""
    path = os.path.join(MESH_DIR, name)
    if not os.path.exists(path):
        return None, None
    try:
        from stl import mesh as stl_mesh
    except ImportError:
        return None, None

    m = stl_mesh.Mesh.from_file(path)
    verts = m.vectors[::decimate].copy().astype(np.float32)
    verts *= 0.001  # mm to m

    # All PX100 meshes use rpy(0, 0, π/2) in the URDF
    R = _rz(math.pi / 2)
    verts = np.einsum('ij,nkj->nki', R, verts)

    # Pre-compute face normals
    e1 = verts[:, 1] - verts[:, 0]
    e2 = verts[:, 2] - verts[:, 0]
    normals = np.cross(e1, e2)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen < 1e-10] = 1.0
    normals /= nlen

    return verts, normals


class PX100MeshModel:
    """Loads PX100 STL meshes and renders them with flat shading."""

    # Robot body colors — brighter for better visibility
    BASE_COLOR = np.array([110, 115, 130], dtype=np.float32)
    # Slightly different color for servos/joints
    SERVO_COLOR = np.array([85, 90, 110], dtype=np.float32)

    def __init__(self, decimate=4):
        self.ready = False
        self._meshes = {}
        self._load_all(decimate)

    def _load_all(self, dec):
        files = {
            'base':        ('px100_1_base.stl', dec),
            'shoulder':    ('px100_2_shoulder.stl', dec),
            'upper_arm':   ('px100_3_upper_arm.stl', dec),
            'forearm':     ('px100_4_forearm.stl', dec),
            'gripper':     ('px100_5_gripper.stl', max(dec, 5)),
            'gripper_bar': ('px100_7_gripper_bar.stl', dec),
            'finger':      ('px100_8_gripper_finger.stl', dec),
        }
        for key, (fname, d) in files.items():
            v, n = _load_stl(fname, d)
            if v is not None:
                self._meshes[key] = (v, n)
        self.ready = len(self._meshes) >= 4

    # ── Public API ────────────────────────────────────────

    def render(self, surf, cam, angles, pygame):
        """Render the full PX100 arm using STL meshes."""
        if not self.ready:
            return

        w  = math.radians(angles['waist'])
        sh = math.radians(angles['shoulder'])
        el = math.radians(angles['elbow'])
        wr = math.radians(angles['wrist'])
        grip = angles['gripper']

        # Collect all visible triangles: (screen_pts[3,2], depth, color_rgb)
        tris = []

        # Camera view direction (for back-face culling)
        az_r = math.radians(cam.az)
        el_r = math.radians(cam.el)
        view = np.array([
            -math.sin(az_r) * math.cos(el_r),
             math.cos(az_r) * math.cos(el_r),
             math.sin(el_r)
        ], dtype=np.float32)

        # Light direction
        light = np.array([0.35, -0.5, 0.75], dtype=np.float32)
        light /= np.linalg.norm(light)

        # ── Build kinematic chain (URDF-based) ────────
        # Base: at world origin
        R0 = np.eye(3, dtype=np.float32)
        p0 = np.zeros(3, dtype=np.float32)
        self._add('base', R0, p0, tris, cam, view, light, self.BASE_COLOR)

        # Waist joint: Rz(waist) at (0, 0, 0.05085)
        p_waist = np.array([0, 0, 0.05085], dtype=np.float32)
        R_waist = _rz(w)

        # Shoulder link mesh at (0, 0, -0.0022) in waist frame
        p_sh_mesh = p_waist + R_waist @ np.array([0, 0, -0.0022], dtype=np.float32)
        self._add('shoulder', R_waist, p_sh_mesh, tris, cam, view, light, self.SERVO_COLOR)

        # Shoulder joint: Ry(shoulder) at (0, 0, 0.04225) from waist
        p_sh = p_waist + R_waist @ np.array([0, 0, 0.04225], dtype=np.float32)
        R_sh = R_waist @ _ry(sh)

        # Upper arm mesh
        self._add('upper_arm', R_sh, p_sh, tris, cam, view, light, self.BASE_COLOR)

        # Elbow joint: Ry(elbow) at (0.035, 0, 0.1) from shoulder
        p_el = p_sh + R_sh @ np.array([0.035, 0, 0.1], dtype=np.float32)
        R_el = R_sh @ _ry(el)

        # Forearm mesh
        self._add('forearm', R_el, p_el, tris, cam, view, light, self.BASE_COLOR)

        # Wrist joint: Ry(wrist) at (0.1, 0, 0) from elbow
        p_wr = p_el + R_el @ np.array([0.1, 0, 0], dtype=np.float32)
        R_wr = R_el @ _ry(wr)

        # Gripper body mesh
        self._add('gripper', R_wr, p_wr, tris, cam, view, light, self.SERVO_COLOR)

        # EE arm at (0.063, 0, 0) from wrist
        p_ee = p_wr + R_wr @ np.array([0.063, 0, 0], dtype=np.float32)

        # Gripper bar mesh: offset (-0.063, 0, 0) from ee
        p_bar = p_ee + R_wr @ np.array([-0.063, 0, 0], dtype=np.float32)
        self._add('gripper_bar', R_wr, p_bar, tris, cam, view, light, self.BASE_COLOR)

        # Fingers: at (0.023, 0, 0) from gripper_bar
        p_fingers = p_ee + R_wr @ np.array([0.023, 0, 0], dtype=np.float32)
        finger_y = 0.015 + grip * (0.037 - 0.015)

        # Left finger: rpy(π, π, 0), prismatic +Y
        p_lf = p_fingers + R_wr @ np.array([0, finger_y + 0.005, 0], dtype=np.float32)
        R_lf = R_wr @ _ry(math.pi) @ _rx(math.pi)
        self._add('finger', R_lf, p_lf, tris, cam, view, light, self.SERVO_COLOR)

        # Right finger: rpy(0, π, 0), prismatic -Y
        p_rf = p_fingers + R_wr @ np.array([0, -finger_y - 0.005, 0], dtype=np.float32)
        R_rf = R_wr @ _ry(math.pi)
        self._add('finger', R_rf, p_rf, tris, cam, view, light, self.SERVO_COLOR)

        # ── Sort by depth (painter's algorithm) and draw ──
        tris.sort(key=lambda t: -t[1])

        for screen_pts, _, color in tris:
            pygame.draw.polygon(surf, color, screen_pts)

    # ── Internal ──────────────────────────────────────────

    def _add(self, name, R, pos, tris, cam, view, light, base_color):
        """Transform mesh, cull back faces, shade, project, collect triangles."""
        if name not in self._meshes:
            return
        verts, normals = self._meshes[name]

        # Transform to world space
        world_v = np.einsum('ij,nkj->nki', R, verts) + pos
        world_n = (R @ normals.T).T

        # Back-face culling
        dots = world_n @ view
        visible = dots < 0.05
        if not np.any(visible):
            return

        vis_v = world_v[visible]
        vis_n = world_n[visible]

        # Flat shading
        light_val = np.abs(vis_n @ light)
        brightness = np.clip(0.3 + 0.7 * light_val, 0.2, 1.0)

        # Project all vertices using vectorized camera projection
        flat_pts = vis_v.reshape(-1, 3)
        screen, depths = _batch_project(flat_pts, cam)
        screen = screen.reshape(-1, 3, 2)
        tri_depths = depths.reshape(-1, 3).mean(axis=1)

        # Build triangle list
        for i in range(len(vis_v)):
            b = brightness[i]
            c = tuple(int(min(255, v * b)) for v in base_color)
            pts = [(int(screen[i, j, 0]), int(screen[i, j, 1])) for j in range(3)]
            tris.append((pts, float(tri_depths[i]), c))


def _batch_project(points, cam):
    """Vectorized projection: Nx3 points → (Nx2 screen coords, N depths)."""
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    z = points[:, 2].astype(np.float64)

    a = math.radians(cam.az)
    ca, sa = math.cos(a), math.sin(a)
    x1 = x * ca - y * sa
    y1 = x * sa + y * ca

    e = math.radians(cam.el)
    ce, se = math.cos(e), math.sin(e)
    y2 = y1 * ce - z * se
    z2 = y1 * se + z * ce

    d = np.maximum(2.0 + y2, 0.05)
    f = cam.fov * cam.zoom / d

    sx = (cam.cx + x1 * f).astype(np.int32)
    sy = (cam.cy - z2 * f).astype(np.int32)

    return np.column_stack([sx, sy]), d
