"""Gripper B (Waveshare, servo CF35-12) — cinématique exacte pour la sim MuJoCo.

Rig V3 (2026-06-10). Le V2 (commit 95ac0f9) avait DEUX pivots inversés bout
pour bout et une loi linéaire approchée (dérive ~3mm, mâchoires qui tournent).
L'extraction numérique des alésages depuis les STL (2026-06-10) a révélé la
vraie topologie — un PARALLÉLOGRAMME parfait par côté (pince à mâchoires
parallèles classique) :

    châssis (body)   : pivots A (levier-engrenage) et G (petit levier)
    levier-engrenage : A → P1 (43.2 mm)   [mesh lever_top_*]
    petit levier     : G → P2 (43.2 mm)   [mesh lever_bottom_*]
    mâchoire (coupleur) : P1 → P2 (29.7 mm = |A→G|)  [mesh gripper_*]

    |A→P1| = |G→P2| et |A→G| = |P1→P2|, directions parallèles au repos
    → parallélogramme → la mâchoire TRANSLATE sans tourner.

Cascade exacte (par côté) : θ_bot = θ_gear ; θ_jaw = −θ_gear.
Engrenages 21 dents 1:1 : servo → gear_L → gear_R, donc
θ_gear_L = +θ_in (convention visuelle V2 conservée, validée metrox),
θ_gear_R = −θ_in, θ_servo_cog = −θ_in (sens physique d'engrènement).

Le « 4-bar non-parallélogramme top=67.8/bot=31.1 » du message de commit V2
était une erreur de mesure (67.8 mm = la diagonale A→P2 ; 31.1 = diagonale
P1→G). Les 3 imperfections documentées de V2 (dérive 3mm, levier bas en
miroir, mâchoires qui tournent) découlent toutes des pivots inversés.

Validation (invariants numériques mesurés DANS MuJoCo, V2 vs V3) :
    python sim/gripper_b_kinematics.py

Viewer interactif (le gripper s'anime en boucle via la cascade exacte,
caméra libre à la souris ; fermer la fenêtre pour quitter) :
    python sim/gripper_b_kinematics.py --view
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes", "gripper_b_parts")

# ── Topologie VRAIE (extraite des alésages STL 2026-06-10, mm, plan XY local).
#    Sert de référence attendue : extract_geometry() doit la retrouver à <0.1mm
#    depuis les STL, sinon elle se rejette.
EXPECTED_MM = {
    "S":    (1.489, 514.005),     # axe servo cog (bore sevo_cog + scene.xml V2)
    "A_L":  (3.489, 534.910),     # pivot châssis du levier-engrenage gauche
    "A_R":  (24.489, 534.910),
    "G_L":  (-17.511, 513.910),   # pivot châssis du petit levier gauche
    "G_R":  (45.489, 513.910),
    "P1_L": (-39.711, 534.910),   # jonction levier-engrenage ↔ mâchoire
    "P1_R": (67.689, 534.910),
    "P2_L": (-60.711, 513.910),   # jonction petit levier ↔ mâchoire
    "P2_R": (88.689, 513.910),
}

# Signes V2 (variante B, commit 95ac0f9) — gardés UNIQUEMENT pour le
# before/after de la suite de validation.
V2_SIGNS = {"grip_servo": +1, "grip_top_L": +1, "grip_top_R": -1,
            "grip_bot_L": -1, "grip_bot_R": +1, "grip_jaw_L": +1, "grip_jaw_R": -1}

# Cascade V3 exacte (parallélogramme + chaîne d'engrenages 1:1).
# Convention d'entrée θ_in inchangée vs V2 : θ_in = (action[5]-73)/113.3,
# 0 = repos/ouvert, -0.6 = fermé (calibration visuelle metrox conservée).
V3_SIGNS = {"grip_servo": -1, "grip_top_L": +1, "grip_top_R": -1,
            "grip_bot_L": +1, "grip_bot_R": -1, "grip_jaw_L": -1, "grip_jaw_R": +1}

THETA_RANGE = (-1.6, 1.6)  # range des joints scene.xml (rad) — couvre la fermeture complète

# Faces internes des doigts (mm, coords assembly) : la zone de préhension est à
# Y > 540 (les pivots vivent à Y 513-535). Sert à mesurer le gap au repos.
FINGER_Y_MIN_MM = 540.0


def solve(theta_in: float) -> dict:
    """θ d'entrée (rad) → qpos exacts des 7 joints du rig V3.

    Convention : 0 = repos export CAD (bras horizontaux = ouverture géométrique
    max, gap faces ~84.2mm) ; sens négatif = fermeture ; fermeture COMPLÈTE
    (faces qui se touchent) à θ ≈ -theta_close() ≈ -1.545 rad (88.5°)."""
    return {j: s * float(theta_in) for j, s in V3_SIGNS.items()}


# ── Conversions gap ↔ θ ↔ angle servo (source unique pour la page SIM) ───────
# Constantes figées depuis l'extraction STL 2026-06-10 (validées <0.0011mm) :
GAP_REST_MM = 84.17   # gap faces internes au repos = ouverture géométrique max
ARM_MM = 43.20        # longueur des bras du parallélogramme (|A→P1| = |G→P2|)


def gap_mm_from_theta(theta: float) -> float:
    """Gap des faces internes (mm) pour un θ de levier (course X = r·(1-cosθ) par côté)."""
    return GAP_REST_MM - 2.0 * ARM_MM * (1.0 - float(np.cos(theta)))


def theta_for_gap_mm(gap_mm: float) -> float:
    """θ (négatif = fermé) qui produit un gap donné. Clampe aux bornes physiques."""
    x = 1.0 - (GAP_REST_MM - float(gap_mm)) / (2.0 * ARM_MM)
    return -float(np.arccos(np.clip(x, -1.0, 1.0)))


def theta_from_servo_deg(a_deg: float, synthetic: bool = False) -> float:
    """Angle servo du dataset → θ levier, conventions UNIFIÉES (2026-06-10).

    Convention RÉELLE (datasets robot + VR) : ferme en DESCENDANT vers 73°
    (73=fermé, 115=ouvert, VR descend jusqu'à ~49 = serrage). Ancrage provisoire
    documenté (Gap #243 le recalibrera au pied à coulisse) :
      a=115° → θ=0 (repos = ouverture max) ; a=78° → gap=25mm (le cube bloque).
    Convention SYNTHÉTIQUE héritée (mtc_to_lerobot, INVERSÉE) : 73=ouvert, 5=fermé
    — détectée par min(action[5]) < 50 sur l'épisode.
    """
    theta_25 = theta_for_gap_mm(25.0)  # ≈ -1.250 rad
    if synthetic:
        u = (73.0 - float(a_deg)) / (73.0 - 5.0)      # 0=ouvert → 1=fermé
        u_full = u * (115.0 - 73.0) / (115.0 - 78.0)  # même profondeur max que le réel
    else:
        u_full = (115.0 - float(a_deg)) / (115.0 - 78.0)
    u_full = float(np.clip(u_full, 0.0, (115.0 - 73.0) / (115.0 - 78.0)))
    return theta_25 * u_full


def jaw_face_gap_rest_mm() -> float:
    """Gap au repos entre les faces internes des doigts, mesuré dans les STL."""
    L = _read_stl_vertices(os.path.join(ASSETS, "gripper_L.stl"))
    R = _read_stl_vertices(os.path.join(ASSETS, "gripper_R.stl"))
    xL = L[L[:, 1] > FINGER_Y_MIN_MM][:, 0].max()
    xR = R[R[:, 1] > FINGER_Y_MIN_MM][:, 0].min()
    return float(xR - xL)


def theta_close(geom: dict | None = None) -> float:
    """θ (rad, valeur positive) où les faces internes se touchent (gap = 0).

    Parallélogramme : course X par côté = r_bras·(1-cosθ) →
    gap(θ) = gap0 - 2·r_bras·(1-cosθ) = 0 → θ = arccos(1 - gap0/(2·r_bras))."""
    g = geom or extract_geometry()
    r_arm = float(np.hypot(*(g["P1_L"] - g["A_L"])))  # m
    gap0 = jaw_face_gap_rest_mm() / 1000.0            # m
    return float(np.arccos(1.0 - gap0 / (2.0 * r_arm)))


# ──────────────────────────────────────────────────────────────────────────────
# Extraction des alésages depuis les STL (binaire Fusion, coords assembly mm)
# ──────────────────────────────────────────────────────────────────────────────

def _read_stl_vertices(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(80)
        (n_tri,) = struct.unpack("<I", f.read(4))
        raw = np.frombuffer(f.read(n_tri * 50), dtype=np.uint8)
    tri = raw.reshape(n_tri, 50)
    verts = np.frombuffer(tri[:, 12:48].tobytes(), dtype="<f4").reshape(-1, 3)
    return np.unique(np.round(verts.astype(np.float64), 4), axis=0)


def _fit_circle(xy: np.ndarray):
    """Fit Kasa (moindres carrés algébriques) → (centre, rayon, rms)."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(c + cx * cx + cy * cy))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return np.array([cx, cy]), r, rms


def find_bores(stl_path: str, r_min=0.8, r_max=5.0, rms_max=0.08,
               min_pts=10, cluster_radius=1.5) -> list[dict]:
    """Centres des alésages (axes Z) d'une pièce extrudée, en mm.

    En projection XY un alésage est une courbe fermée ISOLÉE du contour
    externe : clustering par proximité (1.5mm) puis fit de cercle ; le contour
    (levier + dents d'engrenage) échoue au fit (rms) ou au rayon → rejeté.
    """
    pts = np.unique(np.round(_read_stl_vertices(stl_path)[:, :2], 2), axis=0)
    cell = np.floor(pts / cluster_radius).astype(np.int64)
    from collections import defaultdict
    grid = defaultdict(list)
    for i, c in enumerate(map(tuple, cell)):
        grid[c].append(i)
    seen = np.zeros(len(pts), dtype=bool)
    bores = []
    for seed in range(len(pts)):
        if seen[seed]:
            continue
        stack, members = [seed], []
        seen[seed] = True
        while stack:
            i = stack.pop()
            members.append(i)
            ci = cell[i]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in grid.get((ci[0] + dx, ci[1] + dy), ()):
                        if not seen[j] and np.hypot(*(pts[j] - pts[i])) <= cluster_radius:
                            seen[j] = True
                            stack.append(j)
        if len(members) < min_pts:
            continue
        center, r, rms = _fit_circle(pts[members])
        if r_min <= r <= r_max and rms <= rms_max:
            bores.append({"center_mm": center, "radius_mm": r, "rms_mm": rms,
                          "n_pts": len(members)})
    return bores


def extract_geometry(verbose: bool = False) -> dict:
    """Ré-extrait les 9 points depuis les STL et les CONFRONTE à EXPECTED_MM.

    Auto-validation totale : chaque point doit être retrouvé dans CHAQUE pièce
    qui le porte (jonctions partagées comprises), à <0.1mm. Vérifie aussi les
    identités du parallélogramme. Lève ValueError au moindre écart.
    """
    spec = {  # point → [(fichier STL, ...)] qui doivent le porter
        "S": ["sevo_cog"],
        "A_L": ["body"], "A_R": ["body"],
        "G_L": ["body", "lever_bottom_L"], "G_R": ["body", "lever_bottom_R"],
        "P1_L": ["lever_top_L", "gripper_L"], "P1_R": ["lever_top_R", "gripper_R"],
        "P2_L": ["lever_bottom_L", "gripper_L"], "P2_R": ["lever_bottom_R", "gripper_R"],
    }
    parts = sorted({p for files in spec.values() for p in files})
    bores = {p: find_bores(os.path.join(ASSETS, f"{p}.stl")) for p in parts}
    geom, checks = {}, []
    for point, files in spec.items():
        target = np.array(EXPECTED_MM[point])
        found = []
        for p in files:
            if not bores[p]:
                raise ValueError(f"aucun alésage trouvé dans {p}.stl — extraction rejetée")
            dists = [float(np.hypot(*(b["center_mm"] - target))) for b in bores[p]]
            i = int(np.argmin(dists))
            checks.append((f"{point} dans {p}", dists[i]))
            if dists[i] > 0.1:
                raise ValueError(f"{point} introuvable dans {p}.stl "
                                 f"(plus proche à {dists[i]:.3f}mm) — extraction rejetée")
            found.append(bores[p][i]["center_mm"])
        geom[point] = np.mean(found, axis=0) / 1000.0  # m

    # Identités du parallélogramme (par côté) + engrènement
    for s in ("L", "R"):
        A, G = geom[f"A_{s}"], geom[f"G_{s}"]
        P1, P2 = geom[f"P1_{s}"], geom[f"P2_{s}"]
        crank1, crank2 = np.hypot(*(P1 - A)), np.hypot(*(P2 - G))
        ground, coupler = np.hypot(*(G - A)), np.hypot(*(P2 - P1))
        checks.append((f"[{s}] |A→P1|−|G→P2| (bras égaux)", abs(crank1 - crank2) * 1000))
        checks.append((f"[{s}] |A→G|−|P1→P2| (châssis=coupleur)", abs(ground - coupler) * 1000))
        par = np.hypot(*((P1 - A) - (P2 - G)))  # directions parallèles au repos
        checks.append((f"[{s}] parallélisme bras au repos", par * 1000))
        for label, err in checks[-3:]:
            if err > 0.1:
                raise ValueError(f"identité parallélogramme violée : {label} = {err:.3f}mm")
    mesh1 = abs(np.hypot(*(geom["A_L"] - geom["S"])) * 1000 - 21.0)
    mesh2 = abs(np.hypot(*(geom["A_R"] - geom["A_L"])) * 1000 - 21.0)
    checks.append(("engrènement servo↔gear_L (entraxe 21mm)", mesh1))
    checks.append(("engrènement gear_L↔gear_R (entraxe 21mm)", mesh2))

    geom["_checks"] = checks
    if verbose:
        for label, err in checks:
            print(f"  [check] {label}: {err:.4f} mm")
    return geom


# ──────────────────────────────────────────────────────────────────────────────
# Contre-vérification : solveur 4-bar GÉNÉRAL (intersection de cercles).
# Indépendant de l'hypothèse parallélogramme — doit retomber sur la cascade V3.
# ──────────────────────────────────────────────────────────────────────────────

def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def general_fourbar(geom: dict, side: str, theta_gear: float, last_P2=None):
    """Fermeture 4-bar générale : θ levier-engrenage → (θ_bot, θ_jaw)."""
    A, G = geom[f"A_{side}"], geom[f"G_{side}"]
    P1_0, P2_0 = geom[f"P1_{side}"], geom[f"P2_{side}"]
    r_bot = float(np.hypot(*(P2_0 - G)))
    r_coupler = float(np.hypot(*(P2_0 - P1_0)))
    P1 = A + _rot(theta_gear) @ (P1_0 - A)
    d = float(np.hypot(*(P1 - G)))
    if d > r_bot + r_coupler or d < abs(r_bot - r_coupler) or d < 1e-12:
        return None
    a = (r_bot ** 2 - r_coupler ** 2 + d * d) / (2 * d)
    h = float(np.sqrt(max(r_bot ** 2 - a * a, 0.0)))
    mid = G + a * (P1 - G) / d
    perp = np.array([-(P1 - G)[1], (P1 - G)[0]]) / d
    sols = [mid + h * perp, mid - h * perp]
    ref = P2_0 if last_P2 is None else last_P2
    P2 = min(sols, key=lambda p: float(np.hypot(*(p - ref))))
    th_bot = float(np.arctan2(*(P2 - G)[::-1]) - np.arctan2(*(P2_0 - G)[::-1]))
    th_jaw = float(np.arctan2(*(P2 - P1)[::-1]) - np.arctan2(*(P2_0 - P1_0)[::-1])) - theta_gear
    wrap = lambda t: float(np.arctan2(np.sin(t), np.cos(t)))
    return wrap(th_bot), wrap(th_jaw), P2


# ──────────────────────────────────────────────────────────────────────────────
# Suite de validation — invariants mesurés DANS MuJoCo (V3) + FK numpy (V2)
# ──────────────────────────────────────────────────────────────────────────────

def _mujoco_metrics(model, data, geom, qpos_dict):
    """Mesures monde MuJoCo : résidu de fermeture P2 (mâchoire vs petit levier),
    DANS LE PLAN du mécanisme (XY local gb_root — les pièces s'empilent en Z par
    construction, l'offset Z de 6mm est l'épaisseur d'assemblage, pas une erreur),
    et angle de rotation de la mâchoire vs gb_root (doit rester 0)."""
    import mujoco
    for name, val in qpos_dict.items():
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        data.qpos[adr] = val
    mujoco.mj_forward(model, data)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gb_root")
    R_root = data.xmat[root].reshape(3, 3)
    p_root = data.xpos[root]
    out = {}
    for s in ("L", "R"):
        jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"gb_gripper_{s}")
        bot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"gb_lever_bot_{s}")
        # P2 vu de la mâchoire (origine mâchoire = P1) et vu du petit levier (origine = G)
        off_jaw = np.append(geom[f"P2_{s}"] - geom[f"P1_{s}"], 0.0)
        off_bot = np.append(geom[f"P2_{s}"] - geom[f"G_{s}"], 0.0)
        p_jaw = data.xpos[jaw] + data.xmat[jaw].reshape(3, 3) @ off_jaw
        p_bot = data.xpos[bot] + data.xmat[bot].reshape(3, 3) @ off_bot
        # retour dans le repère gb_root, comparaison XY seulement (plan mécanisme)
        d_local = R_root.T @ (p_jaw - p_bot)
        R_rel = R_root.T @ data.xmat[jaw].reshape(3, 3)
        out[f"res_{s}"] = float(np.hypot(d_local[0], d_local[1]))
        out[f"jawrot_{s}"] = abs(float(np.arctan2(R_rel[1, 0], R_rel[0, 0])))
    return out


def _v2_metrics(geom, theta_in):
    """FK numpy de l'ANCIEN arbre V2 (pivots inversés + signes variante B) :
    résidu de fermeture P2 + rotation mâchoire — pour le before/after."""
    out = {}
    for s, sgn_top in (("L", +1), ("R", -1)):
        A = geom[f"A_{s}"]
        B = geom[f"P1_{s}"]            # V2 : pivot châssis du "levier bas" = P1 (faux)
        D = geom[f"G_{s}"]             # V2 : charnière mâchoire = G (faux)
        th_top = V2_SIGNS[f"grip_top_{s}"] * theta_in
        th_bot = V2_SIGNS[f"grip_bot_{s}"] * theta_in
        th_jaw = V2_SIGNS[f"grip_jaw_{s}"] * theta_in
        # arbre V2 : bot pivote en B (monde) ; jaw enfant de bot, charnière en D
        R_bot = _rot(th_bot)
        D_w = B + R_bot @ (D - B)
        R_jaw = _rot(th_bot + th_jaw)
        # jonctions réelles portées par les meshes : P2 appartient au mesh bot
        # (dessiné depuis B) et au mesh jaw (dessiné depuis D)
        P2_bot = B + R_bot @ (geom[f"P2_{s}"] - B)
        P2_jaw = D_w + R_jaw @ (geom[f"P2_{s}"] - D)
        # P1 appartient au mesh top (pivote en A) et au mesh jaw
        R_top = _rot(th_top)
        P1_top = A + R_top @ (geom[f"P1_{s}"] - A)
        P1_jaw = D_w + R_jaw @ (geom[f"P1_{s}"] - D)
        out[f"res_{s}"] = max(float(np.hypot(*(P2_bot - P2_jaw))),
                              float(np.hypot(*(P1_top - P1_jaw))))
        out[f"jawrot_{s}"] = abs(th_bot + th_jaw)
    return out


def run_validation():
    import mujoco
    here = os.path.dirname(os.path.abspath(__file__))
    print("── Extraction + auto-validation géométrie (STL → 9 points) ──")
    geom = extract_geometry(verbose=True)
    worst = max(e for _, e in geom["_checks"])
    print(f"  pire écart : {worst:.4f} mm")

    print("\n── Contre-vérification : solveur 4-bar général vs cascade parallélogramme ──")
    th_c0 = theta_close(geom)
    max_dev, last = 0.0, {"L": None, "R": None}
    for th in np.linspace(-th_c0, 0.0, 61):
        q = solve(th)
        for s in ("L", "R"):
            sol = general_fourbar(geom, s, q[f"grip_top_{s}"], last[s])
            if sol is None:
                print(f"  ✗ FAIL : boucle ne ferme pas à θ={th:.3f} ({s})")
                return False
            th_bot, th_jaw, last[s] = sol
            max_dev = max(max_dev, abs(th_bot - q[f"grip_bot_{s}"]),
                          abs(th_jaw - q[f"grip_jaw_{s}"]))
    # Seuil exprimé en erreur PHYSIQUE à la jonction : dev(rad) × bras 43.2mm.
    # Le bruit d'extraction (~0.0006mm) s'amplifie près des grands angles
    # (sensibilité du 4-bar quasi-parallélogramme) — tolérance : 0.05mm.
    dev_mm = max_dev * 43.2
    print(f"  écart max cascade↔solveur général : {max_dev:.2e} rad "
          f"(= {dev_mm*1000:.0f} µm à la jonction) "
          + ("✓ PASS (< 0.05 mm)" if dev_mm < 0.05 else "✗ FAIL"))
    max_dev_ok = dev_mm < 0.05

    model = mujoco.MjModel.from_xml_path(os.path.join(here, "roarm_m3_gripper_b.xml"))
    data = mujoco.MjData(model)

    th_c = theta_close(geom)
    print(f"\n  gap faces internes au repos (STL) : {jaw_face_gap_rest_mm():.2f} mm")
    print(f"  fermeture COMPLÈTE (faces qui se touchent) : θ = -{th_c:.4f} rad ({np.degrees(th_c):.1f}°)")

    print(f"\n── Sweep θ ∈ [-{th_c:.3f}, 0] (course utile complète) : V3 (MuJoCo) vs V2 (FK ancien arbre) ──")
    print("    θ      | V3 résidu L/R (mm) | V3 rot mâchoire (°) | V2 résidu (mm) | V2 rot (°)")
    v3_res, v3_rot, v2_res, v2_rot, gaps = 0.0, 0.0, 0.0, 0.0, []
    sweep = np.linspace(-th_c, 0.0, 49)
    for i_th, th in enumerate(sweep):
        m3 = _mujoco_metrics(model, data, geom, solve(float(th)))
        m2 = _v2_metrics(geom, float(th))
        bL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gb_gripper_L")
        bR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gb_gripper_R")
        gaps.append((float(th), float(np.linalg.norm(data.xpos[bL] - data.xpos[bR]))))
        v3_res = max(v3_res, m3["res_L"], m3["res_R"])
        v3_rot = max(v3_rot, m3["jawrot_L"], m3["jawrot_R"])
        v2_res = max(v2_res, m2["res_L"], m2["res_R"])
        v2_rot = max(v2_rot, m2["jawrot_L"], m2["jawrot_R"])
        if i_th % 6 == 0 or i_th == len(sweep) - 1:
            print(f"  {th:+.2f}    | {m3['res_L']*1000:7.4f} {m3['res_R']*1000:7.4f}    | "
                  f"{np.degrees(max(m3['jawrot_L'], m3['jawrot_R'])):7.4f}             | "
                  f"{max(m2['res_L'], m2['res_R'])*1000:6.2f}         | "
                  f"{np.degrees(max(m2['jawrot_L'], m2['jawrot_R'])):5.1f}")

    print(f"\n  MAX  V3 : résidu {v3_res*1000:.4f} mm, rotation mâchoire {np.degrees(v3_rot):.4f}°")
    print(f"  MAX  V2 : résidu {v2_res*1000:.2f} mm, rotation mâchoire {np.degrees(v2_rot):.1f}°")

    ok = True
    checks = [
        (v3_res < 1e-4, f"fermeture V3 < 0.1 mm dans le plan (= {v3_res*1000:.4f})"),
        (v3_rot < 1e-6, f"mâchoires V3 parallèles, rotation nulle (= {np.degrees(v3_rot):.2e}°)"),
        (max_dev_ok, "cascade = solveur 4-bar général à <0.05mm (indép. de l'hypothèse parallélogramme)"),
    ]
    # Repos export CAD = bras horizontaux = ouverture géométrique MAX (sommet de
    # la courbe en U) : la monotonie se valide sur la course utile [-θ_close, 0].
    used = sorted(gaps)
    diffs = np.diff([d for _, d in used])
    checks.append((bool(np.all(diffs >= -1e-12)),
                   f"ouverture monotone sur la course utile (θ ∈ [-{th_c:.3f}, 0])"))
    # Fermeture complète : la course parcourue par les origines mâchoires doit
    # égaler le gap au repos (faces qui se touchent à -θ_close).
    travel = (used[-1][1] - used[0][1]) * 1000
    gap0 = jaw_face_gap_rest_mm()
    checks.append((abs(travel - gap0) < 0.1,
                   f"fermeture COMPLÈTE atteinte (course {travel:.2f} mm = gap repos {gap0:.2f} mm ±0.1)"))
    checks.append((th_c <= THETA_RANGE[1] - 1e-9,
                   f"θ_close ({th_c:.3f}) dans les ranges joints (±{THETA_RANGE[1]})"))
    qall = [solve(t) for t in np.linspace(-th_c, th_c, 13)]
    checks.append((all(THETA_RANGE[0] - 1e-9 <= v <= THETA_RANGE[1] + 1e-9
                       for q in qall for v in q.values()), "qpos dans les ranges joints"))
    for passed, label in checks:
        print(f"  {'✓ PASS' if passed else '✗ FAIL'} : {label}")
        ok = ok and passed
    print("  ⚠ CALIBRATION RESTANTE (2 points physiques) : le mapping servo°→θ "
          "(échelle/offset/sens) n'est PAS dérivable des STL seuls — V2 utilisait "
          "(action-73)/113.3 calibré à l'œil ; engrenages 1:1 suggèrent π/180. "
          "→ 2 mesures pied à coulisse (gap à 2 angles servo) fixent le mapping exactement.")

    report = {
        "date": "2026-06-10",
        "geometry_checks_mm": [(lbl, float(e)) for lbl, e in geom["_checks"]],
        "max_residual_mm": {"V3_mujoco": v3_res * 1000, "V2_fk": v2_res * 1000},
        "max_jaw_rotation_deg": {"V3": float(np.degrees(v3_rot)), "V2": float(np.degrees(v2_rot))},
        "solver_crosscheck_rad": max_dev,
        "gap_curve": gaps,
        "all_pass": ok,
    }
    path = os.path.join(here, "gripper_b_v3_validation.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n  rapport → {path}")
    return ok


def run_viewer():
    """Fenêtre MuJoCo interactive : sweep continu ouvert↔fermé via la cascade
    exacte (les 7 qpos bougent ensemble — bouger un slider seul décrocherait
    les pièces, c'est normal : la fermeture est logicielle, pas equality)."""
    import time
    import mujoco
    import mujoco.viewer
    here = os.path.dirname(os.path.abspath(__file__))
    m = mujoco.MjModel.from_xml_path(os.path.join(here, "roarm_m3_gripper_b.xml"))
    d = mujoco.MjData(m)
    adr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
           for n in V3_SIGNS}
    mujoco.mj_forward(m, d)
    bL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gb_gripper_L")
    bR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gb_gripper_R")
    th_c = theta_close()  # fermeture complète, calculée depuis les STL (~1.545 rad)
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = (d.xpos[bL] + d.xpos[bR]) / 2
        v.cam.distance = 0.35
        t0 = time.time()
        while v.is_running():
            # profil triangulaire 6s/cycle sur la course utile complète [-θ_close, 0]
            phase = ((time.time() - t0) % 6.0) / 6.0
            tri = 2 * phase if phase < 0.5 else 2 * (1 - phase)
            for name, val in solve(-th_c * tri).items():
                d.qpos[adr[name]] = val
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    import sys as _sys
    if "--view" in _sys.argv:
        run_viewer()
    else:
        raise SystemExit(0 if run_validation() else 1)
