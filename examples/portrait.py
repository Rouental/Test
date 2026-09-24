"""Build examples/portrait.json: a painted portrait from brush strokes, smudging and pencil linework.

The layers follow a painter's workflow:
pencil sketch -> flat block-in -> modelled light and shadow -> blending -> features -> hair -> linework.

Run:  python examples/portrait.py && pdnart render examples/portrait.json -o examples/portrait.png
"""

import json
import math
import random
from pathlib import Path

W, H = 700, 880
CX = 350
rng = random.Random(11)


def jc(hex_color, amount=10):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    d = lambda v: max(0, min(255, v + rng.randint(-amount, amount)))  # noqa: E731
    return f"#{d(r):02x}{d(g):02x}{d(b):02x}"


def mirror(pts):
    return [[2 * CX - x, *rest] for x, *rest in pts]


def brush(kind, pts, color, size, **kw):
    return {"op": "brush", "brush": kind, "points": pts, "color": color, "size": size, **kw}


FACE = [[350, 196], [410, 204], [458, 240], [474, 300], [472, 370], [458, 438], [430, 498], [392, 545],
        [350, 560], [308, 545], [270, 498], [242, 438], [228, 370], [226, 300], [242, 240], [290, 204]]
SKIN, SKIN_SHADOW, SKIN_DEEP, SKIN_LIGHT = "#dea283", "#a5654f", "#74402f", "#f8d8bf"
EYES = [(296, 352), (404, 352)]

layers = []

# -- Background: warm-to-teal oil strokes over a dark gradient ------------------------------------
bg = [{"op": "gradient", "kind": "radial", "cx": 200, "cy": 260, "r": 760,
       "stops": [[0, "#7c8a7a"], [0.55, "#475551"], [1, "#222a2b"]]}]
for _ in range(420):
    x, y = rng.uniform(-80, W), rng.uniform(-80, H)
    a = rng.uniform(-0.4, 0.4) + math.pi / 3
    L = rng.uniform(90, 200)
    near_light = math.hypot(x - 200, y - 260) < rng.uniform(250, 400)
    col = rng.choice(["#8a977f", "#76856f", "#978f70", "#66756a"] if near_light else ["#3f4c49", "#4b5853", "#39433f"])
    bend = rng.uniform(-20, 20)
    bg.append(brush(rng.choice(["dry", "oil"]), [[x, y, 0.4], [x + L / 2 * math.cos(a) + bend, y + L / 2 * math.sin(a), 1],
                                                [x + L * math.cos(a), y + L * math.sin(a), 0.5]],
                    jc(col, 5), rng.uniform(22, 40), opacity=0.4, color_variation=0.05, taper=[0.2, 0.3]))
bg.append({"op": "blur", "radius": 1.2})
layers.append({"name": "Background", "ops": bg})

# -- Hair, back mass (behind the face) ------------------------------------------------------------
HAIR_BACK = [[350, 142], [436, 152], [496, 210], [516, 320], [508, 440], [520, 560], [530, 640], [470, 630],
             [350, 600], [230, 630], [170, 640], [180, 560], [192, 440], [184, 320], [204, 210], [264, 152]]
hair_back = [{"op": "polygon", "points": HAIR_BACK, "smooth": True, "fill": "#2e1c14", "feather": 1.5}]
for _ in range(120):  # strands in the shadowed hair behind the neck
    x = rng.uniform(200, 500)
    hair_back.append(brush("oil", [[x, rng.uniform(380, 460)], [x + rng.uniform(-10, 10), rng.uniform(560, 640)]],
                           jc(rng.choice(["#3a2419", "#1e120c", "#4a2d1e"]), 5), rng.uniform(5, 10),
                           opacity=0.7, lock_alpha=True, bristles=8))
for side in (-1, 1):
    for _ in range(110):
        f = rng.random()
        x0 = CX + side * (30 + 110 * f)
        y0 = 150 + 60 * f + rng.uniform(-8, 8)
        x1 = CX + side * rng.uniform(140, 190)
        y1 = rng.uniform(460, 660)
        mid = [CX + side * rng.uniform(150, 180), (y0 + y1) / 2]
        hair_back.append(brush("oil", [[x0, y0, 0.5], mid, [x1, y1, 0.3]],
                               jc(rng.choice(["#3a2419", "#241610", "#4f3020", "#5e3a24"]), 6),
                               rng.uniform(6, 14), opacity=0.85, bristles=10, taper=[0.1, 0.6]))
layers.append({"name": "Hair back", "ops": hair_back})

# -- Neck and shoulders ---------------------------------------------------------------------------
body = [
    {"op": "polygon", "smooth": True, "fill": SKIN,
     "points": [[304, 500], [396, 500], [404, 580], [420, 632], [350, 670], [280, 632], [296, 580]]},
    # the jaw casts a shadow down the neck; the right side of the neck turns away from the light
    brush("soft", [[292, 560], [350, 588], [408, 560]], SKIN_DEEP, 64, opacity=0.8, lock_alpha=True),
    brush("soft", [[398, 560], [410, 640]], SKIN_SHADOW, 36, opacity=0.7, lock_alpha=True),
    brush("soft", [[318, 620], [350, 640], [382, 620]], SKIN_LIGHT, 30, opacity=0.35, lock_alpha=True),
    # shoulders / dress
    {"op": "polygon", "smooth": True, "fill": "#24524f",
     "points": [[20, 900], [50, 760], [170, 690], [285, 640], [350, 700], [415, 640], [530, 690], [650, 760],
                [680, 900]]},
]
for _ in range(140):
    x = rng.uniform(40, 660)
    y = rng.uniform(680, 800)
    drift = (x - CX) * 0.25
    body.append(brush("oil", [[x, y, 0.4], [x + drift * 0.5, y + 60, 1], [x + drift, y + rng.uniform(120, 200), 0.6]],
                      jc(rng.choice(["#2f6461", "#1f4644", "#3a7069", "#1a3a38"]), 5), rng.uniform(14, 26),
                      lock_alpha=True, opacity=0.5, dryness=0.7, taper=[0.2, 0.3]))
body.append(brush("soft", [[480, 700], [600, 780], [650, 880]], "#0f2322", 120, opacity=0.6, lock_alpha=True))
body.append(brush("soft", [[150, 720], [220, 700]], "#5a9189", 70, opacity=0.35, lock_alpha=True))
body.append({"op": "polygon", "smooth": True, "fill": SKIN,  # neckline
             "points": [[284, 648], [350, 740], [416, 648], [390, 640], [350, 668], [310, 640]]})
body.append(brush("soft", [[300, 660], [350, 725], [400, 660]], SKIN_SHADOW, 22, opacity=0.4, lock_alpha=True))
body.append(brush("ink", [[284, 650], [350, 740], [416, 650]], "#153230", 3, opacity=0.8))
layers.append({"name": "Neck and shoulders", "ops": body})

# -- Face: flat block-in, then modelling with soft brushes clipped to it, then smudge ------------
face = [{"op": "polygon", "points": FACE, "smooth": True, "fill": SKIN}]
shade = lambda pts, col, size, op: brush("soft", pts, col, size, opacity=op, lock_alpha=True)  # noqa: E731
face += [
    # light from the upper left: the right side of the face turns into shadow
    shade([[440, 240], [452, 330], [446, 430], [420, 500], [385, 545]], SKIN_SHADOW, 80, 0.9),
    shade([[462, 280], [466, 400], [448, 470]], SKIN_DEEP, 40, 0.6),
    shade([[418, 300], [428, 380], [415, 460], [390, 520]], SKIN_SHADOW, 60, 0.45),  # half-tone
    shade([[418, 400], [440, 412]], SKIN_SHADOW, 34, 0.3),  # core shadow under the cheekbone
    shade([[244, 250], [230, 360], [252, 460]], SKIN_SHADOW, 34, 0.45),  # far edge turning away on the left
    shade([[285, 528], [350, 566], [415, 528]], SKIN_SHADOW, 40, 0.6),  # under the jaw
    shade([[260, 215], [350, 200], [440, 215]], SKIN_SHADOW, 40, 0.45),  # shadow of the hair on the forehead
    # eye sockets and brow ridge
    *[shade([[x - 30, y - 8], [x, y - 16], [x + 30, y - 6]], SKIN_SHADOW, 30, 0.55 if x < CX else 0.75)
      for x, y in EYES],
    shade([[CX - 12, 350], [CX - 14, 372]], SKIN_SHADOW, 16, 0.45),  # side of the nose bridge
    # nose: shadow side plane, under the tip
    shade([[364, 360], [372, 410], [374, 440]], SKIN_SHADOW, 20, 0.7),
    shade([[330, 456], [350, 462], [372, 456]], SKIN_DEEP, 16, 0.75),
    # philtrum and the shadow under the lower lip
    shade([[350, 468], [350, 484]], SKIN_SHADOW, 10, 0.35),
    shade([[332, 528], [350, 532], [368, 528]], SKIN_SHADOW, 18, 0.6),
    # warmth: cheeks and nose tip
    shade([[262, 420], [288, 432]], "#e0806f", 54, 0.35),
    shade([[416, 424], [440, 430]], "#b8604f", 50, 0.35),
    shade([[350, 440]], "#e38677", 22, 0.3),
    # lights: forehead, nose bridge, cheekbone, upper lip, chin
    shade([[290, 250], [335, 238], [380, 252]], SKIN_LIGHT, 56, 0.7),
    shade([[346, 330], [344, 400], [347, 436]], SKIN_LIGHT, 12, 0.8),
    shade([[268, 390], [304, 400]], SKIN_LIGHT, 36, 0.6),
    shade([[336, 540], [350, 546]], SKIN_LIGHT, 18, 0.45),
]
# blend the soft shading, keeping paint inside the face
face.append({"op": "smudge", "size": 30, "strength": 0.3, "lock_alpha": True,
             "paths": [[[x, y0], [x + 4, y0 + 60]] for x in range(240, 470, 18) for y0 in range(210, 540, 50)]})
layers.append({"name": "Face", "ops": face})

# -- Features -----------------------------------------------------------------------------------
feat = []
for i, (ex, ey) in enumerate(EYES):
    s = -1 if i == 0 else 1  # direction of the outer corner
    inner, outer = [ex - s * 24, ey + 3], [ex + s * 26, ey - 1]
    upper = [inner, [ex - s * 10, ey - 10], [ex + s * 8, ey - 11], outer]
    lower = [outer, [ex + s * 10, ey + 7], [ex - s * 10, ey + 8], inner]
    eye_shape = {"points": upper + lower[1:-1], "smooth": True}
    shaded = i == 1  # the right eye sits in the shadow side
    feat.append({"op": "polygon", **eye_shape, "fill": "#d9c6bb" if shaded else "#ece0d7"})
    feat.append({"op": "gradient", "kind": "radial", "cx": ex, "cy": ey, "r": 30, "clip": eye_shape,
                 "stops": [[0, "#00000000"], [1, "#8a5f5288"]]})  # eyeball turns away at the corners
    # iris and pupil, clipped by the lids so the upper lid covers the top of the iris
    feat += [{"op": "ellipse", "cx": ex + 1, "cy": ey - 2, "rx": 10, "ry": 10, "fill": "#4f6b52", "clip": eye_shape},
             {"op": "gradient", "kind": "radial", "cx": ex + 1, "cy": ey - 2, "r": 10, "lock_alpha": True,
              "clip": {"points": [[ex - 12, ey - 14], [ex + 14, ey - 14], [ex + 14, ey + 10], [ex - 12, ey + 10]]},
              "stops": [[0, "#8aa074"], [0.5, "#5d7a58"], [0.9, "#2f4232"], [1, "#1f2b21"]]},
             {"op": "ellipse", "cx": ex + 1, "cy": ey - 2, "rx": 4.2, "ry": 4.2, "fill": "#120d0b"},
             brush("soft", [[ex - 20, ey - 8], [ex + 20, ey - 8]], "#3a2620", 14, opacity=0.6, clip=eye_shape),
             {"op": "ellipse", "cx": ex - 3, "cy": ey - 5, "rx": 2.2, "ry": 2.2, "fill": "#fbf6f0"}]
    feat.append(brush("ink", upper, "#24160f", 3.6, taper=[0.1, 0.25]))  # upper lid line
    for k in range(7):  # lashes along the outer part of the upper lid
        f = 0.4 + k * 0.09
        px = inner[0] + (outer[0] - inner[0]) * f
        py = ey + 3 - 14 * math.sin(math.pi * f)
        feat.append(brush("ink", [[px, py], [px + s * (2 + 5 * f), py - 5 - 3 * f]], "#24160f", 1.5,
                          taper=[0.0, 0.8]))
    feat.append(brush("pencil", lower, "#6b4637", 1.8, opacity=0.45))
    feat.append(brush("soft", [[ex - s * 18, ey - 16], [ex + s * 4, ey - 19], [ex + s * 24, ey - 13]],
                      "#8a5444", 6, opacity=0.4))  # lid crease
    # brow: short hair strokes, rising from the inner end then flowing outward
    for k in range(30):
        f = k / 29
        bx = ex - s * 32 + s * 66 * f + rng.uniform(-1, 1)
        by = ey - 36 - 9 * math.sin(math.pi * min(1, f * 1.25)) + 3 * f + rng.uniform(-1.5, 1.5)
        feat.append(brush("ink", [[bx, by + 4], [bx + s * 8, by - 1]], jc("#3b2419", 8), 2.6 - 1.2 * f,
                          opacity=0.7, taper=[0.1, 0.7]))
# nose: nostrils and the ala curve
feat += [brush("soft", [[336, 454], [340, 452]], "#5a2c26", 9, opacity=0.8),
         brush("soft", [[360, 452], [364, 454]], "#5a2c26", 9, opacity=0.8),
         brush("pencil", [[326, 430], [322, 446], [334, 456]], "#7a4a3c", 2.5, opacity=0.6),
         brush("pencil", [[374, 430], [379, 446], [366, 456]], "#6b3e33", 2.5, opacity=0.7)]
# lips
UPPER_LIP = [[318, 500], [333, 491], [345, 488], [350, 491], [355, 488], [367, 491], [382, 500], [350, 503]]
LOWER_LIP = [[320, 501], [350, 504], [380, 501], [368, 516], [350, 521], [332, 516]]
feat += [{"op": "polygon", "points": UPPER_LIP, "smooth": True, "fill": "#b3645e"},
         {"op": "polygon", "points": LOWER_LIP, "smooth": True, "fill": "#cf7d73"},
         brush("soft", [[362, 494], [378, 499]], "#8a4440", 10, opacity=0.5, lock_alpha=True),
         brush("soft", [[364, 506], [376, 510]], "#a55a52", 12, opacity=0.5, lock_alpha=True),
         brush("soft", [[334, 510], [350, 512]], "#efb2a4", 9, opacity=0.7),  # lower lip highlight
         brush("ink", [[316, 500], [333, 503], [350, 504], [367, 503], [384, 500]], "#5e2a26", 2.4,
               taper=[0.3, 0.3])]
layers.append({"name": "Features", "ops": feat})

# -- Hair, front: strands from the parting sweeping over the forehead and down past the face -----
hair = []
PART = (322, 172)
for side in (-1, 1):
    n = 170 if side == 1 else 110
    for k in range(n):
        f = k / n
        # strands start near the parting and fan out along the head outline
        sx = PART[0] + side * rng.uniform(0, 30)
        sy = PART[1] + rng.uniform(-6, 8)
        edge_x = CX + side * (125 + 55 * f + rng.uniform(-10, 10))
        top = [sx, sy, 0.5]
        mid = [CX + side * (80 + 60 * f), 220 + 70 * f + rng.uniform(-10, 10), 1.0]
        low = [edge_x, 330 + 260 * f + rng.uniform(-20, 40), 0.7]
        end = [edge_x + side * rng.uniform(0, 30), low[1] + rng.uniform(60, 140), 0.1]
        col = rng.choice(["#3a2419", "#5a3622", "#6e4429", "#2a1810", "#8a5a34"])
        hair.append(brush("oil", [top, mid, low, end], jc(col, 6), rng.uniform(5, 12), opacity=0.85,
                          taper=[0.1, 0.5], bristles=8, dryness=0.6))
# the crown: arcs from the parting over the top of the head
for _ in range(90):
    side = rng.choice([-1, 1])
    a0 = rng.uniform(0.1, 1.2)
    pts = [[PART[0] + side * rng.uniform(0, 12), PART[1] + rng.uniform(-4, 4), 0.4]]
    for k in range(1, 4):
        ang = math.pi / 2 - side * (a0 + k * 0.35)
        pts.append([CX + 150 * math.cos(ang) * (0.8 + 0.07 * k), 310 - 160 * math.sin(ang) * (0.95 - 0.02 * k), 0.8])
    hair.append(brush("oil", pts, jc(rng.choice(["#3a2419", "#5a3622", "#6e4429", "#2a1810"]), 6),
                      rng.uniform(6, 11), opacity=0.9, taper=[0.1, 0.4], bristles=8))
# highlights catching the light on the left crown
for _ in range(40):
    x0 = rng.uniform(250, 320)
    hair.append(brush("ink", [[x0 + 30, 180, 0.3], [x0 - 10, 215, 1], [x0 - 45, 280, 0.2]], jc("#c08a55", 10),
                      rng.uniform(1.2, 2.4), opacity=0.7))
layers.append({"name": "Hair front", "ops": hair})

# -- Pencil linework on a multiply layer: the drawing showing through the paint -------------------
contour = FACE[8:] + FACE[:9]
lines = [
    brush("pencil", [[x + rng.uniform(-1.5, 1.5), y + rng.uniform(-1.5, 1.5)] for x, y in contour[::2]],
          "#4a2f25", 2.4, opacity=0.5),
    brush("pencil", [[392, 545], [405, 610], [420, 668]], "#4a2f25", 2.2, opacity=0.45),  # neck
    brush("pencil", [[300, 520], [296, 610], [282, 662]], "#4a2f25", 2.2, opacity=0.35),
    {"op": "hatch", "region": [[440, 300], [476, 320], [470, 440], [440, 490], [430, 420]], "angle": 70, "gap": 5,
     "color": "#5a3a2e", "opacity": 0.25},  # shadow-side hatching
    {"op": "hatch", "region": [[300, 585], [400, 585], [410, 620], [290, 620]], "angle": 20, "gap": 5,
     "color": "#5a3a2e", "opacity": 0.25},
]
layers.append({"name": "Pencil lines", "blend": "multiply", "opacity": 0.9, "ops": lines})

# -- Light: warm glaze from the upper left --------------------------------------------------------
layers.append({"name": "Warm light", "blend": "overlay", "opacity": 0.35, "ops": [
    {"op": "gradient", "kind": "radial", "cx": 180, "cy": 180, "r": 650,
     "stops": [[0, "#ffd9a0"], [1, "#00000000"]]}]})

scene = {"width": W, "height": H, "background": "#2a3030", "layers": layers}
out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(scene))
print(f"wrote {out} ({sum(len(l['ops']) for l in layers)} ops)")
