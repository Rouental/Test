"""Build examples/portrait.json: a painted portrait.

It follows a classical painter's process, using pdnart ops only:

1. Hidden *study* layers model each mass as a lit 3D form (`form` op): a face with bumps for the
   brow ridge, cheekbones, nose, lips and chin and hollows for the eye sockets, plus the hair mass,
   neck and dress. A multiply layer adds local colour (lips, flushed cheeks).
2. `painterly` layers repaint those studies with real oil strokes, big brushes first, with small ones
   only where the form needs detail. The strokes wrap around the forms.
3. Crisp features (eyes, brows, nostrils, the line of the mouth) and hair strands are painted on top
   with small brushes. A pencil layer on multiply adds the underdrawing, and a warm glaze sits on top.

Run:  python examples/portrait.py && pdnart render examples/portrait.json -o examples/portrait.png
"""

import json
import math
import random
from pathlib import Path

W, H = 700, 880
CX = 350
LIGHT = [-0.72, -0.5, 0.5]  # key light from the upper left, slightly in front
rng = random.Random(11)


def jc(hex_color, amount=10):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    d = lambda v: max(0, min(255, v + rng.randint(-amount, amount)))  # noqa: E731
    return f"#{d(r):02x}{d(g):02x}{d(b):02x}"


def brush(kind, pts, color, size, **kw):
    return {"op": "brush", "brush": kind, "points": pts, "color": color, "size": size, **kw}


def bump(x, y, rx, ry, h, **kw):
    return {"x": x, "y": y, "rx": rx, "ry": ry, "h": h, **kw}


def pair(x, y, rx, ry, h, **kw):
    """A bump and its mirror image across the face's centre line."""
    return [bump(x, y, rx, ry, h, **kw), bump(2 * CX - x, y, rx, ry, h, **kw)]


SKIN_RAMP = [[0, "#4a231d"], [0.28, "#8a4a3a"], [0.5, "#bb7560"], [0.72, "#dea084"], [0.9, "#f3c6a8"],
             [1, "#fde2cc"]]
HAIR_RAMP = [[0, "#120a07"], [0.45, "#321d13"], [0.75, "#5e3822"], [0.92, "#94603a"], [1, "#c9955f"]]
DRESS_RAMP = [[0, "#0a1f1e"], [0.45, "#174241"], [0.75, "#2a6763"], [1, "#6fa79c"]]

FACE = [[350, 198], [405, 205], [450, 233], [470, 283], [474, 343], [467, 403], [454, 450], [430, 494],
        [400, 530], [373, 552], [350, 558], [327, 552], [300, 530], [270, 494], [246, 450], [233, 403],
        [226, 343], [230, 283], [250, 233], [295, 205]]
NECK = [[296, 480], [404, 480], [412, 590], [440, 650], [424, 770], [276, 770], [260, 650], [288, 590]]
DRESS = [[-10, 900], [30, 770], [150, 698], [276, 654], [350, 748], [424, 654], [550, 698], [670, 770],
         [710, 900]]
HAIR_BACK = [[350, 126], [424, 134], [480, 166], [512, 226], [522, 320], [514, 430], [526, 548], [540, 650],
             [510, 668], [350, 610], [190, 668], [160, 650], [174, 548], [186, 430],
             [178, 320], [188, 226], [220, 166], [276, 134]]
PART = (318, 160)
HAIR_LEFT = [[PART[0], PART[1]], [262, 176], [218, 222], [198, 300], [200, 400], [194, 500], [182, 600],
             [196, 690], [226, 650], [232, 560], [240, 450], [236, 340], [246, 262], [290, 206]]
HAIR_RIGHT = [[PART[0] + 6, PART[1]], [400, 164], [466, 196], [506, 262], [514, 360], [508, 470], [522, 580],
              [512, 696], [484, 656], [474, 570], [470, 470], [470, 360], [452, 272], [414, 222], [360, 210],
              [336, 196]]
EYES = [(297, 352), (403, 352)]

layers = []

# ---------------------------------------------------------------- 1. studies (hidden)
study_bg = [{"op": "gradient", "kind": "radial", "cx": 170, "cy": 250, "r": 820,
             "stops": [[0, "#9a9a82"], [0.45, "#5f6a5f"], [1, "#1c2224"]]}]
for _ in range(40):
    x, y = rng.uniform(0, W), rng.uniform(0, H)
    study_bg.append(brush("soft", [[x, y], [x + rng.uniform(-80, 80), y + rng.uniform(-80, 80)]],
                          rng.choice(["#8a8c70", "#4e5a55", "#6b6a52", "#2c3534"]), rng.uniform(90, 200),
                          opacity=0.35))
layers.append({"name": "Study: background", "visible": False, "ops": study_bg})

layers.append({"name": "Study: hair mass", "visible": False, "ops": [
    {"op": "form", "region": HAIR_BACK, "inflate": 90, "light": LIGHT, "ramp": HAIR_RAMP, "wrap": 0.3,
     "shine": 0.15, "shine_size": 8, "occlusion": 0.4,
     "bumps": [bump(350, 190, 130, 70, 30), bump(350, 520, 110, 120, -40)]},
]})

layers.append({"name": "Study: neck and dress", "visible": False, "ops": [
    {"op": "form", "region": NECK, "inflate": 45, "light": LIGHT, "ramp": SKIN_RAMP, "wrap": 0.4,
     "bumps": [bump(350, 700, 60, 30, -6), bump(300, 668, 40, 8, 5, angle=12),
               bump(400, 668, 40, 8, 5, angle=-12)]},  # pit of the throat, collarbones
    # the head casts a shadow down the neck
    brush("soft", [[296, 530], [350, 566], [410, 540], [414, 600]], "#5a2a20", 80, opacity=0.85, lock_alpha=True),
    {"op": "form", "region": DRESS, "inflate": 140, "light": LIGHT, "ramp": DRESS_RAMP, "wrap": 0.3,
     "shine": 0.15, "shine_size": 20,
     "bumps": [bump(150, 760, 90, 70, 40), bump(550, 760, 90, 70, 30),
               *[bump(x, 840, 22, 80, rng.choice([-7, 7]), angle=(x - CX) * 0.06) for x in range(120, 620, 70)]]},
]})

face_bumps = [
    bump(350, 255, 95, 55, 10),                                  # forehead
    *pair(296, 316, 36, 12, 9, angle=-8),                       # brow ridge
    *pair(297, 354, 30, 21, -16),                               # eye sockets
    *pair(297, 354, 21, 13, 9, profile="sphere"),               # eyeballs under the lids
    bump(350, 385, 11, 55, 15),                                 # nose bridge
    bump(350, 438, 18, 16, 20),                                 # nose tip
    *pair(333, 449, 11, 9, 9),                                  # nostril wings
    bump(350, 462, 18, 7, -6),                                  # under the nose
    *pair(272, 400, 40, 26, 13, angle=-20),                     # cheekbones
    *pair(284, 478, 30, 26, -6),                                # hollow of the cheek
    bump(350, 505, 48, 27, 9),                                  # the barrel of the mouth
    bump(350, 494, 30, 7, 3),                                   # upper lip
    bump(350, 514, 24, 8, 6),                                   # lower lip
    bump(350, 503, 34, 2.5, -4),                                # mouth line
    *pair(316, 503, 7, 6, -2.5),                                # mouth corners
    bump(350, 530, 22, 6, -5),                                  # groove under the lip
    bump(350, 545, 30, 17, 10),                                 # chin
    *pair(247, 300, 18, 34, -6),                                # temples
]
layers.append({"name": "Study: face", "visible": False, "ops": [
    {"op": "form", "region": FACE, "inflate": 110, "light": LIGHT, "ramp": SKIN_RAMP, "wrap": 0.45,
     "occlusion": 0.8, "bounce": ["#9b7d70", 0.08], "shine": 0.18, "shine_size": 30, "bumps": face_bumps},
    # the hair throws a soft shadow across the top of the forehead and the far side
    brush("soft", [[250, 218], [330, 205], [420, 222], [462, 280]], "#5a2c22", 44, opacity=0.6, lock_alpha=True),
]})

# Local colour on multiply, the way a painter varies skin: warm cheeks, ears and nose, cooler jaw, red lips.
layers.append({"name": "Study: skin colour", "visible": False, "blend": "multiply", "ops": [
    brush("soft", [[258, 430], [290, 445]], "#f0c0b2", 90, opacity=0.5),
    brush("soft", [[412, 430], [444, 445]], "#f0c0b2", 90, opacity=0.5),
    brush("soft", [[350, 440]], "#f3b8aa", 30, opacity=0.7),
    brush("soft", [[300, 540], [350, 556], [400, 540]], "#dcd0d6", 60, opacity=0.7),
    *[brush("soft", [[x - 22, y - 6], [x + 22, y - 6]], "#e3c6cc", 26, opacity=0.8) for x, y in EYES],
    {"op": "polygon", "smooth": True, "fill": "#cf6f68", "feather": 1.5,
     "points": [[317, 502], [332, 492], [345, 489], [350, 492], [355, 489], [368, 492], [383, 502], [350, 505]]},
    {"op": "polygon", "smooth": True, "fill": "#dc857a", "feather": 1.5,
     "points": [[319, 503], [350, 506], [381, 503], [369, 518], [350, 523], [331, 518]]},
]})

# ---------------------------------------------------------------- 2. painting
layers.append({"name": "Background", "ops": [
    {"op": "painterly", "source": "Study: background", "brush": "oil", "sizes": [90, 45], "threshold": 0.04,
     "length": [2, 4], "curvature": 0.5, "dryness": 0.5, "color_jitter": 0.015, "color_variation": 0.04, "seed": 1},
    {"op": "copy", "source": "Study: background", "opacity": 0.35}]})
layers.append({"name": "Hair mass", "ops": [
    {"op": "painterly", "source": "Study: hair mass", "brush": "oil", "sizes": [14, 7], "threshold": 0.04,
     "length": [6, 24], "curvature": 0.9, "color_jitter": 0.02, "taper": [0.2, 0.5], "seed": 2,
     "clip": {"points": HAIR_BACK, "smooth": True, "feather": 2}},
    {"op": "copy", "source": "Study: hair mass", "opacity": 0.3, "lock_alpha": True}]})
layers.append({"name": "Neck and dress", "ops": [
    *[{"op": "painterly", "source": "Study: neck and dress", "brush": "oil", "sizes": [28, 14, 9],
       "threshold": 0.05, "length": [2, 6], "curvature": 0.6, "color_jitter": 0.01, "color_variation": 0.04,
       "seed": 3 + k, "region": {"points": shape}, "clip": {"points": shape, "feather": 1}}
      for k, shape in enumerate((NECK, DRESS))],
    {"op": "copy", "source": "Study: neck and dress", "opacity": 0.4}]})
layers.append({"name": "Face", "ops": [
    {"op": "painterly", "source": ["Study: face", "Study: skin colour"], "brush": "oil", "sizes": [18, 9, 5],
     "threshold": 0.03, "length": [2, 6], "curvature": 0.6, "color_jitter": 0.006, "color_variation": 0.03,
     "dryness": 0.1, "seed": 4,
     "region": {"points": FACE, "smooth": True}, "clip": {"points": FACE, "smooth": True, "feather": 1}},
    # glaze the smooth study back over the strokes: the modelling stays soft and the brushwork shows through
    {"op": "copy", "source": ["Study: face", "Study: skin colour"], "opacity": 0.5, "lock_alpha": True},
]})

# Crisp features on top of the paint.
feat = []
for i, (ex, ey) in enumerate(EYES):
    s = -1 if i == 0 else 1  # direction of the outer corner
    inner, outer = [ex - s * 22, ey + 3], [ex + s * 24, ey]
    upper = [inner, [ex - s * 9, ey - 9], [ex + s * 8, ey - 10], outer]
    lower = [outer, [ex + s * 9, ey + 6], [ex - s * 9, ey + 7], inner]
    eye = {"points": upper + lower[1:-1], "smooth": True}
    shadow_side = i == 1
    feat += [
        {"op": "polygon", **eye, "fill": "#d8c7bd" if shadow_side else "#e9ddd4"},
        {"op": "gradient", "kind": "radial", "cx": ex - 4, "cy": ey - 3, "r": 28, "clip": eye,
         "stops": [[0, "#00000000"], [0.6, "#7a4c4033"], [1, "#6a3c32aa"]]},  # the eyeball turns away
        {"op": "ellipse", "cx": ex + 1, "cy": ey - 1.5, "rx": 9.5, "ry": 9.5, "fill": "#4a6450", "clip": eye},
        {"op": "gradient", "kind": "radial", "cx": ex + 2, "cy": ey + 1, "r": 9.5, "lock_alpha": True,
         "clip": {"points": [[ex - 11, ey - 12], [ex + 12, ey - 12], [ex + 12, ey + 9], [ex - 11, ey + 9]]},
         "stops": [[0, "#9aa872"], [0.35, "#6d8a5c"], [0.8, "#324836"], [1, "#1b261d"]]},
        {"op": "ellipse", "cx": ex + 1, "cy": ey - 1.5, "rx": 3.8, "ry": 3.8, "fill": "#0e0a09"},
        brush("soft", [[ex - 20, ey - 8], [ex + 20, ey - 8]], "#2a1a15", 12, opacity=0.65, clip=eye),  # lid shadow
        {"op": "ellipse", "cx": ex - 3, "cy": ey - 4.5, "rx": 1.8, "ry": 1.8, "fill": "#fdf8f2"},  # catchlight
        brush("ink", upper, "#21140e", 3.2, taper=[0.1, 0.2]),
        brush("pencil", lower, "#7a4c3e", 1.6, opacity=0.5),
        brush("soft", [[ex - s * 16, ey - 15], [ex + s * 4, ey - 18], [ex + s * 22, ey - 12]], "#7a4638", 5,
              opacity=0.45),  # lid crease
        brush("soft", [[inner[0] + s * 2, inner[1]]], "#c98a86", 5, opacity=0.6),  # tear duct
    ]
    for k in range(8):  # lashes
        f = 0.35 + k * 0.085
        px = inner[0] + (outer[0] - inner[0]) * f
        py = ey + 3 - 13 * math.sin(math.pi * min(f, 0.95))
        feat.append(brush("ink", [[px, py], [px + s * (2 + 5 * f), py - 4 - 3 * f]], "#1e120c", 1.3,
                          taper=[0.0, 0.8]))
    for k in range(34):  # brow hairs, rising at the inner end and flowing outward
        f = k / 33
        bx = ex - s * 30 + s * 64 * f + rng.uniform(-1, 1)
        by = ey - 34 - 8 * math.sin(math.pi * min(1, f * 1.2)) + 3 * f + rng.uniform(-1.5, 1.5)
        rise = 5 if f < 0.2 else 2
        feat.append(brush("ink", [[bx, by + 3], [bx + s * 8, by - rise]], jc("#2e1b12", 8), 2.8 - 1.2 * f,
                          opacity=0.75, taper=[0.1, 0.7]))
feat += [
    brush("soft", [[337, 455], [341, 453]], "#4a221d", 8, opacity=0.8),  # nostrils
    brush("soft", [[359, 453], [363, 455]], "#4a221d", 8, opacity=0.85),
    brush("pencil", [[324, 432], [321, 447], [333, 457]], "#7a4638", 2, opacity=0.4),
    brush("pencil", [[376, 432], [380, 447], [367, 457]], "#6a3a2e", 2, opacity=0.5),
    brush("ink", [[318, 502], [334, 504], [350, 505], [366, 504], [382, 502]], "#5a2622", 2.2, taper=[0.35, 0.35],
          opacity=0.85),  # the line between the lips
    brush("soft", [[336, 511], [350, 513]], "#f2b8aa", 8, opacity=0.55),  # wet highlight on the lower lip
    brush("soft", [[345, 486], [355, 486]], "#f6c9b4", 5, opacity=0.4),  # cupid's bow catches light
]
layers.append({"name": "Features", "ops": feat})

# Hair in front of the face: two painted curtains, then individual strands and highlights.
hair = [
    {"op": "form", "region": HAIR_LEFT, "inflate": 14, "light": LIGHT, "ramp": HAIR_RAMP, "wrap": 0.4,
     "shine": 0.2, "shine_size": 6},
    {"op": "form", "region": HAIR_RIGHT, "inflate": 16, "light": LIGHT, "ramp": HAIR_RAMP, "wrap": 0.4,
     "shine": 0.15, "shine_size": 6},
]
layers.append({"name": "Study: hair front", "visible": False, "ops": hair})
strands = [{"op": "painterly", "source": "Study: hair front", "brush": "oil", "sizes": [10, 5], "threshold": 0.04,
            "length": [6, 20], "curvature": 0.95, "bristles": 10, "color_jitter": 0.02, "seed": 5,
            "clip": {"points": HAIR_LEFT, "smooth": True, "feather": 1}},
           {"op": "painterly", "source": "Study: hair front", "brush": "oil", "sizes": [10, 5], "threshold": 0.04,
            "length": [6, 20], "curvature": 0.95, "bristles": 10, "color_jitter": 0.02, "seed": 6,
            "clip": {"points": HAIR_RIGHT, "smooth": True, "feather": 1}},
           {"op": "copy", "source": "Study: hair front", "opacity": 0.3, "lock_alpha": True}]


def strand(points, color, size, opacity=0.8):
    strands.append(brush("ink", points, color, size, opacity=opacity, taper=[0.15, 0.5]))


for side, curtain in ((-1, HAIR_LEFT), (1, HAIR_RIGHT)):
    for _ in range(90 if side < 0 else 120):
        f = rng.random()
        x0 = PART[0] + side * rng.uniform(4, 40)
        pts = [[x0, PART[1] + rng.uniform(-4, 6), 0.4]]
        for k, yy in enumerate((230, 330, 450, 580)):
            xs = [p[0] for p in curtain if abs(p[1] - yy) < 70] or [CX + side * 150]
            lo, hi = min(xs), max(xs)
            pts.append([lo + (hi - lo) * f + rng.uniform(-4, 4), yy + rng.uniform(-10, 10) + 40 * f * k / 3,
                        1.0 if k < 3 else 0.5])
        pts[-1][1] += rng.uniform(0, 70)
        lit = side < 0 and f > 0.4
        col = rng.choice(["#8a5a35", "#a8733f", "#6e4429"] if lit else ["#3a2419", "#2a1810", "#4f3020"])
        strand(pts, jc(col, 6), rng.uniform(1.0, 2.2), opacity=rng.uniform(0.5, 0.9))
for _ in range(50):  # the sheen on the crown, where the hair turns toward the light
    a = rng.uniform(0.25, 1.1)
    r = rng.uniform(150, 175)
    pts = [[PART[0] - 4, PART[1] + 4, 0.3]] + [
        [CX + r * math.cos(math.pi / 2 + a + t * 0.45), 330 - r * 1.05 * math.sin(math.pi / 2 + a + t * 0.45), 0.9]
        for t in (0.4, 0.9, 1.3)]
    strand(pts, jc("#c99a62", 12), rng.uniform(0.8, 1.6), opacity=0.6)
# Loose strands breaking the outer silhouette, so the hair doesn't read as a solid helmet.
outline = HAIR_BACK[1:9] + HAIR_BACK[-8:]
for _ in range(90):
    i = rng.randrange(len(outline) - 1)
    (xa, ya), (xb, yb) = outline[i], outline[i + 1]
    if abs(ya - yb) < 1 and ya > 600:
        continue
    f = rng.random()
    x, y = xa + (xb - xa) * f, ya + (yb - ya) * f
    out = 1 if x > CX else -1
    L = rng.uniform(40, 120)
    pts = [[x - out * 6, y - L * 0.5, 0.3], [x + out * rng.uniform(0, 7), y, 1.0],
           [x + out * rng.uniform(2, 12), y + L * 0.5, 0.2]]
    lit = x < CX and y < 400
    strand(pts, jc("#8a5a35" if lit else "#3a2419", 8), rng.uniform(0.7, 1.3), opacity=rng.uniform(0.35, 0.7))
layers.append({"name": "Hair front", "ops": strands})

# Pencil underdrawing on multiply, and a warm glaze of light.
layers.append({"name": "Pencil", "blend": "multiply", "opacity": 0.7, "ops": [
    brush("pencil", [[404, 540], [410, 600], [428, 650]], "#4a2f25", 2, opacity=0.4),
    {"op": "hatch", "region": [[445, 300], [476, 330], [468, 440], [440, 490], [436, 400]], "angle": 70, "gap": 5,
     "color": "#5a3a2e", "opacity": 0.2},
]})
layers.append({"name": "Warm light", "blend": "overlay", "opacity": 0.3, "ops": [
    {"op": "gradient", "kind": "radial", "cx": 170, "cy": 170, "r": 700,
     "stops": [[0, "#ffd49a"], [1, "#00000000"]]}]})

scene = {"width": W, "height": H, "background": "#2a3030", "layers": layers}
out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(scene))
print(f"wrote {out} ({sum(len(l['ops']) for l in layers)} ops)")
