"""Build examples/landscape.json: a painted mountain valley made almost entirely of brush strokes.

Run:  python examples/landscape.py && pdnart render examples/landscape.json -o examples/landscape.png
"""

import json
import math
import random
from pathlib import Path

W, H = 1000, 640
rng = random.Random(4)


def jitter_color(hex_color, amount=12):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    d = lambda v: max(0, min(255, v + rng.randint(-amount, amount)))  # noqa: E731
    return f"#{d(r):02x}{d(g):02x}{d(b):02x}"


def ridge(y0, amp, seed, levels=6, rough=0.55):
    """Jagged mountain ridge by midpoint displacement."""
    r = random.Random(seed)
    pts = [[-20.0, y0 - r.uniform(0, amp) * 0.4], [W + 20.0, y0 - r.uniform(0, amp) * 0.4]]
    disp = amp
    for _ in range(levels):
        nxt = [pts[0]]
        for (x0, ya), (x1, yb) in zip(pts, pts[1:]):
            nxt += [[(x0 + x1) / 2, (ya + yb) / 2 - r.uniform(-0.4 * disp, disp)], [x1, yb]]
        pts = nxt
        disp *= rough
    return [[round(x, 1), round(min(y, y0 + 20), 1)] for x, y in pts]


def ridge_y(pts, x):
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


layers = []

# 1. Sky: gradient underpainting, soft broad glazes, then cumulus clouds built from soft dabs.
sky = [{"op": "gradient", "kind": "linear", "x0": 0, "y0": 0, "x1": 0, "y1": 440,
        "stops": [[0, "#3f64a3"], [0.55, "#93acd3"], [0.85, "#e9c9ae"], [1, "#f6d3a6"]]}]
for _ in range(40):
    y = rng.uniform(0, 420)
    x = rng.uniform(-100, W)
    t = y / 440
    base = "#5577b0" if t < 0.35 else "#a3b8da" if t < 0.75 else "#f0cfb0"
    sky.append({"op": "brush", "brush": "glaze", "size": rng.uniform(40, 80), "opacity": 0.18,
                "color": jitter_color(base, 8),
                "points": [[x, y], [x + 200, y + rng.uniform(-6, 6)], [x + 420, y + rng.uniform(-10, 10)]]})
for cx, cy, s in [(230, 110, 1.0), (600, 70, 0.7), (860, 170, 0.55)]:
    # grey-violet cloud bodies, then warm lit tops, then a few crisp highlights
    for _ in range(28):
        x, y = cx + rng.gauss(0, 80 * s), cy + abs(rng.gauss(0, 14 * s))
        sky.append({"op": "brush", "brush": "soft", "size": rng.uniform(40, 70) * s, "color": "#b3b0c8",
                    "opacity": 0.65, "points": [[x - 20 * s, y + 10 * s], [x + 20 * s, y + 10 * s]]})
    for _ in range(26):
        x, y = cx + rng.gauss(0, 70 * s), cy - abs(rng.gauss(0, 16 * s))
        sky.append({"op": "brush", "brush": "soft", "size": rng.uniform(30, 55) * s, "color": "#fff1e2",
                    "opacity": 0.7, "points": [[x - 10 * s, y], [x + 10 * s, y - 2]]})
    for _ in range(14):
        x, y = cx + rng.gauss(0, 55 * s), cy - abs(rng.gauss(6, 12 * s))
        sky.append({"op": "brush", "brush": "soft", "size": rng.uniform(16, 26) * s, "color": "#fffaf3",
                    "opacity": 0.8, "points": [[x - 6 * s, y], [x + 6 * s, y]]})
layers.append({"name": "Sky", "ops": sky})

# 2. Far mountains: flat shape, then snow and shadow strokes clipped to it (lock_alpha), then haze.
far = ridge(420, 170, 3)
far_ops = [{"op": "polygon", "points": far + [[W + 20, 470], [-20, 470]], "fill": "#6b75a0", "feather": 0.6}]
for (x0, y0), (x1, y1) in zip(far, far[1:]):
    lit = y1 > y0  # slopes descending to the right face the late sun
    for _ in range(3):
        f = rng.random()
        x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        snow = y < 310
        col = ("#f4f1f2" if lit else "#a9b0d0") if snow else ("#9aa0c4" if lit else "#565f8c")
        length = rng.uniform(20, 60)
        far_ops.append({"op": "brush", "brush": "oil", "size": rng.uniform(6, 12), "lock_alpha": True,
                        "color": jitter_color(col, 8), "opacity": 0.85,
                        "points": [[x, y + 2, 0.9], [x + (0.5 if lit else -0.5) * length, y + length, 0.3]]})
far_ops.append({"op": "gradient", "x0": 0, "y0": 260, "x1": 0, "y1": 470, "lock_alpha": True,
                "stops": [[0, "#00000000"], [1, "#e6c9bfdd"]]})  # aerial haze
layers.append({"name": "Far mountains", "ops": far_ops})

# 3. Warm sunlight wash over the distance.
layers.append({"name": "Sun glow", "blend": "screen", "opacity": 0.7, "ops": [
    {"op": "gradient", "kind": "radial", "cx": 820, "cy": 300, "r": 420,
     "stops": [[0, "#ffcf99"], [0.4, "#ff9f6a55"], [1, "#00000000"]]}]})

# 4. Mid hills with oil strokes following the slope, plus a band of pine trees.
mid = ridge(450, 45, 11, levels=4, rough=0.5)
mid_ops = [{"op": "polygon", "points": mid + [[W + 20, 520], [-20, 520]], "fill": "#4d6b61", "smooth": True}]
for _ in range(220):
    x = rng.uniform(0, W)
    y = ridge_y(mid, x) + rng.uniform(5, 90)
    mid_ops.append({"op": "brush", "brush": "oil", "size": rng.uniform(8, 16), "lock_alpha": True, "opacity": 0.45,
                    "color": jitter_color(rng.choice(["#587866", "#3f5a55", "#6c8568", "#4c6a5c"]), 6),
                    "points": [[x, y], [x + rng.uniform(20, 50), y + rng.uniform(-6, 10)]]})
layers.append({"name": "Hills", "ops": mid_ops})

trees = []
for _ in range(70):
    x = rng.uniform(0, W)
    base = ridge_y(mid, x) + rng.uniform(15, 70)
    h = rng.uniform(30, 70) * (0.6 + (base - 380) / 250)
    dark = jitter_color("#23392f", 8)
    trees.append({"op": "brush", "brush": "ink", "size": 2.5, "color": "#1f2a24",
                  "points": [[x, base + 4], [x, base - h]]})
    for k in range(9):  # zig-zag branches get wider toward the base
        f = k / 8
        yy = base - h + h * f
        w = 3 + h * 0.28 * f
        trees.append({"op": "brush", "brush": "dry", "size": 5 + 4 * f, "color": dark, "opacity": 0.95,
                      "points": [[x - w, yy + 3], [x, yy - 2], [x + w, yy + 3]]})
    trees.append({"op": "brush", "brush": "oil", "size": 4, "color": "#5f7a57", "opacity": 0.7,
                  "points": [[x + 1, base - h * 0.8], [x + h * 0.2, base - h * 0.2]]})  # sunlit side
layers.append({"name": "Pines", "ops": trees})

# 5. Lake: reflected sky in horizontal strokes, then smudged smooth.
lake_top = 470
lake = [{"op": "rect", "x": 0, "y": lake_top, "w": W, "h": 60, "fill": "#8fa6c4"}]
lake.append({"op": "gradient", "x0": 0, "y0": lake_top, "x1": 0, "y1": lake_top + 60, "lock_alpha": True,
             "stops": [[0, "#4a6660"], [0.35, "#9fb3cf"], [1, "#e4c4aa"]]})  # hills, then sky, reflected
for _ in range(260):
    y = rng.uniform(lake_top + 2, lake_top + 58)
    x = rng.uniform(-40, W)
    lake.append({"op": "brush", "brush": "round", "size": rng.uniform(1, 2.5), "hardness": 0.5,
                 "color": jitter_color(rng.choice(["#d5e0ee", "#6f86a8", "#f3d5b5", "#3f5a55"]), 8),
                 "opacity": 0.6, "taper": [0.3, 0.3], "points": [[x, y], [x + rng.uniform(20, 90), y]]})
lake.append({"op": "smudge", "size": 30, "strength": 0.5, "paths": [[[0, y], [W, y]] for y in range(lake_top + 8, lake_top + 60, 14)]})
layers.append({"name": "Lake", "ops": lake})

# 6. Meadow: masses of grass strokes, darker at the bottom, with flowers.
meadow = [{"op": "polygon", "smooth": True, "fill": "#6f8b3e",
           "points": [[-20, 528], [300, 520], [600, 532], [W + 20, 522], [W + 20, H + 20], [-20, H + 20]]},
          {"op": "gradient", "x0": 0, "y0": 520, "x1": 0, "y1": H, "lock_alpha": True,
           "stops": [[0, "#a3b35a"], [1, "#2f4a24"]]}]
for _ in range(1400):
    x = rng.uniform(-10, W + 10)
    y = rng.uniform(528, H + 10)
    depth = (y - 520) / (H - 520)
    h = 6 + 30 * depth * rng.uniform(0.5, 1.2)
    lean = rng.uniform(-0.4, 0.4) * h
    col = rng.choice(["#8da64a", "#5f7f33", "#b8c46a", "#3e5f2a", "#d0c878"] if depth < 0.6 else
                     ["#3e5f2a", "#5f7f33", "#2b4520", "#7a9442"])
    meadow.append({"op": "brush", "brush": "ink", "size": 1.5 + 2.5 * depth, "color": jitter_color(col, 10),
                   "opacity": 0.9, "points": [[x, y, 1], [x + lean * 0.4, y - h * 0.6, 0.8], [x + lean, y - h, 0.2]]})
for _ in range(90):
    x, y = rng.uniform(0, W), rng.uniform(545, H - 10)
    meadow.append({"op": "brush", "brush": "round", "size": 2 + 5 * (y - 520) / 120, "hardness": 0.7,
                   "color": rng.choice(["#f6f0e0", "#f2c14e", "#e27d60", "#c9a0dc"]), "points": [[x, y]]})
layers.append({"name": "Meadow", "ops": meadow})

# 7. Loose pencil drawing over the top, on a multiply layer, as an artist's underdrawing showing through.
sketch = [{"op": "brush", "brush": "pencil", "color": "#3b3531", "opacity": 0.6,
           "points": [[x, y + rng.uniform(-2, 2)] for x, y in far]},
          {"op": "brush", "brush": "pencil", "color": "#3b3531", "opacity": 0.5,
           "points": [[x, ridge_y(mid, x) + 3] for x in range(-10, W + 20, 40)]},
          {"op": "hatch", "region": [[0, 530], [260, 522], [240, 640], [0, 640]], "angle": 60, "gap": 6,
           "color": "#2d3a22", "opacity": 0.35}]
layers.append({"name": "Pencil", "blend": "multiply", "opacity": 0.8, "ops": sketch})

scene = {"width": W, "height": H, "background": "#f3ede2", "layers": layers}
out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(scene))
print(f"wrote {out} ({sum(len(l['ops']) for l in layers)} ops)")
