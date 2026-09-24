"""Render examples/brushes.png: every brush preset with the same pressure-varying stroke."""

from pathlib import Path

from pdnart import Scene, composite
from pdnart.brushes import PRESETS

scene = Scene(900, 70 + 58 * len(PRESETS), "#f4efe4")
strokes = scene.add_layer("strokes")
labels = scene.add_layer("labels")
y = 45
for name in PRESETS:
    strokes.add({"op": "brush", "brush": name, "color": "#2b3f6b",
                 "points": [[130, y, 0.2], [300, y - 18, 0.9], [480, y + 14, 1.0], [660, y - 10, 0.6], [860, y + 4, 0.2]]})
    labels.add({"op": "text", "x": 16, "y": y - 8, "text": name, "size": 16, "color": "#333333"})
    y += 58
hatch = scene.add_layer("hatching", blend="multiply")
hatch.add({"op": "hatch", "region": [[130, y - 20], [480, y - 20], [480, y + 15], [130, y + 15]], "cross": True,
           "gap": 5, "color": "#2b3f6b"})
hatch.add({"op": "brush", "brush": "watercolor", "color": "#c46a4a", "points": [[560, y], [860, y - 4]]})
hatch.add({"op": "smudge", "points": [[700, y - 30], [720, y + 20]], "size": 30, "strength": 0.9})
labels.add({"op": "text", "x": 16, "y": y - 8, "text": "hatch / smudge", "size": 16, "color": "#333333"})
composite(scene).save(Path(__file__).with_suffix(".png"))
