import json
from pathlib import Path

import pytest

from pdnart import Scene, SceneError, composite
from pdnart.cli import main as cli_main
from pdnart.render import parse_color, render_layer

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_parse_color_forms():
    assert parse_color("#ff0000") == (255, 0, 0, 255)
    assert parse_color("#00ff0080") == (0, 255, 0, 128)
    assert parse_color([1, 2, 3]) == (1, 2, 3, 255)
    assert parse_color("navy") == (0, 0, 128, 255)
    with pytest.raises(SceneError):
        parse_color("not-a-color")


def test_op_validation():
    scene = Scene(10, 10)
    layer = scene.add_layer("a")
    with pytest.raises(SceneError, match="unknown op"):
        layer.add({"op": "spiral"})
    with pytest.raises(SceneError, match="missing"):
        layer.add({"op": "rect", "x": 0})
    with pytest.raises(SceneError, match="does not accept"):
        layer.add({"op": "blur", "radius": 1, "colour": "red"})
    with pytest.raises(SceneError, match="already exists"):
        scene.add_layer("a")
    with pytest.raises(SceneError, match="'points' or 'paths'"):
        layer.add({"op": "brush", "brush": "oil"})
    with pytest.raises(SceneError, match="unknown brush"):
        layer.add({"op": "brush", "brush": "crayon", "points": [[0, 0]]})
    with pytest.raises(SceneError, match="blend"):
        scene.add_layer("b", blend="sparkle")


def test_shapes_land_where_expected():
    scene = Scene(100, 100, background=None)
    scene.add_layer("shapes").add({"op": "rect", "x": 10, "y": 10, "w": 30, "h": 30, "fill": "#ff0000"})
    img = composite(scene)
    assert img.getpixel((25, 25)) == (255, 0, 0, 255)
    assert img.getpixel((80, 80))[3] == 0


def test_layer_order_and_opacity():
    scene = Scene(20, 20, background="#000000")
    scene.add_layer("bottom").add({"op": "fill", "color": "#ff0000"})
    scene.add_layer("top", opacity=0.5).add({"op": "fill", "color": "#0000ff"})
    r, g, b, a = composite(scene).getpixel((10, 10))
    assert abs(r - 128) <= 2 and abs(b - 128) <= 2 and a == 255
    scene.layer("top").visible = False
    assert composite(scene).getpixel((10, 10)) == (255, 0, 0, 255)


def test_gradient_endpoints():
    scene = Scene(64, 256, background=None)
    scene.add_layer("g").add({"op": "gradient", "x0": 0, "y0": 0, "x1": 0, "y1": 256,
                              "stops": [[0, "#000000"], [1, "#ffffff"]]})
    img = composite(scene)
    assert img.getpixel((32, 1))[0] < 16
    assert img.getpixel((32, 254))[0] > 240


def test_every_brush_preset_paints():
    from pdnart.brushes import PRESETS

    scene = Scene(300, 40 * len(PRESETS), background=None)
    layer = scene.add_layer("brushes")
    for i, name in enumerate(PRESETS):
        layer.add({"op": "brush", "brush": name, "color": "#000000",
                   "points": [[20, 20 + 40 * i, 0.3], [150, 10 + 40 * i, 1.0], [280, 20 + 40 * i, 0.5]]})
    img = composite(scene)
    for i, name in enumerate(PRESETS):
        band = img.crop((0, 40 * i, 300, 40 * i + 40))
        assert band.getchannel("A").getextrema()[1] > 60, f"{name} left no visible paint"


def test_pressure_changes_width():
    def width_at(pressure):
        scene = Scene(100, 60, background=None)
        scene.add_layer("l").add({"op": "brush", "brush": "ink", "size": 20, "taper": 0, "smooth": False,
                                  "points": [[10, 30, pressure], [90, 30, pressure]]})
        alpha = composite(scene).getchannel("A")
        return sum(alpha.getpixel((50, y)) > 128 for y in range(60))

    assert width_at(1.0) > width_at(0.3) + 5


def test_lock_alpha_clip_and_erase():
    scene = Scene(100, 100, background=None)
    layer = scene.add_layer("l")
    layer.add({"op": "rect", "x": 20, "y": 20, "w": 60, "h": 60, "fill": "#ff0000"})
    layer.add({"op": "fill", "color": "#0000ff", "lock_alpha": True})  # recolor only the square
    img = composite(scene)
    assert img.getpixel((50, 50)) == (0, 0, 255, 255)
    assert img.getpixel((5, 5))[3] == 0

    layer.add({"op": "fill", "color": "#00ff00", "clip": [[0, 0], [50, 0], [50, 100], [0, 100]]})
    img = composite(scene)
    assert img.getpixel((10, 10)) == (0, 255, 0, 255)
    assert img.getpixel((70, 50)) == (0, 0, 255, 255)

    layer.add({"op": "brush", "brush": "round", "size": 30, "erase": True, "points": [[70, 50]]})
    assert composite(scene).getpixel((70, 50))[3] == 0


def test_smudge_blends_without_spreading_when_locked():
    scene = Scene(100, 60, background=None)
    layer = scene.add_layer("l")
    layer.add({"op": "rect", "x": 20, "y": 10, "w": 30, "h": 40, "fill": "#000000"})
    layer.add({"op": "rect", "x": 50, "y": 10, "w": 30, "h": 40, "fill": "#ffffff"})
    layer.add({"op": "smudge", "points": [[30, 30], [70, 30]], "size": 30, "strength": 1, "lock_alpha": True})
    img = composite(scene)
    r = img.getpixel((50, 30))[0]
    assert 40 < r < 215  # the black/white edge is blended
    assert img.getpixel((10, 30))[3] == 0  # nothing leaked outside the shapes


def test_multiply_blend():
    scene = Scene(10, 10, background="#80ff80")
    scene.add_layer("m", blend="multiply").add({"op": "fill", "color": "#ff8080"})
    r, g, b, _ = composite(scene).getpixel((5, 5))
    assert abs(r - 128) <= 2 and abs(g - 128) <= 2 and abs(b - 64) <= 2


def test_hatch_stays_inside_region():
    scene = Scene(100, 100, background=None)
    scene.add_layer("h").add({"op": "hatch", "region": [[20, 20], [80, 20], [80, 80], [20, 80]], "cross": True,
                              "color": "#000000"})
    alpha = composite(scene).getchannel("A")
    assert alpha.crop((25, 25, 75, 75)).getextrema()[1] > 100
    assert alpha.crop((0, 0, 100, 15)).getextrema()[1] == 0


def _disc(cx, cy, r, n=24):
    import math

    return [[cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)] for k in range(n)]


def test_form_is_lit_from_the_light_direction():
    scene = Scene(120, 120, background=None)
    scene.add_layer("f").add({"op": "form", "region": _disc(60, 60, 50), "inflate": 50, "light": [-1, 0, 0.3],
                              "ramp": [[0, "#000000"], [1, "#ffffff"]]})
    img = composite(scene)
    left, right = img.getpixel((25, 60))[0], img.getpixel((95, 60))[0]
    assert left > right + 80  # the side facing the light is much brighter
    assert img.getpixel((2, 2))[3] == 0  # nothing painted outside the silhouette
    # a hollow on the lit side catches less light than the surface around it
    scene.layers[0].ops[0]["bumps"] = [{"x": 40, "y": 60, "rx": 8, "ry": 8, "h": -15}]
    scene.layers[0].ops[0] = dict(scene.layers[0].ops[0])
    assert composite(scene).getpixel((34, 60))[0] < left


def test_painterly_repaints_a_study_with_strokes():
    scene = Scene(160, 120, background="#ffffff")
    study = scene.add_layer("study")
    study.visible = False
    study.add({"op": "rect", "x": 0, "y": 0, "w": 80, "h": 120, "fill": "#cc3322"})
    study.add({"op": "rect", "x": 80, "y": 0, "w": 80, "h": 120, "fill": "#2244aa"})
    scene.add_layer("paint").add({"op": "painterly", "source": "study", "sizes": [16, 8], "seed": 1})
    img = composite(scene)
    r, g, b, _ = img.getpixel((30, 60))
    assert r > 150 and b < 90
    r, g, b, _ = img.getpixel((130, 60))
    assert b > 120 and r < 90
    # changing the study invalidates the painted layer's cache
    study.ops[0] = {"op": "rect", "x": 0, "y": 0, "w": 80, "h": 120, "fill": "#22aa44"}
    assert composite(scene).getpixel((30, 60))[1] > 120


def test_painterly_and_copy_from_an_image_file(tmp_path):
    from PIL import Image

    Image.new("RGB", (40, 30), "#10a050").save(tmp_path / "ref.png")
    scene = Scene(60, 40, background=None)
    layer = scene.add_layer("p")
    layer.add({"op": "painterly", "source": {"path": "ref.png", "x": 10, "y": 5}, "sizes": [8, 4]})
    scene.save(tmp_path / "scene.json")
    loaded = Scene.load(tmp_path / "scene.json")  # relative paths resolve next to the scene file
    img = composite(loaded)
    assert img.getpixel((30, 20))[1] > 120
    assert img.getpixel((2, 2))[3] == 0
    loaded.layers[0].ops = [{"op": "copy", "source": {"path": "ref.png"}, "opacity": 0.5}]
    assert 100 < composite(loaded).getpixel((5, 5))[3] < 160


def test_painterly_source_validation():
    layer = Scene(10, 10).add_layer("l")
    with pytest.raises(SceneError, match="source"):
        layer.add({"op": "painterly", "source": 5})
    layer.add({"op": "painterly", "source": ["a", "b"]})


def test_every_op_renders():
    scene = Scene(120, 80)
    layer = scene.add_layer("all")
    for op in [
        {"op": "fill", "color": "#eeeeee"},
        {"op": "rect", "x": 5, "y": 5, "w": 20, "h": 10, "fill": "red", "stroke": "black", "width": 2, "radius": 3},
        {"op": "ellipse", "cx": 60, "cy": 40, "rx": 10, "ry": 6, "stroke": "blue", "width": 2},
        {"op": "polygon", "points": [[0, 0], [10, 0], [5, 8]], "fill": "green", "stroke": "black"},
        {"op": "line", "points": [[0, 79], [119, 0]], "color": "purple", "width": 2},
        {"op": "stroke", "points": [[10, 60], [30, 70], [50, 60], [70, 70]], "width": 4},
        {"op": "stroke", "points": [[100, 60]], "width": 6},
        {"op": "brush", "brush": "oil", "color": "#884422", "paths": [[[10, 10], [60, 30]], [[10, 20], [60, 40, 0.5]]]},
        {"op": "hatch", "region": [[0, 0], [40, 0], [40, 40]], "angle": 30},
        {"op": "smudge", "points": [[10, 10], [50, 50]], "size": 12},
        {"op": "ellipse", "cx": 20, "cy": 20, "rx": 10, "ry": 4, "angle": 30, "fill": "red", "feather": 2},
        {"op": "polygon", "points": [[60, 10], [90, 20], [80, 40]], "smooth": True, "fill": "blue", "clip": {"points": [[50, 0], [120, 0], [120, 30], [50, 30]], "feather": 2}},
        {"op": "gradient", "kind": "radial", "cx": 60, "cy": 40, "r": 30, "stops": [[0, "#ffffff80"], [1, "#ffffff00"]]},
        {"op": "scatter", "count": 10, "shape": "star", "radius": [2, 3], "seed": 1},
        {"op": "text", "x": 60, "y": 70, "text": "hi", "size": 12, "anchor": "mm"},
        {"op": "blur", "radius": 0.5},
    ]:
        layer.add(op)
    assert render_layer(layer, 120, 80).size == (120, 80)


def test_scene_json_round_trip(tmp_path):
    scene = Scene.load(EXAMPLES / "moonlit_mountains.json")
    path = tmp_path / "s.json"
    scene.save(path)
    assert Scene.load(path).to_dict() == scene.to_dict()


def test_cli_render_writes_layers(tmp_path):
    cli_main(["render", str(EXAMPLES / "moonlit_mountains.json"), "-o", str(tmp_path / "out.png"),
              "--layers", str(tmp_path / "layers")])
    assert (tmp_path / "out.png").exists()
    assert len(list((tmp_path / "layers").glob("*.png"))) == 7


@pytest.mark.parametrize("name", ["landscape", "portrait"])
def test_example_scenes_load(name):
    scene = Scene.load(EXAMPLES / f"{name}.json")
    assert len(scene.layers) >= 6
    for layer in scene.layers:  # every painterly/copy source refers to a real layer
        for op in layer.ops:
            src = op.get("source")
            for n in [src] if isinstance(src, str) else src if isinstance(src, list) else []:
                scene.layer(n)


def test_mcp_tools_build_and_preview():
    mcp_server = pytest.importorskip("pdnart.mcp_server")
    mcp_server.new_canvas(200, 100, "#112233")
    mcp_server.add_layer("sun")
    mcp_server.draw("sun", [{"op": "ellipse", "cx": 100, "cy": 50, "rx": 20, "ry": 20, "fill": "#ffcc00"}])
    with pytest.raises(SceneError):
        mcp_server.draw("sun", [{"op": "rect", "x": 0}])  # rejected batch leaves the layer untouched
    assert len(mcp_server.get_scene()["layers"][0]["ops"]) == 1
    img = mcp_server.preview()
    assert img.data.startswith(b"\x89PNG")
    assert mcp_server.preview(grid=True, region=[50, 0, 100, 100]).data.startswith(b"\x89PNG")
    mcp_server.set_layer("sun", blend="screen")
    with pytest.raises(SceneError):
        mcp_server.set_layer("sun", blend="sparkle")
    assert "oil" in mcp_server.op_reference()["brushes"]
    json.dumps(mcp_server.op_reference())


def test_cli_paint_repaints_a_photo(tmp_path):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (120, 80), "#203050")
    ImageDraw.Draw(img).ellipse([30, 10, 90, 70], fill="#e0a060")
    img.save(tmp_path / "photo.png")
    cli_main(["paint", str(tmp_path / "photo.png"), "-o", str(tmp_path / "out.png"), "--detail", "low",
              "--scene", str(tmp_path / "scene.json")])
    out = Image.open(tmp_path / "out.png").convert("RGB")
    assert out.size == (120, 80)
    from PIL import ImageStat

    r, g, b = ImageStat.Stat(out.crop((45, 25, 75, 55))).mean
    assert r > 170 and b < 130, (r, g, b)  # the orange disc was painted
    r, g, b = ImageStat.Stat(out.crop((0, 0, 15, 80))).mean
    assert b > r, (r, g, b)  # the blue background too
    assert Scene.load(tmp_path / "scene.json").layers[0].ops[0]["op"] == "painterly"
