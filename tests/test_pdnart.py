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
