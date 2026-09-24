"""MCP server that lets Claude (Desktop or Code) paint a scene and push it into paint.net.

Typical loop for the model: new_canvas -> add_layer -> draw -> preview (look at
the result, adjust) -> send_to_paintnet.
"""

from __future__ import annotations

import io
import json
from typing import Any

try:  # mcp >= 2
    from mcp.server.mcpserver import Image as MCPImage
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef]
    from mcp.server.fastmcp import Image as MCPImage  # type: ignore[no-redef]

from PIL import ImageDraw

from .render import composite
from .scene import BRUSH_HELP, OP_HELP, Scene

server = _Server(
    "pdnart",
    instructions=(
        "Paint pictures as layered scenes and open them in paint.net. Call op_reference first for the drawing "
        "ops and brushes. Work like a painter, back to front: a pencil sketch layer (brush='pencil', multiply "
        "blend) to place things; flat block-in shapes (polygon smooth=true); then model light and shadow with "
        "brush strokes clipped to those shapes (lock_alpha=true or clip=...); blend with smudge; add detail "
        "with small brushes; finish with ink/pencil linework and hatching on a multiply layer. Paint many "
        "strokes at once with paths=[...]. Call preview often (grid=true to read coordinates, region=[x,y,w,h] "
        "to zoom into details), fix what looks wrong (undo, set_layer), then send_to_paintnet."
    ),
)

_state: dict[str, Scene] = {"scene": Scene(800, 600)}


def _scene() -> Scene:
    return _state["scene"]


@server.tool()
def op_reference() -> dict[str, dict[str, str]]:
    """List every drawing op and its parameters, and the brush presets. Coordinates are canvas pixels,
    origin top-left. Colors are '#rrggbb', '#rrggbbaa', CSS names, or [r,g,b(,a)]."""
    return {"ops": OP_HELP, "brushes": BRUSH_HELP}


@server.tool()
def new_canvas(width: int = 800, height: int = 600, background: str | None = "#ffffff") -> str:
    """Start a fresh scene. background=None gives a transparent canvas."""
    _state["scene"] = Scene(width, height, background)
    return f"new {width}x{height} canvas"


@server.tool()
def add_layer(name: str, opacity: float = 1.0, blend: str = "normal") -> str:
    """Add a layer on top of the existing ones (layers are painted back-to-front). blend: normal, multiply,
    screen, overlay, darken, lighten, additive, difference."""
    _scene().add_layer(name, opacity, blend)
    return f"layers: {[l.name for l in _scene().layers]}"


@server.tool()
def draw(layer: str, ops: list[dict[str, Any]]) -> str:
    """Append drawing ops to a layer, e.g. [{"op": "ellipse", "cx": 400, "cy": 300, "rx": 80, "ry": 80,
    "fill": "#ffcc00"}]. All ops are validated before any are applied."""
    target = _scene().layer(layer)
    staged = type(target)(target.name)
    for op in ops:
        staged.add(op)
    target.ops.extend(staged.ops)
    return f"{layer}: {len(target.ops)} ops"


@server.tool()
def undo(layer: str, count: int = 1) -> str:
    """Remove the last `count` ops from a layer."""
    target = _scene().layer(layer)
    del target.ops[max(0, len(target.ops) - count):]
    return f"{layer}: {len(target.ops)} ops"


@server.tool()
def set_layer(layer: str, opacity: float | None = None, visible: bool | None = None, blend: str | None = None,
              rename: str | None = None, clear: bool = False) -> str:
    """Change a layer's opacity/visibility/blend mode/name, or clear its ops."""
    target = _scene().layer(layer)
    if blend is not None:
        type(target)(target.name, blend=blend)  # validates the mode
        target.blend = blend
    if opacity is not None:
        target.opacity = opacity
    if visible is not None:
        target.visible = visible
    if rename:
        target.name = rename
    if clear:
        target.ops.clear()
    return json.dumps({k: v for k, v in target.to_dict().items() if k != "ops"})


@server.tool()
def preview(max_size: int = 800, grid: bool = False, region: list[float] | None = None) -> MCPImage:
    """Render the scene and return it as an image so you can see your work. grid=true overlays labelled
    canvas coordinates every 100 px. region=[x, y, w, h] zooms into part of the canvas (for details such
    as eyes or hands)."""
    img = composite(_scene()).convert("RGB")
    if grid:
        d = ImageDraw.Draw(img)
        for x in range(100, img.width, 100):
            d.line([(x, 0), (x, img.height)], fill=(255, 0, 180), width=1)
            d.text((x + 2, 2), str(x), fill=(255, 0, 180))
        for y in range(100, img.height, 100):
            d.line([(0, y), (img.width, y)], fill=(255, 0, 180), width=1)
            d.text((2, y + 2), str(y), fill=(255, 0, 180))
    if region:
        x, y, w, h = (int(v) for v in region)
        img = img.crop((x, y, x + w, y + h))
        scale = min(max_size / max(1, img.width), max_size / max(1, img.height))
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    else:
        img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return MCPImage(data=buf.getvalue(), format="png")


@server.tool()
def get_scene() -> dict[str, Any]:
    """Return the whole scene as JSON."""
    return _scene().to_dict()


@server.tool()
def load_scene(scene: dict[str, Any]) -> str:
    """Replace the current scene with a full scene JSON object (same shape as get_scene)."""
    _state["scene"] = Scene.from_dict(scene)
    return f"loaded {len(_scene().layers)} layers"


@server.tool()
def save_files(path: str) -> str:
    """Save the scene as JSON (path ending .json) or a flattened image (.png/.jpg)."""
    if path.lower().endswith(".json"):
        _scene().save(path)
    else:
        img = composite(_scene())
        (img if path.lower().endswith(".png") else img.convert("RGB")).save(path)
    return f"saved {path}"


@server.tool()
def send_to_paintnet(save_pdn: str | None = None) -> str:
    """Open the scene in paint.net (launching it if needed) as a layered document. Windows only.
    Optionally save it as a .pdn file. Keep hands off the mouse/keyboard while this runs."""
    from .paintnet import PaintDotNet

    warnings = PaintDotNet().send_scene(_scene(), save_pdn)
    msg = "sent to paint.net" + (f" and saved to {save_pdn}" if save_pdn else "")
    return "\n".join([msg, *warnings])


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
