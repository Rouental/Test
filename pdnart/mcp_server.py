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

from .render import composite
from .scene import OP_HELP, Scene

server = _Server(
    "pdnart",
    instructions=(
        "Paint pictures as layered scenes and open them in paint.net. Start with new_canvas, add layers "
        "back-to-front, draw ops onto them, call preview to look at your work and refine it, then "
        "send_to_paintnet. Call op_reference for the list of drawing ops."
    ),
)

_state: dict[str, Scene] = {"scene": Scene(800, 600)}


def _scene() -> Scene:
    return _state["scene"]


@server.tool()
def op_reference() -> dict[str, str]:
    """List every drawing op and its parameters. Coordinates are canvas pixels, origin top-left.
    Colors are '#rrggbb', '#rrggbbaa', CSS names, or [r,g,b(,a)]."""
    return OP_HELP


@server.tool()
def new_canvas(width: int = 800, height: int = 600, background: str | None = "#ffffff") -> str:
    """Start a fresh scene. background=None gives a transparent canvas."""
    _state["scene"] = Scene(width, height, background)
    return f"new {width}x{height} canvas"


@server.tool()
def add_layer(name: str, opacity: float = 1.0) -> str:
    """Add a layer on top of the existing ones (layers are painted back-to-front)."""
    _scene().add_layer(name, opacity)
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
def set_layer(layer: str, opacity: float | None = None, visible: bool | None = None,
              rename: str | None = None, clear: bool = False) -> str:
    """Change a layer's opacity/visibility/name, or clear its ops."""
    target = _scene().layer(layer)
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
def preview(max_size: int = 800) -> MCPImage:
    """Render the current scene and return it as an image so you can see your work."""
    img = composite(_scene())
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

    PaintDotNet().send_scene(_scene(), save_pdn)
    return "sent to paint.net" + (f" and saved to {save_pdn}" if save_pdn else "")


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
