"""Command line: render scenes to PNGs, send them to paint.net, or run the MCP server."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .render import apply_opacity, composite, render_layers
from .scene import Scene


def cmd_render(args: argparse.Namespace) -> None:
    scene = Scene.load(args.scene)
    rendered = render_layers(scene)
    composite(scene).save(args.output)
    print(f"wrote {args.output}")
    if args.layers:
        out = Path(args.layers)
        out.mkdir(parents=True, exist_ok=True)
        for i, (layer, img) in enumerate(rendered, 1):
            safe = re.sub(r"[^\w.-]+", "_", layer.name)
            apply_opacity(img, layer.opacity).save(out / f"{i:02d}_{safe}.png")
        print(f"wrote {len(rendered)} layer PNGs to {out}")


def cmd_send(args: argparse.Namespace) -> None:
    from .paintnet import PaintDotNet

    for warning in PaintDotNet(exe=args.exe, delay=args.delay).send_scene(Scene.load(args.scene), args.save):
        print("warning:", warning)
    print("sent to paint.net")


DETAIL = {  # (brush sizes as fractions of the long side, threshold)
    "low": ([0.05, 0.025, 0.0125], 0.06),
    "medium": ([0.05, 0.025, 0.0125, 0.00625], 0.045),
    "high": ([0.05, 0.025, 0.0125, 0.00625, 0.004], 0.04),
}


def photo_scene(photo: str, width: int | None = None, detail: str = "medium", brush: str = "oil") -> Scene:
    """A scene that repaints `photo` with brush strokes: coarse to fine, then small dabs for highlights."""
    from PIL import Image

    with Image.open(photo) as im:
        w, h = im.size
    if width:
        w, h = width, round(h * width / w)
    long_side = max(w, h)
    fractions, threshold = DETAIL[detail]
    sizes = [max(2.5, round(f * long_side, 1)) for f in fractions]
    src = {"path": str(Path(photo).resolve()), "w": w, "h": h}
    return Scene.from_dict({"width": w, "height": h, "background": "#000000", "layers": [{"name": "Painting", "ops": [
        {"op": "painterly", "source": src, "brush": brush, "sizes": sizes, "threshold": threshold,
         "length": [2, 10], "curvature": 0.8, "color_jitter": 0.01, "seed": 1},
        {"op": "painterly", "source": src, "brush": "round", "sizes": [2], "threshold": 0.1, "length": [0, 2],
         "grid": 0.9, "blur": 0, "seed": 2},  # highlights: stars, catchlights, glints
    ]}]})


def cmd_paint(args: argparse.Namespace) -> None:
    scene = photo_scene(args.photo, args.width, args.detail, args.brush)
    if args.scene:
        scene.save(args.scene)
        print(f"wrote {args.scene}")
    composite(scene).save(args.output)
    print(f"wrote {args.output}")
    if args.send:
        from .paintnet import PaintDotNet

        for warning in PaintDotNet().send_scene(scene, args.save):
            print("warning:", warning)
        print("sent to paint.net")


def cmd_mcp(_: argparse.Namespace) -> None:
    from .mcp_server import main

    main()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pdnart", description="Make art as layered scenes for paint.net")
    sub = p.add_subparsers(required=True)

    r = sub.add_parser("render", help="render a scene JSON to a PNG (works on any OS)")
    r.add_argument("scene")
    r.add_argument("-o", "--output", default="out.png")
    r.add_argument("--layers", metavar="DIR", help="also write each layer as its own PNG")
    r.set_defaults(func=cmd_render)

    s = sub.add_parser("send", help="open a scene in paint.net as a layered image (Windows)")
    s.add_argument("scene")
    s.add_argument("--save", metavar="FILE.pdn", help="save the result as a .pdn file")
    s.add_argument("--exe", help="path to paintdotnet.exe")
    s.add_argument("--delay", type=float, default=0.6, help="seconds to wait between UI steps")
    s.set_defaults(func=cmd_send)

    pp = sub.add_parser("paint", help="repaint a photo or picture with brush strokes")
    pp.add_argument("photo")
    pp.add_argument("-o", "--output", default="painting.png")
    pp.add_argument("--width", type=int, help="resize to this width first (smaller is faster)")
    pp.add_argument("--detail", choices=sorted(DETAIL), default="medium")
    pp.add_argument("--brush", default="oil", help="brush preset for the strokes (oil, dry, watercolor, ...)")
    pp.add_argument("--scene", metavar="FILE.json", help="also save the scene, to edit or re-render later")
    pp.add_argument("--send", action="store_true", help="open the painting in paint.net (Windows)")
    pp.add_argument("--save", metavar="FILE.pdn", help="with --send, save it as a .pdn file")
    pp.set_defaults(func=cmd_paint)

    m = sub.add_parser("mcp", help="run the MCP server over stdio (for Claude Desktop / Claude Code)")
    m.set_defaults(func=cmd_mcp)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
