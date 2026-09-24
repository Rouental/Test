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

    m = sub.add_parser("mcp", help="run the MCP server over stdio (for Claude Desktop / Claude Code)")
    m.set_defaults(func=cmd_mcp)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
