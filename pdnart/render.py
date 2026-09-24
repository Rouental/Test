"""Render a Scene to Pillow images: one RGBA image per layer plus a composite.

Shapes are drawn at SUPERSAMPLE x resolution and downscaled, which gives
anti-aliased edges (Pillow's ImageDraw does not anti-alias on its own).
"""

from __future__ import annotations

import math
import random
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from .scene import Layer, Scene, SceneError

SUPERSAMPLE = 3


def parse_color(c: Any) -> tuple[int, int, int, int]:
    if c is None:
        return (0, 0, 0, 0)
    if isinstance(c, (list, tuple)):
        vals = [int(v) for v in c]
        if len(vals) == 3:
            vals.append(255)
        if len(vals) != 4:
            raise SceneError(f"color lists need 3 or 4 values, got {c!r}")
        return tuple(max(0, min(255, v)) for v in vals)  # type: ignore[return-value]
    try:
        rgba = ImageColor.getcolor(str(c), "RGBA")
    except ValueError as e:
        raise SceneError(f"bad color {c!r}") from e
    return rgba  # type: ignore[return-value]


def _catmull_rom(points: list[tuple[float, float]], steps: int = 12) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    pts = [points[0], *points, points[-1]]
    out: list[tuple[float, float]] = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for s in range(steps):
            t = s / steps
            t2, t3 = t * t, t * t * t
            out.append(
                tuple(  # type: ignore[arg-type]
                    0.5
                    * (2 * p1[k] + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
                       + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)
                    for k in (0, 1)
                )
            )
    out.append(points[-1])
    return out


def _star(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.45
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _gradient(op: dict[str, Any], w: int, h: int) -> Image.Image:
    stops = sorted(((float(o), parse_color(c)) for o, c in op["stops"]), key=lambda s: s[0])
    if not stops:
        raise SceneError("gradient needs at least one stop")
    # 256-entry lookup table from the stops.
    lut = []
    for i in range(256):
        t = i / 255
        if t <= stops[0][0]:
            lut.append(stops[0][1])
            continue
        if t >= stops[-1][0]:
            lut.append(stops[-1][1])
            continue
        for (o0, c0), (o1, c1) in zip(stops, stops[1:]):
            if o0 <= t <= o1:
                f = 0.0 if o1 == o0 else (t - o0) / (o1 - o0)
                lut.append(tuple(round(a + (b - a) * f) for a, b in zip(c0, c1)))
                break

    kind = op.get("kind", "linear")
    if kind == "linear":
        x0, y0 = op.get("x0", 0), op.get("y0", 0)
        x1, y1 = op.get("x1", 0), op.get("y1", h)
        dx, dy = x1 - x0, y1 - y0
        denom = dx * dx + dy * dy or 1
        t_of = lambda x, y: ((x - x0) * dx + (y - y0) * dy) / denom  # noqa: E731
    elif kind == "radial":
        cx, cy = op.get("cx", w / 2), op.get("cy", h / 2)
        r = op.get("r", max(w, h) / 2) or 1
        t_of = lambda x, y: math.hypot(x - cx, y - cy) / r  # noqa: E731
    else:
        raise SceneError(f"gradient kind must be 'linear' or 'radial', got {kind!r}")

    # Compute t on a coarse grid and let Pillow upscale it smoothly; this keeps
    # large canvases fast without numpy.
    step = max(1, min(w, h) // 256)
    gw, gh = math.ceil(w / step), math.ceil(h / step)
    tmap = Image.new("L", (gw, gh))
    tmap.putdata([
        max(0, min(255, round(t_of((gx + 0.5) * step, (gy + 0.5) * step) * 255)))
        for gy in range(gh) for gx in range(gw)
    ])
    if step > 1:
        tmap = tmap.resize((w, h), Image.BILINEAR)
    bands = [tmap.point([c[i] for c in lut]) for i in range(4)]
    return Image.merge("RGBA", bands)


def render_layer(layer: Layer, w: int, h: int) -> Image.Image:
    s = SUPERSAMPLE
    img = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))

    def flush_to(base: Image.Image, overlay: Image.Image) -> Image.Image:
        return Image.alpha_composite(base, overlay)

    for op in layer.ops:
        kind = op["op"]
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        sc = lambda v: v * s  # noqa: E731
        pts = [(sc(x), sc(y)) for x, y in op.get("points", [])]
        width = round(sc(op.get("width", 1)))

        if kind == "fill":
            over = Image.new("RGBA", img.size, parse_color(op["color"]))
        elif kind == "rect":
            box = [sc(op["x"]), sc(op["y"]), sc(op["x"] + op["w"]), sc(op["y"] + op["h"])]
            kw = dict(fill=parse_color(op.get("fill")) if op.get("fill") else None,
                      outline=parse_color(op.get("stroke")) if op.get("stroke") else None, width=width)
            if op.get("radius"):
                d.rounded_rectangle(box, radius=sc(op["radius"]), **kw)
            else:
                d.rectangle(box, **kw)
        elif kind == "ellipse":
            box = [sc(op["cx"] - op["rx"]), sc(op["cy"] - op["ry"]), sc(op["cx"] + op["rx"]), sc(op["cy"] + op["ry"])]
            d.ellipse(box, fill=parse_color(op.get("fill")) if op.get("fill") else None,
                      outline=parse_color(op.get("stroke")) if op.get("stroke") else None, width=width)
        elif kind == "polygon":
            if op.get("fill"):
                d.polygon(pts, fill=parse_color(op["fill"]))
            if op.get("stroke"):
                d.line([*pts, pts[0]], fill=parse_color(op["stroke"]), width=width, joint="curve")
        elif kind in ("line", "stroke"):
            color = parse_color(op.get("color", "#000000"))
            if kind == "stroke" and op.get("smooth", True):
                pts = _catmull_rom(pts)
            if len(pts) > 1:
                d.line(pts, fill=color, width=width, joint="curve")
            if kind == "stroke":  # round caps (and a dot for single-point strokes)
                r = width / 2
                for x, y in (pts[0], pts[-1]):
                    d.ellipse([x - r, y - r, x + r, y + r], fill=color)
        elif kind == "gradient":
            over = _gradient(op, w, h).resize(img.size, Image.BILINEAR)
        elif kind == "scatter":
            rng = random.Random(op.get("seed", 0))
            rx, ry, rw, rh = op.get("region", [0, 0, w, h])
            rmin, rmax = op.get("radius", [1, 2])
            colors = [parse_color(c) for c in op.get("colors", ["#ffffff"])]
            for _ in range(int(op["count"])):
                x, y = sc(rx + rng.random() * rw), sc(ry + rng.random() * rh)
                r = sc(rng.uniform(rmin, rmax))
                c = rng.choice(colors)
                if op.get("shape", "circle") == "star":
                    d.polygon(_star(x, y, r), fill=c)
                else:
                    d.ellipse([x - r, y - r, x + r, y + r], fill=c)
        elif kind == "text":
            size = round(sc(op.get("size", 24)))
            font = ImageFont.truetype(op["font"], size) if op.get("font") else ImageFont.load_default(size)
            d.text((sc(op["x"]), sc(op["y"])), str(op["text"]), fill=parse_color(op.get("color", "#000000")),
                   font=font, anchor=op.get("anchor", "la"))
        elif kind == "blur":
            img = img.filter(ImageFilter.GaussianBlur(sc(op["radius"])))
            continue

        img = flush_to(img, over)

    return img.resize((w, h), Image.LANCZOS)


def render_layers(scene: Scene) -> list[tuple[Layer, Image.Image]]:
    return [(layer, render_layer(layer, scene.width, scene.height)) for layer in scene.layers]


def apply_opacity(img: Image.Image, opacity: float) -> Image.Image:
    if opacity >= 1:
        return img
    r, g, b, a = img.split()
    a = a.point(lambda v: round(v * max(0.0, opacity)))
    return Image.merge("RGBA", (r, g, b, a))


def background_image(scene: Scene) -> Image.Image:
    return Image.new("RGBA", (scene.width, scene.height), parse_color(scene.background))


def composite(scene: Scene, rendered: list[tuple[Layer, Image.Image]] | None = None) -> Image.Image:
    out = background_image(scene)
    for layer, img in rendered if rendered is not None else render_layers(scene):
        if layer.visible:
            out = Image.alpha_composite(out, apply_opacity(img, layer.opacity))
    return out
