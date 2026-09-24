"""Render a Scene: each layer becomes a premultiplied float raster, then layers are blended.

Shapes are rasterised at SUPERSAMPLE x resolution for anti-aliasing. Brushes are
anti-aliased by the dab engine itself (see brushes.py).
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from collections import OrderedDict
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .brushes import Brush, Paper, _stamp, make_brush, stroke_masks
from .raster import composite_over, from_image, gblur, new_layer, paint, to_image
from .scene import Layer, Scene, SceneError

SUPERSAMPLE = 4
_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
_CACHE_SIZE = 48


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
        return ImageColor.getcolor(str(c), "RGBA")  # type: ignore[return-value]
    except ValueError as e:
        raise SceneError(f"bad color {c!r}") from e


def rgba01(c: Any) -> tuple[float, float, float, float]:
    return tuple(v / 255 for v in parse_color(c))  # type: ignore[return-value]


def _catmull_rom_closed(pts: list, steps: int = 12) -> list[tuple[float, float]]:
    n = len(pts)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = (np.asarray(pts[(i + k) % n][:2], float) for k in (-1, 0, 1, 2))
        for s in range(steps):
            t = s / steps
            out.append(tuple(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)))
    return out


def _star(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    return [(cx + (r if i % 2 == 0 else r * 0.45) * math.cos(-math.pi / 2 + i * math.pi / 5),
             cy + (r if i % 2 == 0 else r * 0.45) * math.sin(-math.pi / 2 + i * math.pi / 5)) for i in range(10)]


class LayerRenderer:
    def __init__(self, w: int, h: int, paper: Paper):
        self.w, self.h, self.paper = w, h, paper

    # -- helpers -----------------------------------------------------------
    def _raster(self, bbox, draw: Callable[[ImageDraw.ImageDraw, Callable], None], feather: float = 0):
        """Rasterise `draw` inside bbox at supersampled resolution. Returns (mask, x0, y0) or None."""
        pad = 2 + math.ceil(3 * feather)
        x0 = max(0, math.floor(bbox[0]) - pad)
        y0 = max(0, math.floor(bbox[1]) - pad)
        x1 = min(self.w, math.ceil(bbox[2]) + pad)
        y1 = min(self.h, math.ceil(bbox[3]) + pad)
        if x0 >= x1 or y0 >= y1:
            return None
        s = SUPERSAMPLE
        img = Image.new("L", ((x1 - x0) * s, (y1 - y0) * s))
        draw(ImageDraw.Draw(img), lambda pts: [((x - x0) * s, (y - y0) * s) for x, y in pts])
        mask = np.asarray(img.reduce(s), np.float32) / 255
        if feather:
            mask = gblur(mask, feather)
        return mask, x0, y0

    def _shape(self, layer, op, outline_pts, closed_poly=True, draw_fill=None):
        """Fill and/or outline a closed shape given as a point list."""
        opts = _paint_opts(op)
        xs, ys = [p[0] for p in outline_pts], [p[1] for p in outline_pts]
        width = float(op.get("width", 1))
        bbox = (min(xs) - width, min(ys) - width, max(xs) + width, max(ys) + width)
        feather = float(op.get("feather", 0))
        if op.get("fill"):
            res = self._raster(bbox, draw_fill or (lambda d, T: d.polygon(T(outline_pts), fill=255)), feather)
            if res:
                paint(layer, *res, rgba01(op["fill"]), **opts)
        if op.get("stroke"):
            ss = SUPERSAMPLE

            def outline(d, T):
                pts = T(outline_pts)
                d.line([*pts, pts[0]] if closed_poly else pts, fill=255, width=max(1, round(width * ss)),
                       joint="curve")
            res = self._raster(bbox, outline, feather)
            if res:
                paint(layer, *res, rgba01(op["stroke"]), **opts)

    def _brush_paths(self, layer, op, b: Brush, color, rng, lock_alpha=False, erase=False, opacity=1.0):
        paths = op.get("paths") or [op["points"]]
        for path in paths:
            for mask, x0, y0, rgba in stroke_masks(path, b, color, rng, self.paper):
                paint(layer, mask, x0, y0, rgba, lock_alpha=lock_alpha, erase=erase, opacity=opacity)

    # -- ops ---------------------------------------------------------------
    def render(self, layer_spec: Layer) -> np.ndarray:
        layer = new_layer(self.w, self.h)
        for i, op in enumerate(layer_spec.ops):
            seed = op.get("seed", zlib.crc32(f"{layer_spec.name}:{i}".encode()))
            rng = np.random.default_rng(seed)
            try:
                layer = self.apply(layer, op, rng)
            except SceneError:
                raise
            except Exception as e:  # surface which op failed, e.g. a missing font file
                raise SceneError(f"layer {layer_spec.name!r} op #{i} ({op['op']}): {e}") from e
        return layer

    def apply(self, layer: np.ndarray, op: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
        if op.get("clip") is None:
            return self._apply(layer, op, rng)
        # A clip works like a selection: whatever the op does is kept only inside it.
        before = layer.copy()
        after = self._apply(layer, op, rng)
        cm = self.clip_mask(op["clip"])
        return before + (after - before) * cm[..., None]

    def clip_mask(self, clip: Any) -> np.ndarray:
        spec = clip if isinstance(clip, dict) else {"points": clip}
        pts = [tuple(p[:2]) for p in spec["points"]]
        if spec.get("smooth"):
            pts = _catmull_rom_closed(pts)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        full = np.zeros((self.h, self.w), np.float32)
        res = self._raster((min(xs), min(ys), max(xs), max(ys)), lambda d, T: d.polygon(T(pts), fill=255),
                           float(spec.get("feather", 0)))
        if res:
            m, x0, y0 = res
            full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        if spec.get("invert"):
            full = 1 - full
        return full

    def _apply(self, layer: np.ndarray, op: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
        kind = op["op"]
        opts = _paint_opts(op)
        W, H = self.w, self.h

        if kind == "fill":
            paint(layer, np.ones((H, W), np.float32), 0, 0, rgba01(op["color"]), **opts)
        elif kind == "rect":
            x, y, w, h = op["x"], op["y"], op["w"], op["h"]
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            radius = float(op.get("radius", 0))
            fill = (lambda d, T: d.rounded_rectangle([*T([(x, y)])[0], *T([(x + w, y + h)])[0]],
                                                     radius=radius * SUPERSAMPLE, fill=255)) if radius else None
            self._shape(layer, op, corners, draw_fill=fill)
        elif kind == "ellipse":
            a = math.radians(op.get("angle", 0))
            pts = [(op["cx"] + op["rx"] * math.cos(t) * math.cos(a) - op["ry"] * math.sin(t) * math.sin(a),
                    op["cy"] + op["rx"] * math.cos(t) * math.sin(a) + op["ry"] * math.sin(t) * math.cos(a))
                   for t in np.linspace(0, 2 * math.pi, 180, endpoint=False)]
            self._shape(layer, op, pts)
        elif kind == "polygon":
            pts = [tuple(p[:2]) for p in op["points"]]
            if op.get("smooth"):
                pts = _catmull_rom_closed(pts)
            self._shape(layer, op, pts)
        elif kind == "brush":
            b = make_brush(op.get("brush"), op)
            self._brush_paths(layer, op, b, rgba01(op.get("color", "#000000")), rng,
                              lock_alpha=opts["lock_alpha"], erase=opts["erase"])
        elif kind in ("stroke", "line"):
            b = make_brush("round", {"size": op.get("width", 1), "hardness": 1.0, "spacing": 0.1,
                                     "smooth": kind == "stroke" and op.get("smooth", True)})
            self._brush_paths(layer, op, b, rgba01(op.get("color", "#000000")), rng, **opts)
        elif kind == "hatch":
            self._hatch(layer, op, rng)
        elif kind == "smudge":
            self._smudge(layer, op, rng)
        elif kind == "gradient":
            paint(layer, np.ones((H, W), np.float32), 0, 0, _gradient(op, W, H), **opts)
        elif kind == "scatter":
            rx, ry, rw, rh = op.get("region", [0, 0, W, H])
            rmin, rmax = op.get("radius", [1, 2])
            colors = op.get("colors", ["#ffffff"])
            n = int(op["count"])
            xs, ys = rx + rng.random(n) * rw, ry + rng.random(n) * rh
            rs = rng.uniform(rmin, rmax, n)
            which = rng.integers(0, len(colors), n)
            for ci, c in enumerate(colors):
                sel = which == ci
                if not sel.any():
                    continue
                if op.get("shape", "circle") == "star":
                    res = self._raster((0, 0, W, H), lambda d, T: [d.polygon(T(_star(x, y, r)), fill=255)
                                                                    for x, y, r in zip(xs[sel], ys[sel], rs[sel])])
                    if res:
                        paint(layer, *res, rgba01(c), **opts)
                else:
                    mask, x0, y0 = _stamp(xs[sel], ys[sel], rs[sel], np.ones(sel.sum()), 0.9, False)
                    paint(layer, mask, x0, y0, rgba01(c), **opts)
        elif kind == "text":
            size = int(op.get("size", 24))
            font = ImageFont.truetype(op["font"], size) if op.get("font") else ImageFont.load_default(size)
            img = Image.new("L", (W, H))
            ImageDraw.Draw(img).text((op["x"], op["y"]), str(op["text"]), fill=255, font=font,
                                     anchor=op.get("anchor", "la"))
            paint(layer, np.asarray(img, np.float32) / 255, 0, 0, rgba01(op.get("color", "#000000")), **opts)
        elif kind == "image":
            img = Image.open(op["path"]).convert("RGBA")
            if "w" in op or "h" in op:
                img = img.resize((int(op.get("w", img.width)), int(op.get("h", img.height))), Image.LANCZOS)
            arr = np.asarray(img, np.float32) / 255
            paint(layer, np.ones(arr.shape[:2], np.float32), int(op.get("x", 0)), int(op.get("y", 0)), arr, **opts)
        elif kind == "blur":
            layer = gblur(layer, float(op["radius"]))
        return layer

    def _hatch(self, layer, op, rng):
        region = [tuple(p[:2]) for p in op["region"]]
        xs, ys = [p[0] for p in region], [p[1] for p in region]
        clip = self._raster((min(xs), min(ys), max(xs), max(ys)), lambda d, T: d.polygon(T(region), fill=255), 0.7)
        if clip is None:
            return
        cmask, cx0, cy0 = clip
        ch, cw = cmask.shape
        b = make_brush(op.get("brush", "pencil"), op)
        acc = np.zeros_like(cmask)
        gap = float(op.get("gap", max(3.0, b.size * 2.5)))
        angles = [op.get("angle", 45)] + ([op.get("angle", 45) + 90] if op.get("cross") else [])
        mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 4
        for ang in angles:
            dx, dy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            nx, ny = -dy, dx
            o = -half
            while o <= half:
                cxl, cyl = mx + nx * o, my + ny * o
                a, e = rng.uniform(-0.1, 0.05) * half, rng.uniform(-0.05, 0.1) * half
                path = [[cxl - dx * (half + a), cyl - dy * (half + a), 0.6],
                        [cxl, cyl, 1.0],
                        [cxl + dx * (half + e), cyl + dy * (half + e), 0.6]]
                for mask, x0, y0, _ in stroke_masks(path, b, (0, 0, 0, 1), rng, self.paper):
                    # Merge into the region-sized accumulator.
                    sx0, sy0 = max(x0, cx0), max(y0, cy0)
                    sx1, sy1 = min(x0 + mask.shape[1], cx0 + cw), min(y0 + mask.shape[0], cy0 + ch)
                    if sx0 < sx1 and sy0 < sy1:
                        view = acc[sy0 - cy0:sy1 - cy0, sx0 - cx0:sx1 - cx0]
                        np.maximum(view, mask[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0], out=view)
                o += gap * rng.uniform(0.8, 1.2)
        paint(layer, acc * cmask, cx0, cy0, rgba01(op.get("color", "#333333")), **_paint_opts(op))

    def _smudge(self, layer, op, rng):
        size = float(op.get("size", 20))
        b = Brush(size=size, hardness=float(op.get("hardness", 0.2)), spacing=0.1, pressure_size=0.5,
                  smooth=op.get("smooth", True))
        strength = float(op.get("strength", 0.6))
        sigma = max(1.0, size * 0.25)
        pad = int(3 * sigma) + 2
        for path in op.get("paths") or [op["points"]]:
            for mask, x0, y0, _ in stroke_masks(path, b, (0, 0, 0, 1), rng):
                h, w = mask.shape
                rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
                rx1, ry1 = min(self.w, x0 + w + pad), min(self.h, y0 + h + pad)
                if rx0 >= rx1 or ry0 >= ry1:
                    continue
                region = layer[ry0:ry1, rx0:rx1]
                blurred = gblur(region, sigma)
                m = np.zeros(region.shape[:2], np.float32)
                sx0, sy0 = max(x0, rx0), max(y0, ry0)
                sx1, sy1 = min(x0 + w, rx1), min(y0 + h, ry1)
                m[sy0 - ry0:sy1 - ry0, sx0 - rx0:sx1 - rx0] = mask[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
                old_a = region[..., 3:4].copy()
                region += (blurred - region) * (m * strength)[..., None]
                if op.get("lock_alpha"):  # blend colors only; don't spread paint past its edges
                    with np.errstate(divide="ignore", invalid="ignore"):
                        straight = np.where(region[..., 3:4] > 1e-6, region[..., :3] / region[..., 3:4], 0)
                    region[..., :3] = straight * old_a
                    region[..., 3:4] = old_a


def _paint_opts(op: dict[str, Any]) -> dict[str, Any]:
    return {"opacity": float(op.get("opacity", 1.0)), "lock_alpha": bool(op.get("lock_alpha", False)),
            "erase": bool(op.get("erase", False))}


def _gradient(op: dict[str, Any], w: int, h: int) -> np.ndarray:
    stops = sorted(((float(o), rgba01(c)) for o, c in op["stops"]), key=lambda s: s[0])
    if not stops:
        raise SceneError("gradient needs at least one stop")
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    kind = op.get("kind", "linear")
    if kind == "linear":
        x0, y0 = op.get("x0", 0), op.get("y0", 0)
        dx, dy = op.get("x1", 0) - x0, op.get("y1", h) - y0
        t = ((xs - x0) * dx + (ys - y0) * dy) / (dx * dx + dy * dy or 1)
    elif kind == "radial":
        t = np.hypot(xs - op.get("cx", w / 2), ys - op.get("cy", h / 2)) / (op.get("r", max(w, h) / 2) or 1)
    else:
        raise SceneError(f"gradient kind must be 'linear' or 'radial', got {kind!r}")
    offs = [s[0] for s in stops]
    # Interpolate premultiplied so fading to a transparent stop has no dark fringe.
    alpha = np.interp(t, offs, [s[1][3] for s in stops])
    out = np.empty((h, w, 4), np.float32)
    for ch in range(3):
        prem = np.interp(t, offs, [s[1][ch] * s[1][3] for s in stops])
        out[..., ch] = np.where(alpha > 1e-6, prem / np.maximum(alpha, 1e-6), 0)
    out[..., 3] = alpha
    return out


def _layer_key(layer: Layer, w: int, h: int) -> str:
    return hashlib.sha1(json.dumps([w, h, layer.name, layer.ops], sort_keys=True).encode()).hexdigest()


def render_layer_array(layer: Layer, w: int, h: int) -> np.ndarray:
    """Render one layer to a premultiplied float array (cached by content)."""
    key = _layer_key(layer, w, h)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    arr = LayerRenderer(w, h, Paper(w, h)).render(layer)
    _CACHE[key] = arr
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return arr


def render_layer(layer: Layer, w: int, h: int) -> Image.Image:
    return to_image(render_layer_array(layer, w, h))


def render_layers(scene: Scene) -> list[tuple[Layer, Image.Image]]:
    return [(layer, render_layer(layer, scene.width, scene.height)) for layer in scene.layers]


def apply_opacity(img: Image.Image, opacity: float) -> Image.Image:
    if opacity >= 1:
        return img
    r, g, b, a = img.split()
    return Image.merge("RGBA", (r, g, b, a.point(lambda v: round(v * max(0.0, opacity)))))


def background_image(scene: Scene) -> Image.Image:
    return Image.new("RGBA", (scene.width, scene.height), parse_color(scene.background))


def composite(scene: Scene) -> Image.Image:
    out = from_image(background_image(scene))
    for layer in scene.layers:
        if layer.visible:
            out = composite_over(out, render_layer_array(layer, scene.width, scene.height), layer.blend,
                                 layer.opacity)
    return to_image(out)
