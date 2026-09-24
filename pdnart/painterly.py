"""Stroke-based repainting of a source image (after Hertzmann, "Painterly Rendering with Curved
Brush Strokes of Multiple Sizes", 1998).

The source is painted coarse-to-fine. For each brush size, the source is blurred in proportion to
the brush. Wherever the painting so far still differs from that blurred source, a stroke is
started. The stroke follows the isophotes (perpendicular to the brightness gradient), so strokes
wrap around forms the way a painter's do. It ends when the colour under it drifts too far from
its own colour. Large brushes lay in the masses and small ones only go where detail is needed.
"""

from __future__ import annotations

import colorsys
from typing import Any

import numpy as np

from .brushes import Paper, make_brush, stroke_masks
from .raster import gblur, paint


def _straight(premult: np.ndarray) -> np.ndarray:
    a = premult[..., 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a > 1e-4, premult[..., :3] / np.maximum(a, 1e-4), 0)


def _jitter(rgb, rng, amount: float):
    if not amount:
        return rgb
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb((h + rng.normal(0, amount * 0.15)) % 1, min(1, max(0, l + rng.normal(0, amount))),
                               min(1, max(0, s + rng.normal(0, amount))))


def repaint(layer: np.ndarray, source: np.ndarray, op: dict[str, Any], rng: np.random.Generator,
            paper: Paper, region: np.ndarray | None = None) -> None:
    """Paint `source` (premultiplied, canvas-sized) onto `layer` in place with brush strokes."""
    H, W = layer.shape[:2]
    sizes = sorted((float(s) for s in op.get("sizes", [24, 12, 6])), reverse=True)
    threshold = float(op.get("threshold", 0.06))
    min_len, max_len = op.get("length", [2, 10])
    curve = float(op.get("curvature", 0.75))
    fixed_angle = op.get("angle")
    color_jitter = float(op.get("color_jitter", 0.025))
    opacity = float(op.get("opacity", 1.0))
    grid_factor = float(op.get("grid", 1.0))
    blur_factor = float(op.get("blur", 0.5))
    overrides = {k: v for k, v in op.items() if k not in ("size", "opacity")}

    src_alpha = source[..., 3]
    paintable = src_alpha > 0.5
    if region is not None:
        paintable &= region > 0.5

    for D in sizes:
        R = D / 2
        ref_p = gblur(source, blur_factor * R) if blur_factor * R >= 0.3 else source
        ref = _straight(ref_p)
        lum = gblur(ref @ np.array([0.3, 0.59, 0.11], np.float32), max(1.0, R * 0.5))
        gy, gx = np.gradient(lum)
        gmag = np.hypot(gx, gy)

        # Compare at the scale of this brush: brush texture (bristle streaks, colour jitter) is not error.
        canvas = _straight(gblur(layer, 0.5 * R) if R >= 1 else layer)
        diff = np.abs(canvas - ref).mean(axis=-1)
        diff[layer[..., 3] < 0.5] = 1.0  # unpainted canvas always needs paint
        diff[~paintable] = 0.0

        # Error per grid cell (vectorised), and the worst pixel inside each cell as the stroke start.
        g = max(1, int(round(grid_factor * R)))
        hb, wb = H // g, W // g
        blocks = diff[:hb * g, :wb * g].reshape(hb, g, wb, g).transpose(0, 2, 1, 3).reshape(hb, wb, g * g)
        err = blocks.mean(-1)
        cells = np.argwhere(err > threshold)
        if not len(cells):
            continue
        worst = blocks[cells[:, 0], cells[:, 1]].argmax(-1)
        starts = np.stack([cells[:, 1] * g + worst % g, cells[:, 0] * g + worst // g], axis=1).astype(np.float64)
        rng.shuffle(starts)

        brush = make_brush(op.get("brush", "oil"), {**overrides, "size": D})
        if D < 8 and brush.bristles:  # bristle streaks are invisible at small sizes; plain dabs are cheaper
            brush = make_brush("round", {"size": D, "hardness": 0.7, "spacing": 0.15, "taper": brush.taper})

        for sx, sy in starts:
            ix, iy = int(sx), int(sy)
            color = ref[iy, ix]
            c0, c1, c2 = color.tolist()
            pts = [[sx, sy, 0.7]]
            x, y = sx, sy
            dx, dy = 0.0, 0.0
            for i in range(int(max_len)):
                cx, cy = int(x), int(y)
                if i > min_len:
                    r0, r1, r2 = ref[cy, cx].tolist()
                    if diff[cy, cx] * 3 < abs(r0 - c0) + abs(r1 - c1) + abs(r2 - c2):
                        break
                if fixed_angle is not None:
                    ndx, ndy = np.cos(np.radians(fixed_angle)), np.sin(np.radians(fixed_angle))
                else:
                    gm = float(gmag[cy, cx])
                    if gm < 1e-4:
                        if i == 0:
                            ang = rng.uniform(0, 2 * np.pi)
                            ndx, ndy = np.cos(ang), np.sin(ang)
                        else:
                            ndx, ndy = dx, dy
                    else:
                        ndx, ndy = -float(gy[cy, cx]) / gm, float(gx[cy, cx]) / gm
                if dx * ndx + dy * ndy < 0:
                    ndx, ndy = -ndx, -ndy
                if i > 0:
                    ndx, ndy = curve * ndx + (1 - curve) * dx, curve * ndy + (1 - curve) * dy
                norm = np.hypot(ndx, ndy) or 1
                dx, dy = ndx / norm, ndy / norm
                x, y = x + R * dx, y + R * dy
                if not (0 <= x < W and 0 <= y < H) or not paintable[int(y), int(x)]:
                    break
                pts.append([x, y, 1.0])
            if len(pts) > 1:
                pts[-1][2] = 0.6
            rgb = _jitter(tuple(float(c) for c in color), rng, color_jitter)
            for mask, x0, y0, rgba in stroke_masks(pts, brush, (*rgb, 1.0), rng, paper):
                paint(layer, mask, x0, y0, rgba, opacity=opacity)
