"""Form shading: light a 3D surface described as a silhouette plus bumps and hollows.

Painters model a head, an apple or a fold of cloth by thinking of it as a solid lit from one
direction: the planes that face the light are bright, the planes that turn away go into
shadow with a soft edge (the terminator), cavities collect occlusion, and shadowed planes
catch a little reflected light. The `form` op does exactly that:

1. the silhouette is "inflated" into a rounded dome,
2. bumps (positive h) and hollows (negative h) are added as smooth ellipsoidal mounds,
3. surface normals are lit with a wrapped Lambert term, ambient occlusion, bounce light and
   an optional specular highlight,
4. the resulting value (0 = deepest shadow .. 1 = full light) is mapped through a colour ramp.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .raster import gblur, gblur_exact

DEFAULT_RAMP = [[0, "#1e1a1a"], [0.5, "#7a7070"], [1, "#f2eeea"]]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) or 1)


def edge_distance(outline: np.ndarray, x0: int, y0: int, h: int, w: int, step: int = 2) -> np.ndarray:
    """Exact distance (px) from each pixel centre to the closed outline, computed on a coarse grid
    (every `step` px) and interpolated back up."""
    a = outline
    b = np.roll(outline, -1, axis=0)
    ab = b - a
    ab2 = np.maximum((ab * ab).sum(1), 1e-9)
    gys = np.arange(0, h + step, step, dtype=np.float32) + y0 + 0.5
    gxs = np.arange(0, w + step, step, dtype=np.float32) + x0 + 0.5
    P = np.stack(np.meshgrid(gxs, gys), -1).reshape(-1, 2)
    best = np.full(len(P), np.inf, np.float32)
    for i in range(0, len(a), 64):  # chunk over segments to bound memory
        sa, sab, sab2 = a[i:i + 64], ab[i:i + 64], ab2[i:i + 64]
        ap = P[:, None, :] - sa[None]
        t = np.clip((ap * sab[None]).sum(-1) / sab2[None], 0, 1)
        d = ap - t[..., None] * sab[None]
        best = np.minimum(best, np.sqrt((d * d).sum(-1)).min(1))
    coarse = best.reshape(len(gys), len(gxs))
    # Bilinear upsample to full resolution.
    fy = (np.arange(h) / step)
    fx = (np.arange(w) / step)
    iy, ix = np.floor(fy).astype(int), np.floor(fx).astype(int)
    ty, tx = (fy - iy)[:, None], (fx - ix)[None, :]
    c00 = coarse[iy][:, ix]
    c01 = coarse[iy][:, ix + 1]
    c10 = coarse[iy + 1][:, ix]
    c11 = coarse[iy + 1][:, ix + 1]
    return ((c00 * (1 - tx) + c01 * tx) * (1 - ty) + (c10 * (1 - tx) + c11 * tx) * ty).astype(np.float32)


def height_field(mask: np.ndarray, x0: int, y0: int, op: dict[str, Any], outline) -> np.ndarray:
    """Height (in px) over the mask's rectangle, whose top-left canvas pixel is (x0, y0).

    The silhouette is rounded over with a quarter-circle profile of radius `inflate`: vertical at
    the outline, flat once `inflate` px inside it.
    """
    h, w = mask.shape
    inflate = float(op.get("inflate", 0.22 * min(h, w)))
    # Unsigned distance, so zero it outside the shape (where anti-aliased edge pixels would form a ridge).
    d = edge_distance(np.asarray(outline, np.float32), x0, y0, h, w) * (mask >= 0.5)
    t = np.clip(d / max(inflate, 1e-3), 0, 1)
    height = inflate * np.sqrt(1 - (1 - t) ** 2)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs += x0 + 0.5
    ys += y0 + 0.5
    for bump in op.get("bumps", []):
        a = math.radians(bump.get("angle", 0))
        dx, dy = xs - bump["x"], ys - bump["y"]
        u = (dx * math.cos(a) + dy * math.sin(a)) / bump["rx"]
        v = (-dx * math.sin(a) + dy * math.cos(a)) / bump["ry"]
        d2 = u * u + v * v
        if bump.get("profile") == "sphere":
            shape = np.sqrt(np.clip(1 - d2, 0, 1))
        else:
            shape = np.exp(-2.0 * d2)
        height += float(bump["h"]) * shape
    return height


def shade(mask: np.ndarray, x0: int, y0: int, op: dict[str, Any], ramp_colors, outline) -> np.ndarray:
    """Return straight RGBA (h, w, 4) float colours for the lit form (alpha = 1)."""
    height = height_field(mask, x0, y0, op, outline)
    gy, gx = np.gradient(gblur_exact(height, 0.8))
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    light = _unit(op.get("light", [-0.55, -0.65, 0.55]))
    wrap = float(op.get("wrap", 0.35))
    lam = n @ light.astype(np.float32)
    lit = np.clip((lam + wrap) / (1 + wrap), 0, 1)

    # Ambient occlusion: how far a point sits below its surroundings.
    occlusion = float(op.get("occlusion", 0.6))
    if occlusion:
        scale = float(op.get("inflate", 0.22 * min(mask.shape))) * 0.08 + 1
        cavity = np.clip(gblur_exact(height, max(2.0, scale * 3)) - height, 0, None)
        lit = lit / (1 + occlusion * cavity / scale)

    value = np.clip(float(op.get("ambient", 0.08)) + (1 - float(op.get("ambient", 0.08))) * lit, 0, 1)
    offs = [float(o) for o, _ in ramp_colors]
    rgb = np.stack([np.interp(value, offs, [c[i] for _, c in ramp_colors]) for i in range(3)], axis=-1)

    # Reflected light: planes turned away from the key light pick up colour bounced from below/behind.
    if op.get("bounce"):
        bcol, strength = op["bounce"]
        bdir = _unit([-light[0], 0.7, 0.25])
        amount = float(strength) * np.clip(n @ bdir.astype(np.float32), 0, 1) * (1 - lit)
        rgb = rgb + (np.asarray(bcol[:3], np.float32) - rgb) * amount[..., None]

    if op.get("shine"):
        half = _unit(light + np.array([0, 0, 1.0]))
        spec = np.clip(n @ half.astype(np.float32), 0, 1) ** float(op.get("shine_size", 40))
        rgb = rgb + (1 - rgb) * (float(op["shine"]) * spec)[..., None]

    out = np.empty(mask.shape + (4,), np.float32)
    out[..., :3] = np.clip(rgb, 0, 1)
    out[..., 3] = 1
    return out
