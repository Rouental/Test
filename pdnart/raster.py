"""Float raster helpers: layers are (H, W, 4) float32 arrays of premultiplied RGBA in 0..1."""

from __future__ import annotations

import numpy as np
from PIL import Image

BLEND_MODES = ("normal", "multiply", "screen", "overlay", "darken", "lighten", "additive", "difference")


def new_layer(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 4), np.float32)


def _box_blur_axis(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    if r <= 0:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def gblur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Approximate Gaussian blur (three box passes) over the first two axes."""
    if sigma < 0.3:
        return a
    # Box radius giving the requested sigma after three passes.
    r = max(1, int(round((np.sqrt(4 * sigma * sigma + 1) - 1) / 2)))
    for _ in range(3):
        a = _box_blur_axis(_box_blur_axis(a, r, 0), r, 1)
    return a


def paint(layer: np.ndarray, mask: np.ndarray, x0: int, y0: int, color, *, opacity: float = 1.0,
          lock_alpha: bool = False, erase: bool = False) -> None:
    """Composite `color` through coverage `mask` (placed at x0, y0) onto `layer` in place.

    `color` is straight RGBA in 0..1, either a 4-tuple or an array shaped like mask + (4,).
    """
    h, w = mask.shape
    H, W = layer.shape[:2]
    # Clip the mask rectangle to the layer.
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + w), min(H, y0 + h)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    m = mask[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    col = np.asarray(color, np.float32)
    if col.ndim == 3:
        col = col[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    dst = layer[cy0:cy1, cx0:cx1]
    # Long thin strokes cover a small part of their bounding box: only touch covered pixels.
    sel = m > 0
    if sel.mean() < 0.5:
        m, d = m[sel], dst[sel]
        c = col[sel] if col.ndim == 3 else col
        _composite(d, m, c, opacity, lock_alpha, erase)
        dst[sel] = d
    else:
        _composite(dst, m, col, opacity, lock_alpha, erase)


def _composite(dst, m, col, opacity, lock_alpha, erase):
    a = (m * col[..., 3] * opacity)[..., None]
    if erase:
        dst *= 1 - a
        return
    rgb = col[..., :3]
    if lock_alpha:  # source-atop: recolor existing pixels, keep their alpha
        da = dst[..., 3:4].copy()
        dst[..., :3] = rgb * a * da + dst[..., :3] * (1 - a)
    else:  # source-over
        dst[..., :3] = rgb * a + dst[..., :3] * (1 - a)
        dst[..., 3:4] = a + dst[..., 3:4] * (1 - a)


def _blend(mode: str, cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """Separable blend functions on straight colors (W3C compositing spec)."""
    if mode == "multiply":
        return cb * cs
    if mode == "screen":
        return cb + cs - cb * cs
    if mode == "overlay":
        return np.where(cb <= 0.5, 2 * cb * cs, 1 - 2 * (1 - cb) * (1 - cs))
    if mode == "darken":
        return np.minimum(cb, cs)
    if mode == "lighten":
        return np.maximum(cb, cs)
    if mode == "additive":
        return np.minimum(1, cb + cs)
    if mode == "difference":
        return np.abs(cb - cs)
    return cs


def composite_over(dst: np.ndarray, src: np.ndarray, mode: str = "normal", opacity: float = 1.0) -> np.ndarray:
    """Composite premultiplied `src` onto premultiplied `dst` with a blend mode; returns a new array."""
    src = src * opacity if opacity < 1 else src
    sa, da = src[..., 3:4], dst[..., 3:4]
    if mode == "normal":
        return src + dst * (1 - sa)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = np.where(sa > 0, src[..., :3] / sa, 0)
        cb = np.where(da > 0, dst[..., :3] / da, 0)
    mixed = (1 - da) * src[..., :3] + (1 - sa) * dst[..., :3] + sa * da * _blend(mode, cb, cs)
    return np.concatenate([mixed, sa + da * (1 - sa)], axis=-1).astype(np.float32)


def to_image(layer: np.ndarray) -> Image.Image:
    """Premultiplied float layer -> straight-alpha RGBA Pillow image."""
    a = layer[..., 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        rgb = np.where(a > 1e-6, layer[..., :3] / a, 0)
    out = np.concatenate([rgb, a], axis=-1)
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")


def from_image(img: Image.Image) -> np.ndarray:
    """Straight-alpha Pillow image -> premultiplied float layer."""
    arr = np.asarray(img.convert("RGBA"), np.float32) / 255
    arr[..., :3] *= arr[..., 3:4]
    return arr


def gblur_exact(a: np.ndarray, sigma: float) -> np.ndarray:
    """True (isotropic) Gaussian blur of a 2D array via FFT, with zero padding outside.

    Slower than gblur but free of the box-filter anisotropy, which matters when the result is
    differentiated (surface normals for lighting).
    """
    if sigma < 0.3:
        return a
    pad = int(3 * sigma) + 1
    h, w = a.shape
    H, W = h + 2 * pad, w + 2 * pad
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    kernel = np.exp(-2 * (np.pi * sigma) ** 2 * (fx * fx + fy * fy))
    padded = np.zeros((H, W), np.float64)
    padded[pad:pad + h, pad:pad + w] = a
    out = np.fft.irfft2(np.fft.rfft2(padded) * kernel, s=(H, W))
    return out[pad:pad + h, pad:pad + w].astype(np.float32)
