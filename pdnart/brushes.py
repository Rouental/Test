"""Dab-based brush engine.

A stroke is a path of points ([x, y] or [x, y, pressure]). It is resampled into
closely spaced "dabs" (small round stamps). Their size and alpha follow pressure,
taper and jitter. The dabs are stamped into a coverage mask, and paper grain,
bristles and wet edges then shape that mask into pencil, ink, oil, watercolor,
and so on.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from .raster import gblur


@dataclass(frozen=True)
class Brush:
    size: float = 12.0          # diameter in px at full pressure
    hardness: float = 0.8       # 0 = airbrush-soft edge, 1 = crisp edge
    flow: float = 1.0           # alpha of each dab
    opacity: float = 1.0        # maximum alpha of the whole stroke
    spacing: float = 0.12       # dab spacing as a fraction of size
    buildup: bool = False       # overlapping dabs accumulate (airbrush / watercolor)
    pressure_size: float = 1.0  # how much pressure scales size (0..1)
    pressure_opacity: float = 0.0
    taper: tuple[float, float] = (0.0, 0.0)  # fraction of stroke length that tapers at start / end
    jitter: float = 0.0         # random dab offset, fraction of size
    size_jitter: float = 0.0    # random dab size change, fraction
    wobble: float = 0.0         # slow hand wobble perpendicular to the path, px
    grain: float = 0.0          # paper-grain strength 0..1
    grain_scale: float = 1.0    # 1 = fine (pencil), 3+ = coarse (charcoal)
    bristles: int = 0           # >0 splits the brush into individual bristle tracks
    dryness: float = 0.0        # bristles run out of paint along the stroke
    color_variation: float = 0.0  # per-bristle / per-stroke lightness variation
    wet_edge: float = 0.0       # watercolor pigment pooling at edges
    smooth: bool = True         # pass a smooth curve through the points


PRESETS: dict[str, Brush] = {
    "round": Brush(),
    "soft": Brush(size=40, hardness=0.0, spacing=0.03, pressure_opacity=0.6),
    "airbrush": Brush(size=80, hardness=0.0, flow=0.05, buildup=True, spacing=0.06, pressure_opacity=0.8),
    "pencil": Brush(size=2.2, hardness=0.9, opacity=0.85, spacing=0.2, taper=(0.08, 0.12), wobble=0.4,
                    grain=0.7, grain_scale=1.0, pressure_size=0.4, pressure_opacity=0.7),
    "ink": Brush(size=4, hardness=0.95, spacing=0.1, taper=(0.15, 0.3), pressure_size=1.0),
    "charcoal": Brush(size=10, hardness=0.6, opacity=0.85, spacing=0.1, jitter=0.03, size_jitter=0.1,
                      grain=0.7, grain_scale=1.8, taper=(0.05, 0.1), pressure_opacity=0.5),
    "oil": Brush(size=18, hardness=0.9, spacing=0.1, bristles=20, dryness=0.4, color_variation=0.09,
                 taper=(0.03, 0.12), pressure_size=0.4),
    "dry": Brush(size=22, hardness=0.9, spacing=0.1, bristles=26, dryness=1.3, color_variation=0.05,
                 grain=0.3, grain_scale=1.5, opacity=0.9),
    "watercolor": Brush(size=36, hardness=0.4, flow=0.1, opacity=0.7, buildup=True, spacing=0.06,
                        jitter=0.03, size_jitter=0.08, wet_edge=0.8, grain=0.15, grain_scale=4.0),
    "marker": Brush(size=10, hardness=0.85, opacity=0.6, spacing=0.08),
    "glaze": Brush(size=50, hardness=0.2, opacity=0.25, spacing=0.04),
}

BRUSH_PARAMS = {f.name for f in fields(Brush)}


def make_brush(name: str | None, overrides: dict[str, Any]) -> Brush:
    base = PRESETS.get(name or "round")
    if base is None:
        raise ValueError(f"unknown brush {name!r}; presets are {sorted(PRESETS)}")
    kw = {k: v for k, v in overrides.items() if k in BRUSH_PARAMS}
    if "taper" in kw:
        t = kw["taper"]
        kw["taper"] = (float(t), float(t)) if isinstance(t, (int, float)) else (float(t[0]), float(t[1]))
    return replace(base, **kw)


class Paper:
    """Canvas-sized, repeatable paper texture used for grain."""

    def __init__(self, w: int, h: int, seed: int = 1234):
        self.w, self.h, self.seed = w, h, seed
        self._cache: dict[float, np.ndarray] = {}

    def texture(self, scale: float) -> np.ndarray:
        key = round(scale, 1)
        if key not in self._cache:
            rng = np.random.default_rng(self.seed + int(key * 10))
            t = gblur(rng.random((self.h, self.w), dtype=np.float32), 0.5 * key)
            lo, hi = np.percentile(t, [3, 97])
            self._cache[key] = np.clip((t - lo) / (hi - lo + 1e-6), 0, 1)
        return self._cache[key]


def _catmull_rom(p: np.ndarray, steps: int = 10) -> np.ndarray:
    if len(p) < 3:
        return p
    q = np.vstack([p[:1], p, p[-1:]])
    p0, p1, p2, p3 = (q[k:len(q) - 3 + k][:, None, :] for k in range(4))
    t = np.linspace(0, 1, steps, endpoint=False)[None, :, None]
    seg = 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
    return np.vstack([seg.reshape(-1, p.shape[1]), p[-1:]])


def _dabs(points: list, b: Brush, rng: np.random.Generator):
    """Resample a path into dab centers, radii, alphas and unit normals."""
    p = np.array([[*pt[:2], pt[2] if len(pt) > 2 else 1.0] for pt in points], np.float64)
    if b.smooth and len(p) >= 3:
        p = _catmull_rom(p)
    seg = np.hypot(*np.diff(p[:, :2], axis=0).T) if len(p) > 1 else np.zeros(0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    length = cum[-1]
    step = max(0.25, b.spacing * b.size)
    n = max(1, int(length / step) + 1)
    s = np.linspace(0, length, n)
    x = np.interp(s, cum, p[:, 0])
    y = np.interp(s, cum, p[:, 1])
    pr = np.clip(np.interp(s, cum, p[:, 2]), 0, 1)
    t = s / length if length > 0 else np.zeros(n)

    # Direction and normals along the path.
    if n > 1:
        dx, dy = np.gradient(x), np.gradient(y)
        norm = np.hypot(dx, dy) + 1e-9
        nx, ny = -dy / norm, dx / norm
    else:
        nx, ny = np.zeros(n), np.ones(n)

    taper = np.ones(n)
    ts, te = b.taper
    if ts > 0:
        taper = np.minimum(taper, np.clip(t / ts, 0, 1) ** 0.7)
    if te > 0:
        taper = np.minimum(taper, np.clip((1 - t) / te, 0, 1) ** 0.7)
    taper = np.maximum(taper, 0.08)

    r = b.size / 2 * (1 - b.pressure_size + b.pressure_size * pr) * taper
    if b.size_jitter:
        r *= 1 + rng.uniform(-b.size_jitter, b.size_jitter, n)
    a = b.flow * (1 - b.pressure_opacity + b.pressure_opacity * pr)

    if b.wobble and length > 0:
        phase = rng.uniform(0, 2 * np.pi, 3)
        wavelength = max(40.0, length)
        w = sum(np.sin(2 * np.pi * s * k / wavelength + ph) / k for k, ph in zip((1, 2.3, 5.1), phase))
        x, y = x + nx * w * b.wobble, y + ny * w * b.wobble
    if b.jitter:
        x = x + rng.normal(0, b.jitter * b.size, n)
        y = y + rng.normal(0, b.jitter * b.size, n)
    return x, y, np.maximum(r, 0.05), np.broadcast_to(a, (n,)).astype(np.float64), nx, ny, t


PROFILE_BINS = 128


def _stamp(x, y, r, a, hardness: float, buildup: bool, pad: int = 2, profiles=None, nx=None, ny=None):
    """Stamp dabs into a new coverage mask. Returns (mask, x0, y0).

    With `profiles` (a list of (n, PROFILE_BINS) arrays) and normals, each dab is also multiplied by a
    cross-section profile sampled across the stroke, which is how bristle streaks are drawn. One mask
    is produced per profile, sharing the geometry work; the result is then (masks, x0, y0).
    """
    # Very small dabs keep a minimum footprint and lose alpha instead, which keeps thin lines smooth.
    small = r < 0.6
    a = np.where(small, a * (r / 0.6) ** 2, a)
    r = np.maximum(r, 0.6)
    x0 = int(np.floor((x - r).min())) - pad
    y0 = int(np.floor((y - r).min())) - pad
    x1 = int(np.ceil((x + r).max())) + pad
    y1 = int(np.ceil((y + r).max())) + pad
    W, H = x1 - x0, y1 - y0
    accs = [np.zeros(H * W, np.float32) for _ in (profiles or [None])]

    K = int(np.ceil(r.max() * 2)) + 3
    offs = np.arange(K) - K // 2
    x, y, r, a = (np.asarray(v, np.float32) for v in (x, y, r, a))
    chunk = max(1, 2_000_000 // (K * K))
    for i in range(0, len(x), chunk):
        cx, cy, cr, ca = x[i:i + chunk], y[i:i + chunk], r[i:i + chunk], a[i:i + chunk]
        px = np.floor(cx).astype(np.int64)[:, None] + offs  # (n, K)
        py = np.floor(cy).astype(np.int64)[:, None] + offs
        dx = (px + 0.5 - cx[:, None]).astype(np.float32)
        dy = (py + 0.5 - cy[:, None]).astype(np.float32)
        dist = np.sqrt(dy[:, :, None] ** 2 + dx[:, None, :] ** 2)  # (n, Ky, Kx)
        rr = cr[:, None, None]
        v = np.minimum(np.maximum(rr - dist + 0.5, 0), 1)  # anti-aliased disc
        if hardness < 1:
            u = np.minimum(np.maximum((dist / rr - hardness) * (1 / (1 - hardness + 1e-6)), 0), 1)
            v *= 1 - u * u * (3 - 2 * u)
        v *= ca[:, None, None]
        if profiles is not None:
            u = (dx[:, None, :] * nx[i:i + chunk, None, None] + dy[:, :, None] * ny[i:i + chunk, None, None]) / rr
            bins = ((u + 1) * (0.5 * (PROFILE_BINS - 1)) + 0.5).astype(np.int64)
            np.clip(bins, 0, PROFILE_BINS - 1, out=bins)
            bins = bins.reshape(len(cx), -1)
            layers = [v * np.take_along_axis(p[i:i + chunk], bins, 1).reshape(v.shape) for p in profiles]
        else:
            layers = [v]
        idx_all = (py - y0)[:, :, None] * W + (px - x0)[:, None, :]
        for acc, vv in zip(accs, layers):
            keep = vv > 1e-4
            if buildup:
                np.add.at(acc, idx_all[keep], -np.log1p(-np.minimum(vv[keep], 0.999)))
            else:
                np.maximum.at(acc, idx_all[keep], vv[keep])
    masks = [(1 - np.exp(-acc) if buildup else acc).reshape(H, W) for acc in accs]
    return (masks if profiles is not None else masks[0]), x0, y0


def _shift_lightness(rgba, amount: float):
    h, l, s = colorsys.rgb_to_hls(*rgba[:3])
    return (*colorsys.hls_to_rgb(h, min(1, max(0, l + amount)), s), rgba[3])


def _texture(mask, x0, y0, b: Brush, paper: Paper | None):
    if b.wet_edge > 0:  # pigment pools at the edge of a wash
        flat = np.clip(mask * 2.2, 0, 1)  # the wash settles into a flat puddle...
        edge = np.clip(flat - gblur(flat, max(1.0, b.size * 0.1)), 0, 1)  # ...darker where it dries
        mask = np.clip(flat * (1 - 0.45 * b.wet_edge) + edge * 1.8 * b.wet_edge, 0, 1)
    if b.grain > 0 and paper is not None:
        tex = paper.texture(b.grain_scale)
        h, w = mask.shape
        ys = slice(max(0, y0), max(0, min(paper.h, y0 + h)))
        xs = slice(max(0, x0), max(0, min(paper.w, x0 + w)))
        g = np.ones_like(mask)
        g[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = tex[ys, xs]
        # Grain bites harder where the stroke is lighter, as graphite skips over paper tooth.
        mask = mask * np.clip(1 - b.grain + b.grain * (g * 1.6 - 0.3 + mask * 0.3), 0, 1)
    return mask * b.opacity


def stroke_masks(points: list, b: Brush, color, rng: np.random.Generator, paper: Paper | None = None):
    """Render one stroke. Returns a list of (mask, x0, y0, rgba) pieces to composite in order."""
    color = tuple(color)
    if b.color_variation and not b.bristles:
        color = _shift_lightness(color, rng.uniform(-b.color_variation, b.color_variation))
    x, y, r, a, nx, ny, t = _dabs(points, b, rng)

    if not b.bristles:
        mask, x0, y0 = _stamp(x, y, r, a, b.hardness, b.buildup)
        return [(_texture(mask, x0, y0, b, paper), x0, y0, color)]

    # Bristle brush: a semi-opaque body, then streaks. Every dab carries a cross-section profile made
    # of the bristles (thin Gaussian ridges at random offsets). Bristles are split between three tints,
    # and each runs dry at its own rate and breaks up where its own smooth noise is high, the way a
    # loaded brush leaves streaks and a dry one skips.
    body = max(0.0, 0.8 - 0.6 * b.dryness)
    m = b.bristles
    offsets = rng.uniform(-0.95, 0.95, m)
    width = rng.uniform(0.6, 1.4, m) * 0.9 / m
    strength = rng.uniform(0.5, 1.0, m)
    dry_rate = rng.uniform(0.3, 1.0, m) * b.dryness
    length_px = float(np.hypot(np.diff(x), np.diff(y)).sum()) if len(x) > 1 else 1.0
    u = np.linspace(-1, 1, PROFILE_BINS)
    tints = [color, _shift_lightness(color, b.color_variation), _shift_lightness(color, -b.color_variation)]
    cycles = rng.uniform(1, 4, (m, 3)) * max(1.0, length_px / 60)
    phases = rng.uniform(0, 6.3, (m, 3))
    noise = np.sin(2 * np.pi * cycles[:, :, None] * t[None, None, :] + phases[:, :, None]).sum(1)
    noise = (noise / 3 + 1) / 2  # (m, n): 0..1, smooth along the stroke
    load = np.clip(1 - dry_rate[:, None] * t[None, :], 0, 1)
    vis = (np.clip((load - noise * min(1.0, b.dryness) * 0.7) * 4, 0, 1) * strength[:, None]).astype(np.float32)
    ridge = np.exp(-0.5 * ((u[None, :] - offsets[:, None]) / width[:, None]) ** 2).astype(np.float32)
    profiles = [(vis[g::3, :, None] * ridge[g::3, None, :]).max(0) for g in range(3)]
    colors = tints
    if body > 0:  # the body: a slightly narrower, semi-opaque base under the streaks
        profiles.insert(0, np.broadcast_to(np.where(np.abs(u) < 0.94, body, 0).astype(np.float32),
                                           (len(t), PROFILE_BINS)))
        colors = [color, *tints]
    masks, x0, y0 = _stamp(x, y, r, a, max(b.hardness, 0.6), b.buildup, profiles=profiles, nx=nx, ny=ny)
    return [(_texture(mk, x0, y0, b, paper), x0, y0, c) for mk, c in zip(masks, colors)]
