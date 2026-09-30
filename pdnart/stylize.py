"""Turn a photo (or other art) into a semi-realistic digital painting.

This is a classical image-processing pipeline, not a trained model:

1. Denoise (bilateral), then an *anisotropic Kuwahara filter* (Kyprianidis et al. 2009). Each pixel
   takes the colour of the most uniform sector of an ellipse aligned with the local edge direction.
   Texture flattens into painted planes that follow the forms, while edges stay crisp.
2. *Focus*: faces are found (OpenCV, when installed) or given as a box. They are painted with a small
   filter radius so eyes, lips and jewellery stay sharp, and the background is abstracted more and
   softened, like the lost edges of a painted backdrop.
3. *Colour*: optionally match the palette of one or more style reference paintings (Reinhard colour
   transfer in Lab), then add warm light / cool shadow toning, saturation and gentle local contrast.
4. *Line accents*: an XDoG edge layer on multiply darkens contours the way an illustrator
   crisps the eyes, jaw and folds.
5. *Brush texture*: noise smeared along the stroke-flow field (line integral convolution), strongest
   away from the face.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .raster import gblur

try:  # optional: better denoising, face detection
    import cv2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only without OpenCV
    cv2 = None


@dataclass
class StyleOptions:
    max_size: int = 1400             # working resolution (long side, px)
    min_size: int = 1000             # small images are upscaled to this first: cleaner, finer strokes
    face_radius: float = 4.0         # Kuwahara radius on faces, at 1200 px
    background_radius: float = 10.0  # Kuwahara radius elsewhere, at 1200 px
    sharpness: float = 8.0           # Kuwahara q: higher = crisper plane boundaries
    background_softness: float = 1.2  # extra blur of the background (px at 1200)
    color_strength: float = 0.55     # how far to move toward the reference palette (0..1)
    saturation: float = 1.12
    warm_cool: float = 0.35          # warm highlights / cool shadows toning
    clarity: float = 0.4             # local contrast
    key_light: float = 0.6           # brighten the lit side of faces
    lines: float = 0.4               # opacity of the line-accent layer
    skin_smoothing: float = 0.7      # idealise skin: soften wrinkles and pores, keep features sharp
    skin_warmth: float = 0.6         # saturated warm transition on skin shadows (subsurface glow)
    form_contrast: float = 0.45      # dodge & burn the big light/shadow shapes of the subject
    detail_match: float = 0.7        # match the references' amount of detail at each scale (needs a profile)
    background: str = "paint"        # "keep", "paint" (softer, hazier, more painted) or an image path
    background_strength: float = 0.55
    rim_light: float = 0.25          # soft light along the subject's shadow-side edge
    highlights: float = 1.0          # restore small bright points: stars, glints, catchlights
    brushwork: float = 0.7           # visible brush strokes outside the face (pdnart painterly pass)
    texture: float = 0.04            # brush texture strength
    focus: tuple[int, int, int, int] | None = None  # (x, y, w, h) in input pixels; None = detect faces
    references: Sequence[str] = field(default_factory=tuple)
    palette: tuple[Sequence[float], Sequence[float]] | None = None  # (Lab mean, Lab std), e.g. from a profile
    bands: Sequence[float] | None = None  # the references' detail per scale (see detail_bands)


# ------------------------------------------------------------------ colour helpers

def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
_WHITE = np.array([0.95047, 1.0, 1.08883], np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    xyz = _srgb_to_linear(rgb) @ _M.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
    return _linear_to_srgb(xyz @ np.linalg.inv(_M).T.astype(np.float32))


def palette_stats(paths: Sequence[str | Path]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of each Lab channel over the reference paintings."""
    labs = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((400, 400))
            labs.append(rgb_to_lab(np.asarray(im, np.float32) / 255).reshape(-1, 3))
    lab = np.concatenate(labs)
    return lab.mean(0), lab.std(0)


def transfer_palette(rgb: np.ndarray, ref_mean, ref_std, strength: float) -> np.ndarray:
    """Borrow the references' colour *character* (contrast, vibrancy, overall warmth) without
    repainting the scene in their hues: a night scene stays blue, but gets their saturation."""
    lab = rgb_to_lab(rgb)
    flat = lab.reshape(-1, 3)
    mean, std = flat.mean(0), flat.std(0) + 1e-3
    out = lab.copy()
    # lightness: move toward the references' contrast, and a little toward their key
    out[..., 0] = (lab[..., 0] - mean[0]) * (1 + strength * (ref_std[0] / std[0] - 1)) + mean[0] \
        + 0.3 * strength * (ref_mean[0] - mean[0])
    # colour: scale chroma around the image's own mean to the references' spread; nudge the mean
    chroma_scale = float(np.clip(np.hypot(*ref_std[1:]) / np.hypot(*std[1:]), 0.7, 1.6))
    out[..., 1:] = (lab[..., 1:] - mean[1:]) * (1 + strength * (chroma_scale - 1)) + mean[1:] \
        + 0.25 * strength * (ref_mean[1:] - mean[1:])
    return lab_to_rgb(out)


# ------------------------------------------------------------------ analysis

def detect_faces(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    if cv2 is None or not hasattr(cv2, "CascadeClassifier"):
        return []
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    min_side = max(40, min(gray.shape) // 12)
    faces = cascade.detectMultiScale(gray, 1.1, 6, minSize=(min_side, min_side))
    return [tuple(int(v) for v in f) for f in faces]


def focus_mask(h: int, w: int, boxes: Sequence[tuple[int, int, int, int]]) -> np.ndarray:
    """Soft mask: 1 over each face (widened to take in hair and neck), fading out smoothly."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.zeros((h, w), np.float32)
    for x, y, bw, bh in boxes:
        cx, cy = x + bw / 2, y + bh * 0.55
        d = ((xs - cx) / (bw * 0.85)) ** 2 + ((ys - cy) / (bh * 0.95)) ** 2
        mask = np.maximum(mask, np.clip(1.6 - d, 0, 1))
    return mask


def structure(lum: np.ndarray, sigma: float):
    """Edge tangent direction and anisotropy from the smoothed structure tensor."""
    gy, gx = np.gradient(gblur(lum, 1.0))
    jxx, jxy, jyy = (gblur(v, sigma) for v in (gx * gx, gx * gy, gy * gy))
    tmp = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    l1, l2 = (jxx + jyy + tmp) / 2, (jxx + jyy - tmp) / 2
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2  # along the edge
    aniso = np.where(l1 + l2 > 1e-9, (l1 - l2) / (l1 + l2 + 1e-9), 0)
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32), aniso.astype(np.float32)


# ------------------------------------------------------------------ the filter

def anisotropic_kuwahara(rgb: np.ndarray, radius: np.ndarray, tx, ty, aniso, sectors: int = 8,
                         q: float = 8.0, grid: int = 11) -> np.ndarray:
    """Anisotropic Kuwahara filter with a per-pixel radius.

    For every pixel an ellipse (long axis along the edge, elongated by the anisotropy) is split into
    `sectors` overlapping sectors. Each sector's weighted mean and variance are gathered, and the
    output favours the sectors with the lowest variance, i.e. the side of the edge the pixel is on.
    """
    h, w, _ = rgb.shape
    a = radius * (1 + aniso)          # semi-axis along the edge
    b = radius / (1 + aniso)          # semi-axis across it
    xs0 = np.arange(w, dtype=np.float32)[None, :]
    ys0 = np.arange(h, dtype=np.float32)[:, None]
    sums = np.zeros((sectors, h, w, 3), np.float32)
    sq = np.zeros((sectors, h, w), np.float32)
    wts = np.zeros((sectors, 1, 1), np.float32)
    centers = 2 * np.pi * np.arange(sectors) / sectors
    flat = rgb.reshape(-1, 3)
    for gv in np.linspace(-1, 1, grid):
        for gu in np.linspace(-1, 1, grid):
            rho2 = gu * gu + gv * gv
            if rho2 > 1:
                continue
            g = np.exp(-rho2 / (2 * 0.45 ** 2))
            if rho2 < 1e-9:
                sw = np.full(sectors, g / sectors)
            else:
                d = np.angle(np.exp(1j * (np.arctan2(gv, gu) - centers)))
                sw = g * np.maximum(0, np.cos(d * sectors / 4)) ** 2
            if not sw.any():
                continue
            dx = gu * a * tx - gv * b * ty
            dy = gu * a * ty + gv * b * tx
            px = np.clip(np.rint(xs0 + dx), 0, w - 1).astype(np.int32)
            py = np.clip(np.rint(ys0 + dy), 0, h - 1).astype(np.int32)
            c = flat[py * w + px]
            c2 = (c * c).sum(-1)
            for k in np.nonzero(sw > 1e-4)[0]:
                sums[k] += sw[k] * c
                sq[k] += sw[k] * c2
                wts[k] += sw[k]
    wts = np.maximum(wts, 1e-6)
    means = sums / wts[..., None]
    var = np.maximum(sq / wts - (means * means).sum(-1), 0)
    alpha = 1 / (1 + (np.sqrt(var) * 255) ** (q / 2))
    return (alpha[..., None] * means).sum(0) / np.maximum(alpha.sum(0), 1e-12)[..., None]


# ------------------------------------------------------------------ finishing layers

def xdog_lines(lum: np.ndarray, sigma: float, k: float = 1.6, p: float = 20.0, eps: float = 0.02,
               phi: float = 60.0) -> np.ndarray:
    """Coverage (0..1) of dark lines along contours (extended difference of Gaussians)."""
    g1, g2 = gblur(lum, sigma), gblur(lum, sigma * k)
    d = (1 + p) * g1 - p * g2
    edge = np.where(d >= eps, 1.0, 1 + np.tanh(phi * (d - eps)))
    return np.clip(1 - edge, 0, 1).astype(np.float32)


def brush_texture(h: int, w: int, tx, ty, length: int, rng: np.random.Generator) -> np.ndarray:
    """Noise smeared along the flow field (line integral convolution): streaks like brushwork."""
    noise = gblur(rng.random((h, w), dtype=np.float32), 0.7)
    acc = noise.copy()
    for sign in (1, -1):
        x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
        for _ in range(length):
            ix = np.clip(x.astype(np.int32), 0, w - 1)
            iy = np.clip(y.astype(np.int32), 0, h - 1)
            x += sign * tx[iy, ix]
            y += sign * ty[iy, ix]
            acc += noise[np.clip(y.astype(np.int32), 0, h - 1), np.clip(x.astype(np.int32), 0, w - 1)]
    acc /= 2 * length + 1
    return (acc - acc.mean()) / (acc.std() + 1e-6)


def smooth_skin(rgb: np.ndarray, focus: np.ndarray, amount: float, face_size: float) -> np.ndarray:
    """Frequency separation, as a retoucher would: flatten the mid-frequency texture (wrinkles, pores)
    in the focus area but keep detail where edges are strong (eyes, brows, lips, nostrils).
    Scales are relative to the face, so it works the same on a close-up and a full figure."""
    unit = face_size / 300
    sigma = max(1.5, 6.0 * unit)
    if cv2 is not None:  # edge-aware, so the background never bleeds into the face outline
        low = cv2.bilateralFilter(rgb.astype(np.float32), 0, 0.12, sigma)
    else:
        low = gblur(rgb, sigma)
    detail = rgb - low
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    gy, gx = np.gradient(gblur(lum, 1.0 * unit))
    edge = gblur(np.hypot(gx, gy), 1.5 * unit)
    strong = np.clip((edge - np.percentile(edge, 80)) / (np.percentile(edge, 97) - np.percentile(edge, 80) + 1e-6),
                     0, 1)
    keep = 1 - amount * focus * (1 - strong)
    return low + detail * keep[..., None]


def add_brushwork(rgb: np.ndarray, focus: np.ndarray, amount: float, unit: float, seed: int) -> np.ndarray:
    """Repaint the picture with pdnart's painterly strokes and blend them in away from the face."""
    from .brushes import Paper
    from .painterly import repaint

    h, w, _ = rgb.shape
    src = np.concatenate([rgb, np.ones((h, w, 1), np.float32)], -1)
    layer = np.zeros_like(src)
    sizes = [round(s * unit, 1) for s in (28, 14, 7)]
    repaint(layer, src, {"brush": "oil", "sizes": sizes, "threshold": 0.05, "length": [2, 8],
                         "curvature": 0.8, "color_jitter": 0.008, "color_variation": 0.04},
            np.random.default_rng(seed), Paper(w, h, seed))
    a = layer[..., 3:4]
    strokes = np.where(a > 1e-4, layer[..., :3] / np.maximum(a, 1e-4), rgb)
    mix = (amount * (1 - 0.85 * focus) * a[..., 0])[..., None]
    return rgb + (strokes - rgb) * mix


BAND_SIGMAS = (1.0, 2.0, 4.0, 8.0)  # at 1200 px long side


def detail_bands(L: np.ndarray, unit: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Standard deviation of lightness in each band of a difference-of-Gaussians pyramid:
    how much fine vs broad detail a picture has."""
    out, prev = [], L
    for s in BAND_SIGMAS:
        blurred = gblur(L, s * unit)
        band = prev - blurred
        out.append(float(band[mask > 0.5].std() if mask is not None else band.std()))
        prev = blurred
    return np.asarray(out, np.float32)


def match_bands(L: np.ndarray, ref_bands, unit: float, strength: float) -> np.ndarray:
    """Rescale each detail band of L toward the references' amount of detail at that scale."""
    have = detail_bands(L, unit)
    prev, rebuilt, bands = L, None, []
    for s in BAND_SIGMAS:
        blurred = gblur(L if not bands else prev, s * unit)
        bands.append(prev - blurred)
        prev = blurred
    rebuilt = prev
    for band, h_std, r_std in zip(bands, have, ref_bands):
        gain = float(np.clip((r_std / max(h_std, 1e-3)) ** 0.5, 0.7, 1.5))
        rebuilt = rebuilt + band * (1 + strength * (gain - 1))
    return rebuilt


def subject_mask(rgb: np.ndarray, faces, unit: float) -> np.ndarray | None:
    """Separate the person from the background (GrabCut seeded from the face), feathered."""
    if cv2 is None or not faces:
        return None
    h, w, _ = rgb.shape
    small = min(1.0, 600 / max(h, w))
    sw, sh = max(1, round(w * small)), max(1, round(h * small))
    img = cv2.resize((rgb * 255).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA)
    mask = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    for x, y, fw, fh in faces:
        x, y, fw, fh = (v * small for v in (x, y, fw, fh))
        # beside the head, above the shoulders: probably background, the far corners certainly
        top = int(y + 0.9 * fh)
        cv2.rectangle(mask, (0, 0), (int(x - 0.7 * fw), top), cv2.GC_BGD, -1)
        cv2.rectangle(mask, (int(x + 1.7 * fw), 0), (sw, top), cv2.GC_BGD, -1)
        # probably the person: the head, then shoulders widening down to the bottom edge
        body = np.array([[x - 0.1 * fw, y - 0.55 * fh], [x + 1.1 * fw, y - 0.55 * fh], [x + 1.15 * fw, y + 1.0 * fh],
                         [x + 1.9 * fw, y + 1.5 * fh], [x + 2.6 * fw, sh], [x - 1.6 * fw, sh],
                         [x - 0.9 * fw, y + 1.5 * fh], [x - 0.15 * fw, y + 1.0 * fh]], np.int32)
        cv2.fillPoly(mask, [body], cv2.GC_PR_FGD)
        # certainly the person: the middle of the face and the neck below it
        cv2.rectangle(mask, (int(x + 0.2 * fw), int(y + 0.15 * fh)), (int(x + 0.8 * fw), int(y + 1.25 * fh)),
                      cv2.GC_FGD, -1)
        cv2.rectangle(mask, (int(x + 0.05 * fw), int(y + 1.25 * fh)), (int(x + 0.95 * fw), sh), cv2.GC_FGD, -1)
    border = max(2, round(0.01 * max(sw, sh)))
    mask[:border, :] = np.where(mask[:border, :] == cv2.GC_FGD, cv2.GC_FGD, cv2.GC_BGD)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    # keep the main connected shape, and fill holes inside it (buttons, badges, dark folds)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        fg = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    holes = (1 - fg).copy()
    flood = np.zeros((sh + 2, sw + 2), np.uint8)
    for sx, sy in ((0, 0), (sw - 1, 0), (0, sh - 1), (sw - 1, sh - 1)):
        if holes[sy, sx]:
            cv2.floodFill(holes, flood, (sx, sy), 0)
    fg = np.maximum(fg, holes).astype(np.float32)  # what the flood from the edges didn't reach is a hole
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(gblur(fg, 4 * unit), 0, 1)


def cover_fit(path: str, w: int, h: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = max(w / im.width, h / im.height)
        im = im.resize((max(w, round(im.width * scale)), max(h, round(im.height * scale))), Image.LANCZOS)
        left, top = (im.width - w) // 2, (im.height - h) // 3
        return np.asarray(im.crop((left, top, left + w, top + h)), np.float32) / 255


# ------------------------------------------------------------------ pipeline

@dataclass
class Stylized:
    painting: Image.Image       # the finished picture
    base: Image.Image           # painted colour layer
    lines: Image.Image          # line accents, meant for a multiply layer
    faces: list[tuple[int, int, int, int]]


def stylize(image: Image.Image, opts: StyleOptions | None = None, seed: int = 1) -> Stylized:
    opts = opts or StyleOptions()
    img = image.convert("RGB")
    scale_in = min(1.0, opts.max_size / max(img.size))
    if max(img.size) < opts.min_size:
        scale_in = opts.min_size / max(img.size)
    if scale_in != 1:
        img = img.resize((round(img.width * scale_in), round(img.height * scale_in)), Image.LANCZOS)
    rgb = np.asarray(img, np.float32) / 255
    h, w, _ = rgb.shape
    unit = max(h, w) / 1200  # parameters are specified for a 1200 px image

    # 1. denoise, so the filter paints the form and not the photo grain
    if cv2 is not None:
        den = cv2.bilateralFilter(rgb, 0, 0.06, 2.5 * unit)
    else:
        den = gblur(rgb, 0.8 * unit)

    # 2. focus: faces stay detailed, the background gets abstracted and softened
    if opts.focus:
        faces = [tuple(round(v * scale_in) for v in opts.focus)]
    else:
        faces = detect_faces(rgb)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:3]
        if faces:  # drop small false positives next to a clear main face
            faces = [f for f in faces if f[2] * f[3] >= 0.25 * faces[0][2] * faces[0][3]]
    focus = focus_mask(h, w, faces) if faces else np.zeros((h, w), np.float32)
    face_size = float(np.sqrt(faces[0][2] * faces[0][3])) if faces else 0.25 * max(h, w)
    subject = subject_mask(rgb, faces, unit) if (opts.background != "keep" or opts.form_contrast
                                                  or opts.rim_light) else None
    if subject is not None and opts.background not in ("keep", "paint"):
        # a new setting behind the person, painted by the same filter so the two belong together
        backdrop = cover_fit(opts.background, w, h)
        backdrop = cv2.bilateralFilter(backdrop, 0, 0.06, 2.5 * unit) if cv2 is not None else backdrop
        den = backdrop + (den - backdrop) * subject[..., None]
        rgb = backdrop + (rgb - backdrop) * subject[..., None]
        focus = focus * subject

    lum = den @ np.array([0.299, 0.587, 0.114], np.float32)
    tx, ty, aniso = structure(lum, 2.0 * unit)
    # Busy, detailed regions (architecture, ornament, a crowd of small shapes) keep more of their
    # detail too; big smooth areas get painted broadly.
    gy, gx = np.gradient(lum)
    busy = gblur(np.hypot(gx, gy), 10 * unit)
    busy = np.clip(busy / (np.percentile(busy, 95) + 1e-6), 0, 1)
    detail = np.maximum(focus, 0.75 * busy)
    radius = (opts.background_radius + (opts.face_radius - opts.background_radius) * detail) * unit
    painted = anisotropic_kuwahara(den, radius.astype(np.float32), tx, ty, aniso, q=opts.sharpness)
    if opts.skin_smoothing and faces:
        painted = smooth_skin(painted, focus, opts.skin_smoothing, float(np.sqrt(faces[0][2] * faces[0][3])))
    if opts.background_softness:
        soft = gblur(painted, opts.background_softness * unit)
        painted = soft + (painted - soft) * detail[..., None]

    if subject is not None and opts.background != "keep" and opts.background_strength:
        # the background recedes: softer, lower in contrast and a little hazier, like a painted backdrop
        bg = 1 - subject
        soft = gblur(painted, 2.5 * unit)
        mean = (soft * bg[..., None]).sum((0, 1)) / max(bg.sum(), 1)
        hazy = soft + (mean - soft) * 0.2
        hazy = hazy + (1 - hazy) * 0.05
        painted = painted + (hazy - painted) * (opts.background_strength * bg)[..., None]

    # 3. colour
    palette = opts.palette or (palette_stats(opts.references) if opts.references else None)
    if palette is not None:
        mean, std = (np.asarray(v, np.float32) for v in palette)
        painted = transfer_palette(painted, mean, std, opts.color_strength)
    lab = rgb_to_lab(np.clip(painted, 0, 1))
    L = lab[..., 0]
    if opts.clarity:  # local contrast against an edge-preserving base, so dark/light edges don't halo
        base = cv2.bilateralFilter(L.astype(np.float32), 0, 12, 10 * unit) if cv2 is not None else gblur(L, 10 * unit)
        L = L + opts.clarity * (L - base)
    if opts.form_contrast and subject is not None:
        # dodge & burn at the scale of the big forms (cheek planes, the shadow side of the face and
        # body), which painters exaggerate far more than a camera records
        big = gblur(L, 0.13 * face_size) - gblur(L, 0.5 * face_size)  # planes, not creases
        L = L + opts.form_contrast * 0.9 * big * subject
    if opts.detail_match and opts.bands is not None:
        L = match_bands(L, np.asarray(opts.bands, np.float32), unit, opts.detail_match)
    if opts.key_light and faces:  # lift the lit side of faces toward a brighter, painted high key
        L = L + opts.key_light * 10 * focus * np.clip((L - 35) / 40, 0, 1)
    lab[..., 0] = L
    lab[..., 1:] *= opts.saturation
    if opts.skin_warmth:
        # Skin-like hues (warm, moderately saturated) get richer and redder in the half-tones, where
        # light scatters under the skin: the red ears, nose and cheek edges of a painted portrait.
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
        skin = np.clip(1 - np.abs(hue - 55) / 40, 0, 1) * np.clip((chroma - 8) / 12, 0, 1)
        halftone = np.exp(-((L - 52) / 18) ** 2)
        k = opts.skin_warmth * skin * halftone
        lab[..., 1] += 14 * k
        lab[..., 2] += 4 * k
    if opts.warm_cool:  # warm the lights (toward orange), cool the shadows (toward blue)
        t = np.clip((L - 50) / 50, -1, 1) * opts.warm_cool
        lab[..., 1] += 6 * t
        lab[..., 2] += 12 * t
    painted = np.clip(lab_to_rgb(lab), 0, 1)

    if opts.rim_light and subject is not None and faces:
        # which side of the face is lit? the rim goes on the other, shadow side
        x, y, fw, fh = faces[0]
        fl = rgb[y:y + fh, x:x + fw] @ np.array([0.299, 0.587, 0.114], np.float32)
        lit_left = fl[:, : fw // 2].mean() > fl[:, fw // 2:].mean()
        inner = subject * np.clip(1 - gblur(subject, 5 * unit) * 1.15, 0, 1) * 3
        side = np.clip((np.arange(w, dtype=np.float32)[None, :] - (x + fw / 2)) / (1.5 * fw), -1, 1)
        side = np.clip(side if lit_left else -side, 0, 1)
        rim = np.clip(gblur(inner * side, 2 * unit), 0, 1) * opts.rim_light
        rim_color = np.array([1.0, 0.93, 0.82], np.float32)
        painted = painted + (rim_color - painted) * rim[..., None]
    if opts.brushwork:
        painted = add_brushwork(painted, detail, opts.brushwork, unit, seed)
    if opts.highlights:  # small bright points (stars, glints, catchlights) the filter flattened
        src_lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        spark = np.clip(src_lum - gblur(src_lum, 2.5 * unit) - 0.06, 0, 1) * opts.highlights * 2.5
        painted = np.clip(painted + (rgb - painted) * np.clip(spark, 0, 1)[..., None], 0, 1)

    # 5. brush texture, mostly away from the face
    if opts.texture:
        # Texture follows a wide, smooth flow: on plain areas (a wall, sky) the local flow is just noise,
        # which would turn the streaks into speckle.
        wtx, wty, _ = structure(gblur(lum, 3 * unit), 14 * unit)
        tex = brush_texture(h, w, wtx, wty, max(6, round(12 * unit)), np.random.default_rng(seed))
        amount = opts.texture * (1 - 0.7 * focus)
        painted = np.clip(painted * (1 + amount[..., None] * tex[..., None]), 0, 1)

    # 4. line accents (crisper on the face)
    plum = painted @ np.array([0.299, 0.587, 0.114], np.float32)
    lines = np.maximum(xdog_lines(plum, 1.0 * unit) * 0.55,
                       xdog_lines(plum, 0.7 * unit, eps=0.01) * focus)  # finer, crisper accents on faces
    line_rgba = np.zeros((h, w, 4), np.float32)
    line_rgba[..., :3] = np.array([0.22, 0.13, 0.10], np.float32)  # warm dark brown, not black
    line_rgba[..., 3] = lines * opts.lines
    final = painted * (1 - line_rgba[..., 3:4]) + line_rgba[..., :3] * line_rgba[..., 3:4]

    to_img = lambda a, mode="RGB": Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8), mode)  # noqa: E731
    inv = 1 / scale_in
    return Stylized(to_img(final), to_img(painted), to_img(line_rgba, "RGBA"),
                    [tuple(round(v * inv) for v in f) for f in faces])


# ------------------------------------------------------------------ style profiles

def make_profile(references: Sequence[str | Path], **option_overrides) -> dict:
    """A reusable style profile: the references' palette statistics plus any option overrides."""
    mean, std = palette_stats(references)
    bands = []
    for r in references:
        with Image.open(r) as im:
            im = im.convert("RGB")
            im.thumbnail((1200, 1200))
            L = rgb_to_lab(np.asarray(im, np.float32) / 255)[..., 0]
            bands.append(detail_bands(L, max(L.shape) / 1200))
    known = {f for f in StyleOptions.__dataclass_fields__} - {"references", "palette", "focus", "bands"}
    bad = set(option_overrides) - known
    if bad:
        raise ValueError(f"unknown style options: {sorted(bad)}")
    return {"lab_mean": [round(float(v), 3) for v in mean], "lab_std": [round(float(v), 3) for v in std],
            "detail_bands": [round(float(v), 3) for v in np.mean(bands, axis=0)],
            "references": [Path(r).name for r in references], "options": option_overrides}


def options_from_profile(profile: dict, **overrides) -> StyleOptions:
    opts = {**profile.get("options", {}), **{k: v for k, v in overrides.items() if v is not None}}
    return StyleOptions(palette=(profile["lab_mean"], profile["lab_std"]), bands=profile.get("detail_bands"), **opts)
