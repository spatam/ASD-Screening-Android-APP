"""Generate the README artwork: banner and pipeline (light and dark) and the social preview.

Text is converted to outlines with HarfBuzz and the Inter typeface, so every viewer sees the
same glyphs and every pill is sized from the real width of its label.

    pip install uharfbuzz fonttools
    python docs/assets/generate.py            # downloads Inter 4.1 (SIL OFL) on first use

Set ASD_FONT_DIR to use Inter TTF files you already have.

The social preview PNG needs rsvg-convert (librsvg).
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

OUT = Path(__file__).resolve().parent
FONT_DIR = Path(os.environ.get('ASD_FONT_DIR', Path.home() / '.cache' / 'asd-screening-fonts'))
INTER_URL = 'https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip'


def ensure_fonts() -> None:
    if (FONT_DIR / 'Inter-Regular.ttf').exists():
        return
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(INTER_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for name in archive.namelist():
        if name.startswith('extras/ttf/') and name.endswith('.ttf'):
            (FONT_DIR / Path(name).name).write_bytes(archive.read(name))


def num(value: float) -> str:
    text = f'{value:.2f}'.rstrip('0').rstrip('.')
    return '0' if text == '-0' else text


class Font:
    def __init__(self, filename: str, key: str) -> None:
        path = str(FONT_DIR / filename)
        self.key = key
        self.hb_font = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
        tt = TTFont(path)
        self.upem = tt['head'].unitsPerEm
        self.cap = tt['OS/2'].sCapHeight / self.upem
        self.glyphs = tt.getGlyphSet()
        self.order = tt.getGlyphOrder()

    def _shape(self, text: str, size: float, tracking: float):
        """Glyphs as (glyph id, x, y) in font units, and the advance width in pixels."""
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf, {'kern': True, 'liga': True})
        track = tracking * self.upem / size
        placed, x = [], 0.0
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            placed.append((info.codepoint, x + pos.x_offset, pos.y_offset))
            x += pos.x_advance + track
        width_units = x - track if placed else 0.0
        return placed, width_units * size / self.upem

    def width(self, text: str, size: float, tracking: float = 0.0) -> float:
        return self._shape(text, size, tracking)[1]

    def glyph_path(self, gid: int) -> str:
        pen = SVGPathPen(self.glyphs, ntos=lambda v: str(round(v)))
        self.glyphs[self.order[gid]].draw(pen)
        return pen.getCommands()

    def path(self, text: str, x: float, baseline: float, size: float, fill: str,
             anchor: str = 'start', tracking: float = 0.0) -> str:
        placed, width = self._shape(text, size, tracking)
        x -= {'start': 0.0, 'middle': width / 2, 'end': width}[anchor]
        uses = []
        for gid, gx, gy in placed:
            ref = f'{self.key}{gid}'
            if ref not in GLYPHS:
                GLYPHS[ref] = self.glyph_path(gid)
            if GLYPHS[ref]:
                y_attr = f' y="{round(gy)}"' if round(gy) else ''
                uses.append(f'<use href="#{ref}" x="{round(gx)}"{y_attr}/>')
        k = size / self.upem
        return (f'<g fill="{fill}" transform="translate({num(x)} {num(baseline)}) scale({k:.6f} {-k:.6f})">'
                f'{"".join(uses)}</g>')

    def baseline_for(self, center_y: float, size: float) -> float:
        """Baseline that centres the capital letters on ``center_y``."""
        return center_y + self.cap * size / 2


GLYPHS: dict[str, str] = {}


def glyph_defs() -> str:
    """Every glyph used so far, once, for the <use> references of the current document."""
    return ''.join(f'<path id="{ref}" d="{d}"/>' for ref, d in GLYPHS.items() if d)


@dataclass(frozen=True)
class Theme:
    bg: str
    bg2: str
    title: str
    text: str
    muted: str
    accent: str
    accent_soft: str
    accent_text: str
    line: str
    card: str
    card_line: str
    phone: str
    phone_line: str
    lane: str
    arrow: str


THEMES = {
    'light': Theme(bg='#ffffff', bg2='#fff7f3', title='#0f172a', text='#475569', muted='#64748b',
                   accent='#e8552e', accent_soft='#fff0ea', accent_text='#c2410c', line='#e2e8f0',
                   card='#ffffff', card_line='#e2e8f0', phone='#1e293b', phone_line='#334155',
                   lane='#f8fafc', arrow='#94a3b8'),
    'dark': Theme(bg='#0d1117', bg2='#1a1411', title='#f0f6fc', text='#b4bdc8', muted='#8b949e',
                  accent='#f0643c', accent_soft='#2a1712', accent_text='#ff8a65', line='#30363d',
                  card='#161b22', card_line='#30363d', phone='#0b0f14', phone_line='#3d444d',
                  lane='#161b22', arrow='#6e7681'),
}

ensure_fonts()
REGULAR = Font('Inter-Regular.ttf', 'r')
MEDIUM = Font('Inter-Medium.ttf', 'm')
SEMIBOLD = Font('Inter-SemiBold.ttf', 's')
BOLD = Font('Inter-Bold.ttf', 'b')
DISPLAY = Font('InterDisplay-ExtraBold.ttf', 'd')


def pill(x: float, cy: float, label: str, t: Theme, size: float = 15, pad: float = 16,
         height: float = 34, dot: bool = False, filled: bool = True) -> tuple[str, float]:
    """Rounded label with equal padding on both sides; returns (svg, width)."""
    dot_w = 10 + 9 if dot else 0
    width = pad + dot_w + SEMIBOLD.width(label, size) + pad
    fill, stroke, ink = (t.accent_soft, 'none', t.accent_text) if filled else (t.card, t.card_line, t.title)
    parts = [f'<rect x="{num(x)}" y="{num(cy - height / 2)}" width="{num(width)}" height="{num(height)}" '
             f'rx="{num(height / 2 if filled else 9)}" fill="{fill}" stroke="{stroke}"/>']
    if dot:
        parts.append(f'<circle cx="{num(x + pad + 5)}" cy="{num(cy)}" r="5" fill="{t.accent}"/>')
    parts.append(SEMIBOLD.path(label, x + pad + dot_w, SEMIBOLD.baseline_for(cy, size), size, ink))
    return ''.join(parts), width


def scene(x: float, y: float, w: float, h: float) -> str:
    """Flat illustrated stimulus with a scanpath and a heatmap blob, clipped to the screen."""
    sx = lambda v: x + v * w  # noqa: E731
    sy = lambda v: y + v * h  # noqa: E731
    fix = [(0.30, 0.42), (0.52, 0.35), (0.58, 0.55), (0.40, 0.62), (0.70, 0.40)]
    path = ' '.join(f'{"M" if i == 0 else "L"}{num(sx(a))},{num(sy(b))}' for i, (a, b) in enumerate(fix))
    dots = ''.join(
        f'<circle cx="{num(sx(a))}" cy="{num(sy(b))}" r="{5.5 if i == len(fix) - 1 else 4}" '
        f'fill="#22c55e" stroke="#052e16" stroke-width="1.2"/>' for i, (a, b) in enumerate(fix))
    return f'''<clipPath id="screen"><rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" rx="6"/></clipPath>
  <g clip-path="url(#screen)">
    <rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" fill="#bfdbfe"/>
    <rect x="{num(x)}" y="{num(sy(0.68))}" width="{num(w)}" height="{num(h * 0.32)}" fill="#86c77a"/>
    <circle cx="{num(sx(0.84))}" cy="{num(sy(0.2))}" r="{num(h * 0.1)}" fill="#fcd34d"/>
    <rect x="{num(sx(0.12))}" y="{num(sy(0.42))}" width="{num(w * 0.04)}" height="{num(h * 0.3)}" fill="#8b5e3c"/>
    <circle cx="{num(sx(0.14))}" cy="{num(sy(0.38))}" r="{num(h * 0.14)}" fill="#3f8f4f"/>
    <rect x="{num(sx(0.45))}" y="{num(sy(0.36))}" width="{num(w * 0.2)}" height="{num(h * 0.34)}" rx="3" fill="#f8fafc"/>
    <polygon points="{num(sx(0.43))},{num(sy(0.37))} {num(sx(0.55))},{num(sy(0.2))} {num(sx(0.67))},{num(sy(0.37))}" fill="#dc2626"/>
    <rect x="{num(sx(0.52))}" y="{num(sy(0.52))}" width="{num(w * 0.06)}" height="{num(h * 0.18)}" fill="#7c2d12"/>
    <ellipse cx="{num(sx(0.5))}" cy="{num(sy(0.47))}" rx="{num(w * 0.2)}" ry="{num(h * 0.2)}" fill="url(#heat)"/>
    <path d="{path}" fill="none" stroke="#facc15" stroke-width="1.8" stroke-linejoin="round"/>
    {dots}
  </g>'''


def gradients(t: Theme, glow: float = 0.22) -> str:
    return f'''<defs>
    <radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="{t.accent}" stop-opacity="{glow}"/>
      <stop offset="1" stop-color="{t.accent}" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="heat" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#ef4444" stop-opacity="0.75"/>
      <stop offset="0.45" stop-color="#f59e0b" stop-opacity="0.45"/>
      <stop offset="1" stop-color="#22d3ee" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0.45" stop-color="{t.bg}"/>
      <stop offset="1" stop-color="{t.bg2}"/>
    </linearGradient>
  </defs>'''


def phone(px: float, py: float, pw: float, ph: float, t: Theme, radius: float = 30) -> str:
    return f'''<ellipse cx="{num(px + pw / 2)}" cy="{num(py + ph / 2)}" rx="{num(pw * 0.87)}" ry="{num(ph * 0.83)}" fill="url(#glow)"/>
  <rect x="{num(px)}" y="{num(py)}" width="{num(pw)}" height="{num(ph)}" rx="{num(radius)}" fill="{t.phone}" stroke="{t.phone_line}" stroke-width="2"/>
  <circle cx="{num(px + 14)}" cy="{num(py + ph / 2)}" r="3.5" fill="#475569"/>
  <rect x="{num(px + 30)}" y="{num(py + 12)}" width="{num(pw - 52)}" height="{num(ph - 24)}" rx="8" fill="#111827"/>
  {scene(px + 62, py + 16, pw - 116, ph - 32)}
  <rect x="{num(px + 30)}" y="{num(py + 12)}" width="{num((pw - 52) * 0.62)}" height="3" fill="{t.accent}"/>'''


def banner(t: Theme) -> str:
    GLYPHS.clear()
    W, H, x0 = 1280, 360, 72
    parts = [f'<rect width="{W}" height="{H}" rx="18" fill="url(#bg)"/>',
             f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="18" fill="none" stroke="{t.line}"/>']
    badge, _ = pill(x0, 73, 'IEEE OJCS 2026 · open source', t, dot=True)
    parts.append(badge)
    parts.append(DISPLAY.path('ASD Screening', x0 - 3, 158, 68, t.title, tracking=-0.6))
    parts.append(REGULAR.path("Autism screening from a child's gaze on single pictures,", x0, 205, 22.5, t.text))
    parts.append(REGULAR.path('distilled to run offline on an ordinary Android phone.', x0, 237, 22.5, t.text))
    x = x0
    for label in ['1.6M-parameter student', '5-fold ONNX ensemble', 'No network access']:
        chip, width = pill(x, 287, label, t, height=36, filled=False)
        parts.append(chip)
        x += width + 12

    px, py, pw, ph = 868, 72, 344, 206
    parts.append(phone(px, py, pw, ph, t))
    # Session card: equal padding on every side, value and caption on one baseline.
    pad, label, value, caption = 16, 'session score', '0.58', '40 / 300 pictures'
    row = BOLD.width(value, 22) + 10 + REGULAR.width(caption, 12.5)
    cw = pad + max(MEDIUM.width(label, 12.5), row) + pad
    ch = pad + 12.5 * MEDIUM.cap + 12 + 22 * BOLD.cap + pad
    cx, cy = px - 64, py + ph - 40
    top = cy + pad
    parts.append(f'<rect x="{num(cx)}" y="{num(cy)}" width="{num(cw)}" height="{num(ch)}" rx="12" fill="{t.card}" stroke="{t.card_line}"/>')
    parts.append(MEDIUM.path(label, cx + pad, top + 12.5 * MEDIUM.cap, 12.5, t.muted))
    value_base = top + 12.5 * MEDIUM.cap + 12 + 22 * BOLD.cap
    parts.append(BOLD.path(value, cx + pad, value_base, 22, t.title))
    parts.append(REGULAR.path(caption, cx + pad + BOLD.width(value, 22) + 10, value_base, 12.5, t.muted))
    body = '\n  '.join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
            f'aria-labelledby="t">\n  <title id="t">ASD Screening: autism screening from a child\'s gaze on single '
            f'pictures, distilled to run offline on an ordinary Android phone</title>\n  {gradients(t)}\n  '
            f'<defs>{glyph_defs()}</defs>\n  {body}\n</svg>\n')


def pipeline(t: Theme) -> str:
    GLYPHS.clear()
    W, H = 1240, 576
    bw, bh, gap, left = 262, 132, 36, 42
    xs = [left + i * (bw + gap) for i in range(4)]
    y1, y2 = 96, 392
    lanes = [
        (y1, 'TRAINING · PYTHON · ASD_GAZE', [
            ('Saliency4ASD', ['28 children, 300 pictures', 'Tobii T120, 3 s per picture', '14 ASD and 14 TD']),
            ('Teacher · Phases 1-2', ['frozen ViT-B/16', '+ gaze transformer', '154K trainable parameters']),
            ('Student · Phase 3', ['MobileNetV3-Small', 'KL + BCE distillation', '1.6M parameters']),
            ('ONNX · Phase 4', ['5 folds, opset 18', '6.4 MB per fold', 'max error 6e-7 vs PyTorch']),
        ], 3),
        (y2, 'ON THE PHONE · ANDROID · OFFLINE', [
            ('Camera to gaze', ['CameraX + ML Kit Face Mesh', 'pupil centres', '9-point calibration']),
            ('Fixations', ['I-DT detector', 'stimulus pixel coordinates', 'one window per picture']),
            ('5-fold ensemble', ['gaze sequence 25 × 3', 'picture + heatmap 3 × 224²', 'mean of 5 logits']),
            ('Session score', ['sigmoid of the mean logit', 'study threshold 0.492', 'verdict after 40 pictures']),
        ], 2),
    ]
    parts = [f'<rect width="{W}" height="{H}" rx="18" fill="{t.bg}"/>',
             f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="18" fill="none" stroke="{t.line}"/>']
    for y, lane_label, boxes, highlight in lanes:
        parts.append(f'<rect x="20" y="{y - 58}" width="{W - 40}" height="{bh + 84}" rx="16" fill="{t.lane}" stroke="{t.line}"/>')
        parts.append(SEMIBOLD.path(lane_label, left, y - 22, 13.5, t.accent_text, tracking=1.4))
        for i, (title, lines) in enumerate(boxes):
            x = xs[i]
            for text, font, size in [(title, BOLD, 18)] + [(line, REGULAR, 15) for line in lines]:
                assert font.width(text, size) <= bw - 36, f'{text!r} does not fit its box'
            stroke, sw = (t.accent, 2) if i == highlight else (t.card_line, 1.2)
            parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="14" fill="{t.card}" stroke="{stroke}" stroke-width="{sw}"/>')
            # Block from the cap top of the title to the last baseline, centred in the box.
            block = 18 * BOLD.cap + 28 + 22 * (len(lines) - 1)
            title_base = y + (bh - block) / 2 + 18 * BOLD.cap
            parts.append(BOLD.path(title, x + 18, title_base, 18, t.title))
            for k, line in enumerate(lines):
                parts.append(REGULAR.path(line, x + 18, title_base + 28 + k * 22, 15, t.text))
            if i < 3:
                ay, ax1, ax2 = y + bh / 2, x + bw + 5, xs[i + 1] - 5
                parts.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2 - 7}" y2="{ay}" stroke="{t.arrow}" stroke-width="2"/>')
                parts.append(f'<path d="M{ax2 - 9},{ay - 6} L{ax2},{ay} L{ax2 - 9},{ay + 6}" fill="none" stroke="{t.arrow}" stroke-width="2"/>')
    # The ONNX files of Phase 4 ship inside the app and feed the ensemble.
    ox, ex = xs[3] + bw / 2, xs[2] + bw / 2
    ymid = ((y1 - 58 + bh + 84) + (y2 - 58)) / 2
    parts.append(f'<path d="M{ox},{y1 + bh + 4} L{ox},{ymid} L{ex},{ymid} L{ex},{y2 - 8}" fill="none" stroke="{t.accent}" stroke-width="2.2" stroke-dasharray="7 6"/>')
    parts.append(f'<path d="M{ex - 7},{y2 - 16} L{ex},{y2 - 6} L{ex + 7},{y2 - 16}" fill="none" stroke="{t.accent}" stroke-width="2.2"/>')
    label = 'ships in the app'
    lw = SEMIBOLD.width(label, 15) + 2 * 18
    parts.append(f'<rect x="{num((ox + ex) / 2 - lw / 2)}" y="{num(ymid - 17)}" width="{num(lw)}" height="34" rx="17" fill="{t.accent_soft}"/>')
    parts.append(SEMIBOLD.path(label, (ox + ex) / 2, SEMIBOLD.baseline_for(ymid, 15), 15, t.accent_text, anchor='middle'))
    body = '\n  '.join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
            f'aria-labelledby="t">\n  <title id="t">Training in four phases on Saliency4ASD, then on-device inference: '
            f'camera to gaze, fixations, five-fold ensemble, session score</title>\n  <defs>{glyph_defs()}</defs>\n  {body}\n</svg>\n')


def social(t: Theme) -> str:
    GLYPHS.clear()
    W, H, x0 = 1280, 640, 72
    parts = [f'<rect width="{W}" height="{H}" fill="{t.bg}"/>']
    badge, _ = pill(x0, 104, 'IEEE OJCS 2026 · open source', t, size=17, height=38, pad=18, dot=True)
    parts.append(badge)
    parts.append(DISPLAY.path('ASD Screening', x0 - 4, 222, 86, t.title, tracking=-0.8))
    for k, line in enumerate(["Autism screening from a child's gaze on", 'single pictures, running offline on an',
                              'ordinary Android phone.']):
        parts.append(REGULAR.path(line, x0, 280 + k * 36, 27, t.text))
    parts.append(f'<line x1="{x0}" y1="452" x2="720" y2="452" stroke="{t.line}" stroke-width="2"/>')
    stats = [('0.959', 'per-child AUC'), ('40', 'pictures in 3 min'), ('1.6M', 'parameters'), ('0', 'network permissions')]
    widths = [max(DISPLAY.width(v, 40), REGULAR.width(label, 17)) for v, label in stats]
    gap = (720 - x0 - sum(widths)) / (len(stats) - 1)
    x = x0
    for (value, label), w in zip(stats, widths):
        parts.append(DISPLAY.path(value, x, 530, 40, t.accent_text))
        parts.append(REGULAR.path(label, x, 562, 17, t.muted))
        x += w + gap
    parts.append(phone(770, 150, 430, 258, t, radius=36))
    body = '\n  '.join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
            f'  {gradients(t, glow=0.25)}\n  <defs>{glyph_defs()}</defs>\n  {body}\n</svg>\n')


def main() -> None:
    for name, theme in THEMES.items():
        (OUT / f'banner-{name}.svg').write_text(banner(theme))
        (OUT / f'pipeline-{name}.svg').write_text(pipeline(theme))
    if shutil.which('rsvg-convert'):
        svg = OUT / 'social-preview.svg'
        svg.write_text(social(THEMES['dark']))
        subprocess.run(['rsvg-convert', '-w', '1280', '-h', '640', str(svg), '-o', str(OUT / 'social-preview.png')], check=True)
        svg.unlink()
    print(f'artwork written to {OUT}')


if __name__ == '__main__':
    main()
