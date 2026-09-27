"""Render the tobaskAplus mark (same geometry as logo.svg) to PNG with Pillow.

python make_logo_png.py  ->  logo-512.png, favicon.png (64 px)
"""
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
S = 16                                  # supersampling: draw at 64*S, then downscale
N = 64 * S
INK = (4, 22, 28, 255)


def gradient():
    img = Image.new("RGBA", (N, N))
    a, b = (0x00, 0xF2, 0xFE), (0x25, 0x63, 0xEB)
    px = img.load()
    for y in range(N):
        for x in range(N):
            t = (x + y) / (2 * (N - 1))
            px[x, y] = tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3)) + (255,)
    return img


def polyline(d, pts, width):
    """Round-capped, round-joined stroke: straight segments + a disc at every vertex."""
    pts = [(x * S, y * S) for x, y in pts]
    r = width * S / 2
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        d.line([(x0, y0), (x1, y1)], fill=INK, width=round(2 * r))
    for x, y in pts:
        d.ellipse((x - r, y - r, x + r, y + r), fill=INK)


def quad(p0, p1, p2, n=12):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in (i / n for i in range(n + 1))]


def mark():
    bg = gradient()
    mask = Image.new("L", (N, N), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, N - 1, N - 1), radius=18 * S, fill=255)
    out = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    out.paste(bg, (0, 0), mask)
    d = ImageDraw.Draw(out)
    polyline(d, [(16.5, 48)] + quad((30.2, 17.6), (32, 14.2), (33.8, 17.6)) + [(47.5, 48)], 6.5)
    polyline(d, [(22.8, 39.5), (28.6, 35.2), (33.2, 37.6), (41.2, 31.2)], 4.4)
    polyline(d, [(51, 8.5), (51, 19.5)], 3.6)
    polyline(d, [(45.5, 14), (56.5, 14)], 3.6)
    return out


if __name__ == "__main__":
    m = mark()
    m.resize((512, 512), Image.LANCZOS).save(HERE / "logo-512.png")
    m.resize((64, 64), Image.LANCZOS).save(HERE / "favicon.png")
    print("ok")
