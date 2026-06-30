"""Generate placeholder assets for OptiCrop."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
ASSETS = BASE / "assets"
CROP_DIR = ASSETS / "crop_images"

COLORS = {
    "rice": "#f4d03f", "maize": "#f39c12", "chickpea": "#d4ac0d",
    "kidneybeans": "#8b4513", "pigeonpeas": "#27ae60", "mothbeans": "#a0522d",
    "mungbean": "#2ecc71", "blackgram": "#1a1a2e", "lentil": "#e67e22",
    "pomegranate": "#c0392b", "banana": "#f1c40f", "mango": "#e74c3c",
    "grapes": "#8e44ad", "watermelon": "#2ecc71", "muskmelon": "#f5cba7",
    "apple": "#e74c3c", "orange": "#e67e22", "papaya": "#f9e79f",
    "coconut": "#795548", "cotton": "#ecf0f1", "jute": "#689f38", "coffee": "#4e342e",
}

CROPS = list(COLORS.keys())


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def make_background() -> None:
    img = Image.new("RGB", (1920, 600), (27, 67, 50))
    draw = ImageDraw.Draw(img)
    for y in range(600):
        r = int(27 + (y / 600) * 30)
        g = int(67 + (y / 600) * 40)
        b = int(50 + (y / 600) * 20)
        draw.line([(0, y), (1920, y)], fill=(r, g, b))
    for x in range(0, 1920, 120):
        draw.ellipse([x, 400, x + 80, 560], fill=(45, 106, 79))
    img.save(ASSETS / "background.jpg", quality=90)


def make_logo() -> None:
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 236, 236], fill=(45, 106, 79))
    draw.ellipse([60, 60, 196, 196], fill=(82, 183, 136))
    draw.text((88, 100), "OC", fill=(255, 255, 255))
    img.save(ASSETS / "logo.png")


def make_crop_images() -> None:
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    for crop in CROPS:
        color = hex_to_rgb(COLORS[crop])
        img = Image.new("RGB", (400, 300), color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 200, 400, 300], fill=(45, 106, 79))
        label = crop.replace("_", " ").title()
        draw.text((20, 130), label, fill=(255, 255, 255))
        draw.ellipse([150, 40, 250, 140], fill=(255, 255, 255, 128))
        img.save(CROP_DIR / f"{crop}.png")


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_background()
    make_logo()
    make_crop_images()
    print("Assets generated.")
