import os
import sys
from PIL import Image

# Output icon specs: (filename, size)
ICONS = [
    ('pwa-192x192.png', (192, 192)),
    ('pwa-512x512.png', (512, 512)),
    ('apple-touch-icon.png', (180, 180)),
    ('favicon-16x16.png', (16, 16)),
]

FAVICON_ICO = 'favicon.ico'
FAVICON_SIZES = [(16, 16), (32, 32), (48, 48), (64, 64)]


def generate_icons(source_icon_path):
    public_dir = os.path.join(os.path.dirname(__file__), '../frontend/public')
    if not os.path.exists(source_icon_path):
        print(f"Source icon not found: {source_icon_path}")
        return
    img = Image.open(source_icon_path).convert('RGBA')

    # Generate PNG icons
    for filename, size in ICONS:
        out_path = os.path.join(public_dir, filename)
        resized = img.resize(size, Image.Resampling.LANCZOS)
        resized.save(out_path, format='PNG')
        print(f"Wrote {out_path}")

    # Generate favicon.ico (multi-size)
    ico_path = os.path.join(public_dir, FAVICON_ICO)
    icons = [img.resize(size, Image.Resampling.LANCZOS) for size in FAVICON_SIZES]
    icons[0].save(ico_path, format='ICO', sizes=FAVICON_SIZES)
    print(f"Wrote {ico_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python generate_pwa_icons.py <source_image_path>")
        sys.exit(1)
    generate_icons(sys.argv[1]) 