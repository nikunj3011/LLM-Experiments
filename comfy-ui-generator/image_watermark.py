import os
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def add_watermark(
    image_path,
    text,
    output_path,
    font_path=None,
    font_size=36,
    opacity=255,
    bg_opacity=100,
    bg_padding=10,
):
    """Adds watermark and strips all standard and non-standard metadata."""
    # 1. Open original image
    with Image.open(image_path) as raw_img:
        # Convert to RGBA to handle transparency
        src_img = raw_img.convert("RGBA")

        # STRIP METADATA: Create a completely fresh RGBA canvas.
        # Pixel data is copied, but exifs, XMP, and custom metadata are dropped.
        base_img = Image.new("RGBA", src_img.size)
        base_img.paste(src_img, (0, 0))

    # 2. Load font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        print("Warning: Custom font not found. Falling back to default font.")
        font = ImageFont.load_default()

    # 3. Create transparent layer for watermark
    overlay_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_layer)

    # 4. Calculate dimensions and positioning
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    margin = 20
    x = base_img.width - text_width - margin - bg_padding
    y = base_img.height - text_height - margin - bg_padding

    # 5. Draw background box
    if bg_opacity > 0:
        box_coords = [
            x - bg_padding,
            y - bg_padding,
            x + text_width + bg_padding,
            y + text_height + bg_padding,
        ]
        bg_color = (0, 0, 0, bg_opacity)
        draw.rounded_rectangle(box_coords, radius=8, fill=bg_color)

    # 6. Draw text
    text_color = (255, 255, 255, opacity)
    draw.text((x - bbox[0], y - bbox[1]), text, font=font, fill=text_color)

    # 7. Composite overlay onto clean canvas
    watermarked = Image.alpha_composite(base_img, overlay_layer)

    # 8. Convert to RGB and save WITHOUT passing any exif/metadata objects
    final_img = watermarked.convert("RGB")

    # Parameters optimize image saving without attaching metadata
    final_img.save(
        output_path,
        format="JPEG",
        quality=95,
        optimize=True,
    )


def process_folder(
    input_folder,
    output_folder,
    text,
    font_path=None,
    font_size=36,
    opacity=200,
    bg_opacity=80,
):
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_extensions]

    # Sort files by modification date from newest to oldest
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    total_files = len(files)
    if total_files == 0:
        print("No valid images found in input folder.")
        return

    print(f"Starting watermark & metadata removal process for {total_files} images...\n")

    start_time = time.perf_counter()

    for index, file in enumerate(files, start=1):
        # Force JPG output or retain original filename
        output_file = output_dir / f"watermarked_{file.stem}.jpg"
        print(f"[{index}/{total_files}] Processing & Stripping Metadata: {file.name}")

        add_watermark(
            image_path=file,
            text=text,
            output_path=output_file,
            font_path=font_path,
            font_size=font_size,
            opacity=opacity,
            bg_opacity=bg_opacity,
        )

    end_time = time.perf_counter()
    total_elapsed = end_time - start_time
    avg_per_image = total_elapsed / total_files if total_files > 0 else 0

    print("-" * 50)
    print("Process Complete! All metadata stripped.")
    print(f"Total Images : {total_files}")
    print(f"Total Time   : {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)")
    print(f"Average Speed: {avg_per_image:.3f} seconds/image")
    print("-" * 50)


# --- CONFIGURATION ---
INPUT_FOLDER = r"D:\Comfy-Desktop\ComfyUI-Shared\output"
OUTPUT_FOLDER = "./output"

WATERMARK_TEXT = ""
FONT_PATH = r"D:\dev\LLM-Experiments\comfy-ui-generator\fonts\Poppins-MediumItalic.ttf"
FONT_SIZE = 40
OPACITY = 255
BG_OPACITY = 50

# Run process
process_folder(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    text=WATERMARK_TEXT,
    font_path=FONT_PATH,
    font_size=FONT_SIZE,
    opacity=OPACITY,
    bg_opacity=BG_OPACITY,
)