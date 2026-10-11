"""Debug image of the reading order: a page with its blocks numbered in the order they are read."""
from PIL import Image, ImageDraw, ImageFont

from .render import pdf_bbox_to_pixels

ORDER_IMAGE_PATTERN = "page_{:03d}_order.png"
BLOCK_COLOUR = (0, 110, 220)
FULL_WIDTH_COLOUR = (220, 90, 0)
GUTTER_COLOUR = (0, 160, 0)


def order_image_name(page_number):
    return ORDER_IMAGE_PATTERN.format(page_number)


def draw_reading_order(image_path, blocks, gutters, scale_x, scale_y, target):
    """Save a copy of a page image with the main-flow blocks of the page (document blocks, in reading
    order) outlined and numbered 1, 2, 3, ... Full-width blocks are orange, column blocks blue; the
    gutters between columns are green lines. A line joins each block to the next one read."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    for gutter in gutters:
        draw.line([(gutter * scale_x, 0), (gutter * scale_x, image.height)], fill=GUTTER_COLOUR, width=2)
    centres = []
    for number, block in enumerate(blocks, 1):
        x0, y0, x1, y1 = pdf_bbox_to_pixels(block["bbox"], scale_x, scale_y)
        colour = FULL_WIDTH_COLOUR if block["full_width"] else BLOCK_COLOUR
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=3)
        label = str(number)
        left, top, right, bottom = draw.textbbox((x0 + 4, y0 + 2), label, font=font)
        draw.rectangle([left - 3, top - 2, right + 3, bottom + 2], fill=colour)
        draw.text((x0 + 4, y0 + 2), label, fill="white", font=font)
        centres.append(((x0 + x1) / 2, (y0 + y1) / 2))
    if len(centres) > 1:
        draw.line(centres, fill=(200, 0, 120), width=2)
    image.save(target, format="PNG")
    return target
