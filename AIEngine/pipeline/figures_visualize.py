"""Debug overlay of a chart reading: what was detected, drawn over the crop that was analysed."""
from PIL import ImageDraw

from .charts import ROLE_LEGEND

PLOT_AREA_COLOUR = (0, 150, 0)
LEGEND_COLOUR = (150, 0, 200)
MARK_COLOUR = (255, 0, 255)
TEXT_COLOUR = (255, 150, 0)


def draw_chart_overlay(crop, origin, chart_data, texts=()):
    """A copy of `crop` (PIL; `origin` is the page pixel of its top-left corner) with the plot area
    (green), every text box (orange; legend entries violet) and the detected bars, line corners and
    points (magenta) drawn on it."""
    image = crop.convert("RGB").copy()
    draw = ImageDraw.Draw(image)

    def local(box):
        return [box[0] - origin[0], box[1] - origin[1], box[2] - origin[0], box[3] - origin[1]]

    for item in texts:
        draw.rectangle(local(item.bbox), outline=LEGEND_COLOUR if item.role == ROLE_LEGEND else TEXT_COLOUR)
    if chart_data.get("plot_area_image_bbox"):
        draw.rectangle(local(chart_data["plot_area_image_bbox"]), outline=PLOT_AREA_COLOUR, width=2)
    for series in chart_data.get("series", ()):
        for bar in series.get("bars", ()):
            draw.rectangle(local(bar["image_bbox"]), outline=MARK_COLOUR, width=2)
        points = [(p["image_point"][0] - origin[0], p["image_point"][1] - origin[1]) for p in series.get("points", ())]
        for x, y in points:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=MARK_COLOUR, width=2)
    return image
