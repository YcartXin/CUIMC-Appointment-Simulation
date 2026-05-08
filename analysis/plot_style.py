from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb


ARRIVAL_COLOR = "#1f77b4"
BALKING_COLOR = "#9467bd"
NO_SHOW_COLOR = "#2ca02c"
CANCELLATION_COLOR = "#d62728"
BASELINE_COLOR = "#7f7f7f"

# Backward-compatible aliases for older figure code. The color meaning is now
# driver family, not metric family.
OVERALL_COLOR = ARRIVAL_COLOR
ACCESS_COLOR = ARRIVAL_COLOR
UTILIZATION_COLOR = NO_SHOW_COLOR
WAIT_COLOR = BALKING_COLOR
ACCEPTED_WAIT_COLOR = BALKING_COLOR
CLASS_1_COLOR = ARRIVAL_COLOR
CLASS_2_COLOR = CANCELLATION_COLOR

DRIVER_COLORS = {
    "arrival": ARRIVAL_COLOR,
    "balking": BALKING_COLOR,
    "no_show": NO_SHOW_COLOR,
    "cancellation": CANCELLATION_COLOR,
}
DRIVER_LABELS = {
    "arrival": "Arrival pressure / mix",
    "balking": "Balking",
    "no_show": "No-show",
    "cancellation": "Cancellation",
}


def blend_color(color: str, target: str = "#ffffff", amount: float = 0.5) -> tuple[float, float, float]:
    base_rgb = np.array(to_rgb(color))
    target_rgb = np.array(to_rgb(target))
    mixed = (1.0 - amount) * base_rgb + amount * target_rgb
    return tuple(mixed)


def driver_cmap(driver: str) -> LinearSegmentedColormap:
    color = DRIVER_COLORS[driver]
    return LinearSegmentedColormap.from_list(
        f"{driver}_sequential",
        [
            blend_color(color, amount=0.92),
            blend_color(color, amount=0.55),
            color,
            blend_color(color, "#000000", 0.18),
        ],
    )


def driver_gap_cmap(driver: str) -> LinearSegmentedColormap:
    color = DRIVER_COLORS[driver]
    return LinearSegmentedColormap.from_list(
        f"{driver}_gap",
        [blend_color(color, amount=0.78), "#ffffff", blend_color(color, "#000000", 0.12)],
    )


DRIVER_CMAPS = {driver: driver_cmap(driver) for driver in DRIVER_COLORS}
DRIVER_GAP_CMAPS = {driver: driver_gap_cmap(driver) for driver in DRIVER_COLORS}
ACCESS_CMAP = DRIVER_CMAPS["arrival"]
UTILIZATION_CMAP = DRIVER_CMAPS["no_show"]
WAIT_CMAP = DRIVER_CMAPS["balking"]
BALKING_CMAP = DRIVER_CMAPS["balking"]
CLASS_GAP_CMAP = DRIVER_GAP_CMAPS["arrival"]


def driver_from_text(*values: object) -> str | None:
    text = " ".join(str(value).lower() for value in values if value is not None)
    if "no_show" in text or "no-show" in text or "no show" in text:
        return "no_show"
    if "balk" in text:
        return "balking"
    if "cancel" in text:
        return "cancellation"
    if "arrival" in text or "lambda" in text or "class_1_share" in text or "capacity stress" in text:
        return "arrival"
    return None


def series_role(label: object, index: int = 0) -> str:
    text = str(label).lower()
    if "class 1" in text or "class_1" in text:
        return "class_1"
    if "class 2" in text or "class_2" in text:
        return "class_2"
    if index == 1:
        return "class_1"
    if index == 2:
        return "class_2"
    return "overall"


def driver_line_style(driver: str | None, label: object, index: int = 0) -> dict[str, object]:
    if driver is None:
        return {}

    color = DRIVER_COLORS[driver]
    role = series_role(label, index)
    styles = {
        "overall": {
            "color": color,
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.5,
        },
        "class_1": {
            "color": blend_color(color, "#000000", 0.10),
            "linestyle": "--",
            "marker": "s",
            "linewidth": 2.2,
        },
        "class_2": {
            "color": blend_color(color, amount=0.24),
            "linestyle": ":",
            "marker": "^",
            "linewidth": 2.2,
        },
    }
    return styles[role]


def plot_driver_line(ax, x, y, label, driver=None, color=None, index=0, **kwargs):
    style = driver_line_style(driver, label, index)
    if not style:
        style = {
            "marker": "o",
            "linewidth": kwargs.pop("linewidth", 2.2),
            "color": color,
        }
    style.update(kwargs)
    ax.plot(x, y, label=label, **style)


def driver_heatmap_cmap(driver: str | None, diverging: bool = False):
    if driver is None:
        return CLASS_GAP_CMAP if diverging else ACCESS_CMAP
    return DRIVER_GAP_CMAPS[driver] if diverging else DRIVER_CMAPS[driver]

