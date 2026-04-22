from typing import Dict

import matplotlib.pyplot as plt


ROLE_COLORS: Dict[str, str] = {
    "FFSlim": "#4F7CAC",
    "TDP": "#5A90C4",
    "LRU": "#E59D52",
    "Quiver": "#D98145",
    "CoorDL": "#C7633A",
    "EMD-Cache": "#2E8B57",
    "IACMA": "#2E8B57",
    "DGIM-LRU": "#E59D52",
    "DGIM-Quiver": "#D98145",
    "DGIM-CoorDL": "#C7633A",
    "DGIM-IACMA": "#1F6D45",
}

FORGET_COLORS: Dict[str, str] = {
    "DFR": "#6C757D",
    "SFR": "#8A63A8",
    "FFR": "#4E79A7",
}

METHOD_MARKERS: Dict[str, str] = {
    "FFSlim": "o",
    "TDP": "o",
    "LRU": "s",
    "Quiver": "^",
    "CoorDL": "D",
    "EMD-Cache": "P",
    "IACMA": "P",
    "DGIM-LRU": "X",
    "DGIM-Quiver": "v",
    "DGIM-CoorDL": "*",
    "DGIM-IACMA": "h",
}


def apply_eurosys_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 16.0,
            "axes.labelsize": 16.0,
            "axes.titlesize": 16.8,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "legend.fontsize": 12.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def method_color(name: str) -> str:
    return ROLE_COLORS.get(name.strip(), "#7F7F7F")


def method_marker(name: str) -> str:
    return METHOD_MARKERS.get(name.strip(), "o")


def is_ours(name: str) -> bool:
    n = name.strip()
    return n == "EMD-Cache" or n == "IACMA" or n.startswith("DGIM-")


def is_dgim(name: str) -> bool:
    return name.strip().startswith("DGIM-")
