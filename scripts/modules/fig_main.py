r"""Generate the manuscript figures and the square graphical abstract.

Style invariants:
  - No in-plot subplot titles (descriptions belong in LaTeX captions).
  - Bold panel tags formatted strictly as 'A.', 'B.', 'C.', 'D.' outside axes.
  - Native widths matched to \textwidth (6.80 in) so rendered PDF text is >= 7.0 pt everywhere (>= 6.5 pt floor).
  - Consistent Okabe-Ito semantic color palette across all figures.
  - Vector PDF (fonttype 42) and 300-DPI PNG, plus SVG for the two schematics.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
FIG_DIR = REPO / "figures"

# Okabe-Ito colorblind-safe palette
C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_GREEN = "#009E73"
C_PURP = "#CC79A7"
C_GOLD = "#E69F00"
C_SKY = "#56B4E9"
C_GRAY = "#666666"
C_LIGHT = "#E5E7EB"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8.4,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.15, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        f"{label}.",
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111111",
    )


def save_fig(fig: plt.Figure, stem: str, *, svg: bool = False, tight: bool = True) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight", "pad_inches": 0.04} if tight else {"bbox_inches": fig.bbox_inches}
    fig.savefig(FIG_DIR / f"{stem}.pdf", **kw)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, **kw)
    if svg:
        with plt.rc_context({"svg.fonttype": "none"}):
            fig.savefig(FIG_DIR / f"{stem}.svg", **kw)
    plt.close(fig)


# Schematic coordinates run from left to right and from cytosol down to the surface.
INK = "#193346"
MUTED = "#506675"
MEMBRANE = "#85A1A6"
LUMEN = "#EAF5F7"
BLUE_INK = "#175A78"
COPPER = "#AB4D19"
TEAL = "#167767"
GOLD = "#E3A52C"
PEPTIDE = "#D87A34"
LOCUS_COLORS = ("#468CB4", "#D97736", "#43A089")


class VectorArt:
    """Vector drawing in local biological coordinates, with print-scaled strokes and >= 6.8 pt text floor."""

    def __init__(self, ax, transform=None, pt=None):
        self.ax = ax
        self.transform = ax.transData if transform is None else transform
        width_in = ax.get_position().width * ax.figure.get_figwidth()
        self.pt = width_in * 72 / 120 if pt is None else pt

    def at(self, x, y, scale=1, angle=0):
        local = Affine2D().scale(scale).rotate_deg(angle).translate(x, y)
        return VectorArt(self.ax, local + self.transform, self.pt)

    def line(self, points, color=INK, width=0.28, dashed=False):
        x, y = zip(*points)
        self.ax.plot(
            x,
            y,
            transform=self.transform,
            color=color,
            lw=width * self.pt,
            ls=(0, (2.1, 2.4)) if dashed else "-",
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )

    def curve(self, start, segments, fill="none", edge=INK, width=0.28, closed=False):
        points = [start, *segments]
        codes = [MplPath.MOVETO] + [MplPath.CURVE4] * len(segments)
        if closed:
            points.append(start)
            codes.append(MplPath.CLOSEPOLY)
        patch = mpatches.PathPatch(
            MplPath(points, codes),
            transform=self.transform,
            fc=fill,
            ec=edge,
            lw=width * self.pt,
            zorder=2,
            capstyle="round",
            joinstyle="round",
        )
        self.ax.add_patch(patch)

    def oval(self, x, y, w, h, fill, edge="none", width=0.28, angle=0):
        self.ax.add_patch(
            mpatches.Ellipse(
                (x, y),
                w,
                h,
                angle=angle,
                transform=self.transform,
                fc=fill,
                ec=edge,
                lw=width * self.pt,
                zorder=2,
            )
        )

    def arrow(self, start, end, color=MUTED, bend=0, dashed=False):
        self.ax.add_patch(
            mpatches.FancyArrowPatch(
                start,
                end,
                transform=self.transform,
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={bend}",
                mutation_scale=1.95 * self.pt,
                lw=0.29 * self.pt,
                ls=(0, (2, 2.5)) if dashed else "-",
                color=color,
                shrinkA=0,
                shrinkB=0,
                zorder=2,
            )
        )

    def label(self, x, y, text, size=2.65, color=INK, bold=False, align="center"):
        fs = max(6.8, size * self.pt)
        self.ax.text(
            x,
            y,
            text,
            transform=self.transform,
            ha=align,
            va="center",
            fontsize=fs,
            color=color,
            weight="bold" if bold else "normal",
            zorder=3,
        )


def peptide(art, points, anchors=(), color=PEPTIDE, diameter=0.92):
    art.line(points, color, 0.25)
    for i, (x, y) in enumerate(points):
        art.oval(x, y, diameter, diameter, GOLD if i in anchors else color, "#825426" if i in anchors else "white", 0.09)


def folded_protein(art):
    art.curve(
        (-3.8, 0.6),
        [
            (-5.5, -1.8), (-2.4, -4.5), (-0.7, -3.2),
            (1.8, -5.1), (4.6, -2.4), (3.7, -0.9),
            (5.9, 1.1), (3.6, 4.2), (1.5, 3.3),
            (0.5, 5.3), (-2.6, 4.3), (-2.7, 2.8),
            (-4.4, 3.3), (-5.4, 1.6), (-3.8, 0.6),
        ],
        "#DDE5E8",
        "#94A5AD",
        closed=True,
    )
    art.curve((-3, 0.4), [(-1.7, -2.1), (0.2, -2.6), (0.5, -0.7), (0.9, 1.2), (2.7, 1.4), (3.1, -0.3)], edge="#A6B6BD")


def split_siblings(art):
    art.line([(0, 0), (3.5, 0)], "#91A1A9", dashed=True)
    art.line([(1.1, 0.8), (2.2, -0.8)], "#71858F", 0.36)
    for sign, color in [(-1, BLUE_INK), (1, COPPER)]:
        art.arrow((3.7, 0), (7.7, 2.9 * sign), color)
        peptide(art, [(8.8 + j * 0.83, 3.1 * sign + 0.22 * np.sin(j)) for j in range(4)], color=color, diameter=0.79)


def ubiquitin_substrate(art):
    art.curve(
        (-7.9, 2.7),
        [
            (-6.8, 1), (-6.2, 2.8), (-5.4, 1.6), (-4, -0.8), (-6.3, -1.5), (-6.1, 0.3),
            (-5.8, 2), (-2.9, 0.3), (-1.3, 0.6), (0.3, 0.9), (1.4, 4), (2.5, 2.2),
            (3.6, 0.5), (5.3, 1.2), (8.3, 1.1),
        ],
        edge=BLUE_INK,
        width=0.59,
    )
    art.line([(-1.3, 0.5), (-2.8, -1.7), (-4.8, -3.2)], "#AA7313", 0.39)
    for x, y in [(-2.8, -1.7), (-4.8, -3.2)]:
        art.oval(x, y, 2.4, 2.2, "#F6C451", "#AE751D", width=0.23, angle=-28)


def immunoproteasome(art):
    edge = "#275B70"
    for x, fill, radius in [(3.2, "#A2CADA", 3.15), (1.0, "#5294A8", 3.7), (-1.2, "#629EB2", 3.7), (-3.4, "#BBDCE3", 3.15)]:
        art.curve(
            (x, -radius),
            [
                (x + 0.7, -radius - 0.3), (x + 1.6, -radius - 0.2), (x + 2.1, -radius + 0.1),
                (x + 3.65, -radius + 0.2), (x + 3.65, radius - 0.2), (x + 2.1, radius),
                (x + 1.5, radius + 0.2), (x + 0.8, radius + 0.2), (x, radius),
                (x - 1.6, radius - 0.1), (x - 1.6, -radius + 0.1), (x, -radius),
            ],
            fill,
            edge,
            closed=True,
        )
        for dy in (-0.44, 0.47):
            art.line([(x + 1.05, dy * radius), (x + 2.55, dy * radius)], edge, 0.16)
    art.oval(-3.5, 0, 2.65, 6.1, "#D7EAF0", edge, 0.25)
    for theta in np.linspace(0, 2 * np.pi, 7, endpoint=False):
        art.oval(-3.5 + 0.88 * np.cos(theta), 2.19 * np.sin(theta), 0.59, 1.04, "#90B9C6", edge, 0.10)
    art.oval(-3.65, 0, 1.25, 3.0, "#254D60")
    art.line([(-6.1, 0.6), (-3.8, 0.15)], BLUE_INK, 0.47)


def lipid_bilayer(art, midline, thickness):
    points = np.array(midline)
    tangent = np.gradient(points, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    offset = normal * thickness / 2
    leaflets = np.vstack((points - offset, (points + offset)[::-1]))
    art.ax.add_patch(mpatches.Polygon(leaflets, transform=art.transform, closed=True, fc="#DBE7E5", ec="none", zorder=2))
    for side in (-1, 1):
        art.line(points + side * offset, "#9AB4B5", 0.16)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    for d in np.arange(0.65, distance[-1] - 0.3, 1.53):
        i = np.searchsorted(distance, d) - 1
        fraction = (d - distance[i]) / (distance[i + 1] - distance[i])
        center = points[i] + fraction * (points[i + 1] - points[i])
        for side in (-1, 1):
            head = center + side * offset[i]
            tail = center + side * 0.07 * normal[i]
            for fork in (-0.23, 0.23):
                art.line([head, tail + fork * tangent[i]], "#8DABA8", 0.12)
            art.oval(*head, 0.52, 0.52, "#9DBBBC", "#708F94", 0.09)


def membrane_pore(art, leader=False):
    edge = "#65516F" if leader else BLUE_INK
    colors = ("#C4AEC5", "#AE93B1") if leader else ("#8BBACC", "#67A1B9")
    art.oval(0, 0, 6.6, 4, "white")
    for sign, fill in zip((-1, 1), colors):
        outline = [
            (-1.9, -2.1), (-0.6, -2.9), (0.1, -1.9),
            (0.8, -1.0), (-0.4, 0.8), (0.45, 2.1),
            (0.1, 3.0), (-1.8, 3.0), (-2.3, 2.0),
            (-3.1, 1.0), (-2.2, 0), (-2.7, -0.9),
        ]
        art.curve((sign * 2.7, -0.9), [(-sign * x, y) for x, y in outline], fill, edge, 0.29, closed=True)
        art.line([(sign * 1.7, -1.55), (sign * 1.35, 0), (sign * 1.65, 1.7)], "#F4F9FA", 0.23)
        if not leader:
            art.oval(sign * 2.5, -2.75, 3.05, 2.3, fill, edge, 0.26, angle=sign * 23)


def ribosome(art):
    art.curve(
        (-3.0, 0.9),
        [
            (-5.3, -0.6), (-3.6, -3.1), (-1.6, -3.0),
            (-0.7, -4.8), (2.6, -4.3), (3.0, -2.0),
            (5.1, -0.8), (3.9, 1.3), (2.3, 1.45),
            (0.4, 1.9), (-1.2, 0.9), (-3.0, 0.9),
        ],
        "#D7C9DA",
        "#785F7C",
        closed=True,
    )
    art.curve((-2.85, -0.7), [(-2.1, -2.05), (-1.45, -1.25), (-1.6, -0.65), (-1.75, 0.0), (-0.6, 0.35), (0.0, -0.5)], edge="#A794AB", width=0.19)
    art.curve((-0.2, -2.1), [(0.8, -2.75), (1.9, -2.15), (1.83, -1.35)], edge="#EFE6F0", width=0.25)
    art.oval(0, 2.1, 5.8, 2.1, "#BDA7C1", "#785F7C", 0.23)
    art.curve((-5.9, 1.0), [(-2.4, 2.1), (1.2, -0.2), (5.7, 1.4)], edge="#785F7C", width=0.3)


def scissors(art, color=TEAL):
    for sign in (-1, 1):
        art.oval(-2.15, 1.26 * sign, 2.65, 1.77, "white", color, 0.43, angle=-sign * 24)
        art.line([(-1.13, sign * 0.85), (0.55, 0), (3.55, -sign * 1.65)], color, 0.46)
    art.oval(0.55, 0, 0.90, 0.90, "#F5C25A", color, 0.19)


def pmhc(art, fill):
    art.line([(0.1, -1.15), (0.2, 2.5), (-0.4, 3.8)], BLUE_INK, 0.55)
    art.oval(0.55, 3.0, 2.65, 3.35, fill, INK, 0.25, angle=-20)
    art.oval(-1.6, 3.35, 2.25, 2.03, "#CADDE4", BLUE_INK, 0.22, angle=22)
    art.curve((0.15, 2.25), [(0.65, 2.2), (1.38, 2.75), (1.36, 3.32)], edge="#F3F7F8", width=0.18)
    art.curve((-2.25, 3.4), [(-2.24, 3.0), (-1.7, 2.8), (-1.4, 2.94)], edge="white", width=0.15)
    art.curve(
        (-3.65, 4.15),
        [
            (-3.75, 3.05), (-2.65, 3.25), (-2.1, 3.8),
            (-1.1, 4.65), (1.1, 4.8), (2.3, 4.0),
            (3.5, 3.08), (4.05, 3.52), (3.7, 4.85),
            (3.22, 6.12), (2.6, 6.10), (2.23, 5.45),
            (0.65, 5.94), (-1.7, 5.7), (-2.22, 5.17),
            (-2.9, 6.20), (-3.72, 5.60), (-3.65, 4.15),
        ],
        fill,
        INK,
        0.23,
        closed=True,
    )
    peptide(art, [(x, 5.5 + 0.54 * np.cos(x * 0.53)) for x in np.linspace(-3.43, 3.43, 9)], anchors=(1, 8), color="#924A30", diameter=0.77)


def loading_complex(art):
    pmhc(art, LOCUS_COLORS[0])
    art.curve(
        (3.5, -0.9),
        [
            (4.5, 0.4), (3.65, 2.75), (4.3, 3.4),
            (5.8, 4.9), (5.5, 6.8), (3.35, 7.0),
            (1.5, 7.35), (0.6, 6.6), (1.7, 6.55),
            (4.0, 6.6), (4.4, 5.3), (3.1, 4.1),
            (2.25, 2.95), (3.55, 1.9), (2.5, -0.65),
        ],
        "#88C0AE",
        TEAL,
        closed=True,
    )


def stromal_cell(art):
    art.curve(
        (-6.5, 2.25),
        [
            (-4.1, 1.55), (-3.0, -2.35), (0.05, -2.2),
            (2.05, -2.28), (3.15, -0.65), (6.65, -1.85),
            (4.25, -0.18), (3.1, 0.1), (1.19, 1.45),
            (-1.6, 3.55), (-2.5, 1.25), (-6.5, 2.25),
        ],
        "#DBE3E7",
        "#91A3AC",
        width=0.24,
        closed=True,
    )
    art.oval(0.15, -0.1, 3.35, 1.7, "#A7B9C2", angle=-21)
    art.curve((-4.8, 1.6), [(-3.1, 1.2), (-2.5, 0.2), (-1.7, -0.45)], edge="#A7B9C2", width=0.18)


def processing_section(art, abstract=False):
    def er_roof(x):
        return 27.2 + 0.95 * np.cos((x - 12) / 19) - 1.7 * np.exp(-((x - 108) / 10) ** 2)

    def er_floor(x):
        return 44.7 + 1.4 * np.sin((x - 3) / 22) - 1.9 * np.exp(-((x - 93) / 9) ** 2)

    def surface(x):
        return 51.0 + 1.2 * np.sin((x - 5) / 25) - 0.00063 * (x - 40) ** 2

    xs = np.linspace(0, 120, 241)
    surface_points = list(zip(xs, surface(xs)))
    art.ax.fill_between(xs, 0, surface(xs), transform=art.transform, fc="#FDFCF9", lw=0, zorder=1)
    xs = np.linspace(8, 120, 181)
    top = er_roof(xs)
    bottom = er_floor(xs) - 5.4 / (1 + np.exp(-(xs - 113) / 1.9))
    middle, radius = (top[0] + bottom[0]) / 2, (bottom[0] - top[0]) / 2
    t = np.linspace(-np.pi / 2, np.pi / 2, 45)[1:-1]
    left_cap = list(zip(8 - 5.55 * np.cos(t) - 1.8 * np.cos(t) * np.sin(t), middle - radius * np.sin(t)))
    contour = [*zip(xs[::-1], bottom[::-1]), *left_cap, *zip(xs, top)]
    art.ax.add_patch(mpatches.Polygon(contour, transform=art.transform, closed=True, fc=LUMEN, ec="none", zorder=2))
    lipid_bilayer(art, contour, thickness=1.29)
    lipid_bilayer(art, surface_points, thickness=1.7)

    if abstract:
        art.label(1.5, 2.0, "Cytosol", 2.65, MUTED, True, "left")
        art.label(14.0, 6.9, "Bulk protein", 2.65, bold=True)
    for x, y, s in [(6.8, 14.0, 0.67), (10.3, 11.1, 0.64), (13.3, 14.7, 0.65)]:
        folded_protein(art.at(x, y, s, (x - 10) * 13))
    split_siblings(art.at(16.9, 13.1, 0.84))
    art.label(15.2, 20.5, r"Mature protein ($R^2 \leq 15.6\%$)" if not abstract else r"Parent $R^2 \leq 15.6\%$", 2.30 if not abstract else 2.50)
    if abstract:
        art.label(17.0, 24.4, "Sibling split 27–67%", 2.45, MUTED)
        art.label(36.3, 6.85, "K-ε-GG Ub", 2.57, bold=True)
    else:
        art.label(43.2, 6.8, "Ub-tagged substrate", 2.25, BLUE_INK, True)
    ubiquitin_substrate(art.at(43.8, 12.9, 1))
    immunoproteasome(art.at(55.8, 13.85, 1.13))
    art.label(56.0 if abstract else 56.0, 3.3, "20S immunoproteasome" if abstract else "Immunoproteasome", 2.55 if abstract else 2.35, bold=True)
    if abstract:
        art.label(50.8, 20.9, r"IFN-γ  β1i/β2i/β5i", 2.50, BLUE_INK)
    for x, y in [(65, 13.8), (69.2, 15.1)]:
        peptide(art, [(x + j * 0.76, y + 0.23 * np.sin(j)) for j in range(4)])
    art.arrow((72.1, 16.5), (73.15, 21.7), BLUE_INK, bend=-0.18)
    art.label(99.5 if abstract else 101.5, 4.2, "Signal bypass" if abstract else "Sec61 signal bypass", 2.55 if abstract else 2.30, bold=True)
    art.label(99.5 if abstract else 101.5, 9.6, "83% signal span" if not abstract else "83% signal span kept", 2.45 if abstract else 2.25, COPPER)
    if abstract:
        art.label(99.5, 13.8, "in TAP2-KO", 2.40, COPPER)
    ribosome(art.at(102.2, 20.6, 0.91))

    tap_x, sec_x = 73.2, 102.2
    membrane_pore(art.at(tap_x, er_roof(tap_x)))
    membrane_pore(art.at(sec_x, er_roof(sec_x), 0.92), leader=True)
    art.label(81.5, 21.0, "TAP1/2", 2.55, BLUE_INK, True)
    if abstract:
        art.label(112.4, 21.7, "Sec61", 2.55, "#705776", True)
    art.curve((102.4, 22.5), [(101.8, 24.8), (102, 27.0), (102.2, 28.7), (102.7, 31.0), (100.0, 32.6), (97.8, 31.5)], edge=COPPER, width=0.51)
    peptide(art, [(tap_x + 0.18 * np.cos(y), y) for y in np.linspace(26.2, 30.8, 5)])
    art.arrow((72.6, 31.9), (62.2, 34.7), BLUE_INK, bend=-0.15)
    art.arrow((96.1, 32.1), (78.0, 34.7), COPPER, bend=-0.18)
    art.label(103.0, 38.3, "ER lumen", 2.55, MUTED, True)

    peptide(art, [(58.0 + j * 0.78, 34.0 + 0.17 * np.cos(j)) for j in range(5)])
    peptide(art, [(55.0, 31.9), (55.7, 32.1)])
    scissors(art.at(54.45, 33.65, 0.89, -20))
    art.label(55.8, 39.2, "ERAP1", 2.55, bold=True)
    if abstract:
        art.label(55.8, 43.0, r"Length $r = -0.75$", 2.45, TEAL)
    art.arrow((46.8, 34.6), (33, 34.5), BLUE_INK)
    loading_complex(art.at(23, er_roof(23), 1.03))
    art.label(23.0, 39.0, "PLC / tapasin", 2.55, bold=True)
    if abstract:
        art.label(23.0, 42.8, "P2 / P9 anchors", 2.45, MUTED)

    art.arrow((26.3, er_floor(26.3) + 0.45), (25, surface(25) - 1.6), BLUE_INK, bend=0.23)
    if abstract:
        art.label(68.2, 48.25, "Plasma membrane", 2.40, MUTED)
    for x, locus in [(7.9, 0), (19.4, 1), (28.0, 1), (39.4, 2)]:
        pmhc(art.at(x, surface(x), 1.03), LOCUS_COLORS[locus])
    for x, name, ink in [(7.5 if abstract else 4.8, "HLA-A", INK), (23.5 if abstract else 22.8, "HLA-B ↑" if abstract else "HLA-B", COPPER), (39.2 if abstract else 41.5, "HLA-C", INK)]:
        art.label(x, 63.8, name, 2.55 if abstract else 2.38, ink, True)


def biopsy_admixture(art):
    for x, y, r in [(-5.3, 0.7, 1.6), (-3.1, -1.4, 1.8), (-2.25, 1.65, 1.6), (-0.15, -0.1, 1.9), (1.7, -2.0, 1.3)]:
        cell = art.at(x, y, r)
        cell.curve(
            (-0.93, -0.20),
            [
                (-1.13, -0.84), (-0.48, -1.02), (0.1, -0.96),
                (0.68, -1.08), (1.04, -0.55), (1, 0.15),
                (1, 0.76), (0.34, 0.90), (-0.29, 0.86),
                (-0.95, 0.82), (-1.12, 0.24), (-0.93, -0.20),
            ],
            "#E9EFF1",
            "#A4B6BD",
            width=0.17,
            closed=True,
        )
        cell.oval(-0.02, 0.04, 0.81, 0.71, "#A3B8BF")
    for x, y, s in [(3.65, 1.05, 0.68), (0.35, -3.55, 0.53), (-1.2, 3.55, 0.48)]:
        stromal_cell(art.at(x, y, s))


def draw_fig1_section(ax):
    ax.set(xlim=(0, 120), ylim=(70, 0))
    ax.axis("off")
    art = VectorArt(ax)
    processing_section(art, abstract=False)

    # Visual Inverse Encoding Pipeline: Eluted 9-mer peptide MS distribution pi_i -> binary one-hot Phi -> q_i -> Paired Delta q
    art.arrow((42.4, 57.2), (45.6, 57.2), BLUE_INK)

    # 1. Neutral multi-peptide MS distribution pi_i (many distinct 9-mer sequences summing to 1, NOT 3 HLA buckets)
    px0, py_base, pw, ph = 46.6, 62.6, 10.0, 10.6
    art.line([(px0, py_base), (px0 + pw, py_base)], INK, 0.22)
    art.line([(px0, py_base), (px0, py_base - ph)], INK, 0.18)
    peak_heights = [0.34, 0.88, 0.22, 0.64, 0.45, 0.96, 0.28, 0.52, 0.74, 0.19, 0.41, 0.60]
    for ki, h_rel in enumerate(peak_heights):
        bx = px0 + 0.45 + ki * (pw - 0.9) / len(peak_heights)
        bh = h_rel * (ph - 1.2)
        art.ax.add_patch(
            mpatches.Rectangle(
                (bx, py_base - bh),
                0.48,
                bh,
                transform=art.transform,
                fc="#475569",
                ec="none",
                zorder=2,
            )
        )
    art.label(px0 + pw / 2, 66.4, r"Peptide $\pi_i$ ($\Sigma=1$)", 2.02, INK, True)

    art.arrow((57.4, 57.2), (60.4, 57.2), MUTED)

    # 2. Compact visual one-hot indicator matrix Phi (positions P1..P9 x amino acids)
    gx0, gy0, gw, gh = 61.3, 52.0, 12.2, 10.6
    art.ax.add_patch(mpatches.Rectangle((gx0, gy0), gw, gh, transform=art.transform, fc="#F8FAFC", ec="#64748B", lw=0.22 * art.pt, zorder=2))
    n_c, n_r = 9, 6
    cw, ch = gw / n_c, gh / n_r
    hot_cells = {(0, 1), (1, 4), (2, 2), (3, 0), (4, 3), (5, 2), (6, 5), (7, 1), (8, 4)}
    for ci in range(n_c):
        for ri in range(n_r):
            fc = (COPPER if ci in (1, 8) else BLUE_INK) if (ci, ri) in hot_cells else "none"
            art.ax.add_patch(
                mpatches.Rectangle(
                    (gx0 + ci * cw, gy0 + ri * ch),
                    cw,
                    ch,
                    transform=art.transform,
                    fc=fc,
                    ec="#CBD5E1",
                    lw=0.07 * art.pt,
                    zorder=3,
                )
            )
    art.label(gx0 + gw / 2, 66.4, r"Matrix $\Phi$", 2.05, INK, True)

    art.arrow((74.4, 57.2), (77.4, 57.2), BLUE_INK)

    # 3. Motif summary q_i = Phi pi_i and Paired shift Delta q_c = q_1 - q_0
    box_x, box_y, box_w, box_h = 78.3, 51.6, 17.4, 11.4
    art.ax.add_patch(
        mpatches.FancyBboxPatch(
            (box_x, box_y),
            box_w,
            box_h,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            transform=art.transform,
            fc="#EFF6FF",
            ec=BLUE_INK,
            lw=0.24 * art.pt,
            zorder=2,
        )
    )
    art.label(box_x + box_w / 2, 54.9, r"$q_i = \Phi\pi_i$", 2.18, INK, True)
    art.label(box_x + box_w / 2, 59.9, r"Paired $\Delta q_c$", 2.18, BLUE_INK, True)
    art.label(box_x + box_w / 2, 66.4, "0.95 / 0.79", 2.05, TEAL, True)

    # 4. Contrasting unpaired tumor biopsy on far right
    biopsy_admixture(art.at(109.2, 56.2, 0.70))
    art.label(109.2, 66.4, "Unpaired biopsy", 2.05, MUTED, True)


def perturbation_inputs(art):
    receptor = art.at(8.6, 15.3)
    for s in (-1, 1):
        receptor.curve((1.2 * s, 4.0), [(s * 1.2, 2.6), (s * 1.5, 0), (s * 1.6, -0.5), (s * 3.0, -0.8), (s * 3.2, -1.8), (s * 2.6, -2.6)], edge=BLUE_INK, width=0.59)
        receptor.oval(s * 1.9, -1.7, 1.25, 2.25, "#A7CDDD", BLUE_INK, width=0.24, angle=s * 23)
        receptor.oval(s * 0.75, -3.05, 2.0, 1.7, "#66A399", TEAL, width=0.22)
    receptor.line([(-4, 1.25), (4, 1.25)], MEMBRANE, 0.22)
    receptor.line([(-4, 1.85), (4, 1.85)], MEMBRANE, 0.22)
    kinase = art.at(47.5, 15.4)
    kinase.curve((-3.6, -0.8), [(-4.9, -3.5), (-1.4, -4.4), (0.5, -3.0), (3.0, -3.9), (4.1, -1.7), (2.8, -0.5), (1.5, 0), (0.4, -1.3), (-0.6, -0.6), (-2.6, 0.5), (-2.1, 1.0), (-3.6, -0.8)], "#E8B68F", COPPER, closed=True)
    kinase.curve((-3.6, 0.7), [(-1.4, 2.7), (0.35, -0.8), (1.7, 0.5), (3.3, 1.9), (2.1, 3.7), (0, 3.5), (-2.25, 4.3), (-4.5, 2.85), (-3.6, 0.7)], "#F2D5B7", COPPER, closed=True)
    kinase.oval(2.6, 0.0, 1.85, 0.9, GOLD, COPPER, 0.15, angle=-20)
    cas = art.at(85, 15.3)
    cas.curve((-1.1, -2.8), [(-3.6, -3.3), (-4.3, -0.6), (-2.6, 1.2), (-2.6, 3.5), (0.7, 4.6), (1.45, 2.25), (3.6, 2.05), (4.05, -0.65), (2.05, -1.75), (1.5, -3.4), (0, -3.9), (-1.1, -2.8)], "#BED7CC", TEAL, closed=True)
    for dy in (-0.52, 0.52):
        cas.curve((-4.6, 2.0 + dy), [(-1.9, 1.9 + dy), (1.8, -2.0 + dy), (4.6, -1.9 + dy)], edge=BLUE_INK, width=0.21)
    scissors(cas.at(1.75, -2.55, 0.51, 55))
    for x, title, subtitle in [(15.2, "IFN-γ / IFN-α", "JAK/STAT · ISRE"), (54.1, "MAPKi / TGF-β", "Kinase signaling"), (91.7, "APPM knockout", "CRISPR–Cas9")]:
        art.label(x, 13.55, title, 2.85, INK, True, "left")
        art.label(x, 17.6, subtitle, 2.46, MUTED, align="left")
    for x in (8.6, 47.5, 85):
        art.arrow((x, 20.3), (x, 22.9), "#7D959B")


def paired_readout(art):
    art.label(4.0, 89.2, "Paired readout", 2.86, BLUE_INK, True, "left")
    art.label(46.9, 89.2, r"9-mer position  $\Delta q_s$", 2.7, INK, True)
    art.label(94.4, 89.2, r"Locus share  $\Delta w_{\rm A,B,C,U}$", 2.7, INK, True)
    for y, name in [(94.0, "Baseline"), (98.9, "Perturbed")]:
        art.label(4.0, y, name, 2.52, INK, align="left")
    for i, x in enumerate(np.linspace(29.2, 64.9, 9)):
        anchor = i in (1, 8)
        for y, perturbed in [(94.0, False), (98.9, True)]:
            accent = 0.35
            if anchor and perturbed:
                accent = 0.77 if i == 1 else 0.17
            top = y - 1.45
            for fraction, color in [(1 - accent, "#CBDADE"), (accent, COPPER if anchor else "#648B9D")]:
                height = 2.9 * fraction
                art.ax.add_patch(mpatches.Rectangle((x - 1.65, top), 3.3, height, transform=art.transform, fc=color, ec="none", zorder=2))
                top += height
        art.label(x, 102.8, str(i + 1), 2.30, COPPER if anchor else MUTED, bold=anchor)
    share_colors = (*LOCUS_COLORS, "#C6D2D6")
    for y, widths in [(94.0, (17, 8, 7, 4)), (98.9, (11, 15, 6, 4))]:
        left = 76.4
        for width, color in zip(widths, share_colors):
            art.ax.add_patch(mpatches.Rectangle((left, y - 1.39), width, 2.78, transform=art.transform, fc=color, ec="white", lw=0.13 * art.pt, zorder=2))
            left += width
    for x, name, col in zip((76.8, 82.8, 88.8, 94.8), ("A", "B", "C", "Unass."), share_colors):
        art.oval(x, 102.8, 1.16, 1.16, col)
        art.label(x + 1.4, 102.8, name, 2.30, MUTED, align="left")


def make_graphical_abstract() -> None:
    fig = plt.figure(figsize=(5.5, 5.5), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 120), ylim=(120, 0))
    ax.axis("off")
    art = VectorArt(ax)
    art.label(3.85, 5.25, "Paired HLA-I shifts track signaling", 4.10, INK, True, "left")
    art.label(116.1, 5.5, "11 human datasets", 2.60, MUTED, align="right")
    perturbation_inputs(art)

    inset_width = 0.938
    inset_height = inset_width * 70 / 120
    section = fig.add_axes((0.031, 1 - 23.5 / 120 - inset_height, inset_width, inset_height))
    section.set(xlim=(0, 120), ylim=(70, 0))
    section.axis("off")
    cell = VectorArt(section)
    processing_section(cell, abstract=True)
    cell.label(53.5, 55.0, "HLA-I 9-mer", 2.65, BLUE_INK, True, "left")
    peptide(cell, [(54.4 + i * 2.0, 59.2 + 0.56 * np.sin(i * 0.7)) for i in range(9)], anchors=(1, 8), color="#7493A1", diameter=1.47)
    cell.label(56.3, 63.9, "P2", 2.35, COPPER)
    cell.label(70.5, 63.9, "P9", 2.35, COPPER)
    biopsy_admixture(cell.at(99.6, 59.2, 0.81))
    cell.label(104.1, 53.3, "Unpaired biopsy", 2.55, MUTED, True)
    cell.label(104.1, 65.3, "Stromal admixture", 2.45, MUTED)
    cell.line([(106.6, 58.0), (109.4, 60.8)], "#8E9DA3", 0.43)
    cell.line([(106.6, 60.8), (109.4, 58.0)], "#8E9DA3", 0.43)

    paired_readout(art)
    art.line([(3.9, 105.7), (116.1, 105.7)], "#CFDADF", 0.22)
    for x, score, metric, detail, color in [
        (20.5, "0.95 / 0.79", "PosAA180 AUROC", "LOLO / strict cross-study", BLUE_INK),
        (60.0, "76.8%", "Within-locus motif", "Cross-study shift share", COPPER),
        (99.5, "+4.86", "IFN-α β-cell bridge", "Frozen IFN-γ decoder", TEAL),
    ]:
        art.label(x, 109.75, score, 4.35, color, True)
        art.label(x, 114.10, metric, 2.65, INK, True)
        art.label(x, 117.37, detail, 2.35, MUTED)
    save_fig(fig, "graphical_abstract", svg=True, tight=False)


def make_fig1() -> None:
    """Figure 1: Full-width Subcellular Schematic (Panel A) over 3-column Empirical Row (Panels B, C, D)."""
    fig = plt.figure(figsize=(6.80, 6.45))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.88, 1.0], hspace=0.38, top=0.97, bottom=0.10, left=0.12, right=0.98)

    # Panel A: Full-width subcellular processing schematic (exact 120:70 aspect)
    ax_a = fig.add_subplot(gs[0, 0])
    width = 0.86
    height = width * fig.get_figwidth() / fig.get_figheight() * 70 / 120
    ax_a.set_position([0.07, 0.97 - height, width, height])
    draw_fig1_section(ax_a)
    add_panel_label(ax_a, "A", x=-0.04, y=1.02)

    gs_bot = gs[1, 0].subgridspec(1, 3, wspace=0.52, width_ratios=[1.12, 1.0, 1.08])

    # Panel B: Cross-donor vs paired within-donor & Velz partition ARI
    ax_b = fig.add_subplot(gs_bot[0, 0])
    add_panel_label(ax_b, "B", x=-0.42, y=1.08)
    bench = pd.read_csv(RES / "exp1_concentration_decoupling/decoding_benchmark_synthesis.tsv", sep="\t")
    vari = pd.read_csv(RES / "exp8_immunity_bridge/velz_ari_all_pairs.csv")
    ari_rna_meth = float(vari[(vari["first"] == "RNA") & (vari["second"] == "methylation")]["score"].iloc[0])
    ari_c96_meth = float(vari[(vari["first"] == "coarse96") & (vari["second"] == "methylation")]["score"].iloc[0])
    ari_c96_rna = float(vari[(vari["first"] == "RNA") & (vari["second"] == "coarse96")]["score"].iloc[0])

    labels_b = [
        "Cross-PDO (C96)",
        "Paired PDO (C96)",
        "Paired PDO (AA20)",
        "RNA vs. Meth ARI",
        "C96 vs. Meth ARI",
        "C96 vs. RNA ARI",
    ]
    vals_b = [
        float(bench.iloc[0]["primary_metric"]),
        float(bench.iloc[1]["primary_metric"]),
        float(bench.iloc[1]["secondary_metric"]),
        ari_rna_meth,
        ari_c96_meth,
        ari_c96_rna,
    ]
    cols_b = [C_VERM, C_BLUE, C_SKY, C_GREEN, C_GOLD, C_VERM]
    y_pos = np.arange(len(labels_b))[::-1]
    ax_b.barh(y_pos, vals_b, height=0.60, color=cols_b, edgecolor="none", alpha=0.90)
    ax_b.axvline(0.5, ls="--", lw=0.75, color=C_GRAY, alpha=0.7)
    ax_b.axvline(0.0, ls="-", lw=0.65, color="#111111")
    for yp, v in zip(y_pos, vals_b):
        ax_b.text(v + 0.025, yp, f"{v:.2f}", va="center", fontsize=6.8, fontweight="bold" if v > 0.6 else "normal")
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(labels_b, fontsize=6.7)
    ax_b.set_xlim(-0.02, 1.24)
    ax_b.set_xlabel("Score / ARI", fontsize=7.6)
    ax_b.tick_params(axis="x", labelsize=6.8)
    despine(ax_b)

    # Panel C: Velz N=28 Medulloblastoma Angiogenesis vs HLA-I/B2M mRNA
    ax_c = fig.add_subplot(gs_bot[0, 1])
    add_panel_label(ax_c, "C", x=-0.22, y=1.08)
    vcov = pd.read_csv(RES / "exp1_concentration_decoupling/velz_microenvironment_patient_scores.tsv", sep="\t")
    vcol = pd.read_csv(RES / "exp1_concentration_decoupling/velz_angiogenesis_collinearity.tsv", sep="\t")
    sub_palette = {"WNT": C_GOLD, "SHH": C_VERM, "G3": C_BLUE, "G4": C_GREEN}
    sub_display = {"WNT": "WNT", "SHH": "SHH", "G3": "G3", "G4": "G4"}
    for sg in ["WNT", "SHH", "G3", "G4"]:
        g = vcov[vcov["molecular_subgroup"] == sg]
        if len(g):
            ax_c.scatter(g["hla_i_b2m_mrna"], g["angiogenesis_target"], s=24, color=sub_palette[sg], label=sub_display[sg], alpha=0.9, edgecolors="white", linewidth=0.35)
    x_line = np.linspace(vcov["hla_i_b2m_mrna"].min() - 0.1, vcov["hla_i_b2m_mrna"].max() + 0.1, 100)
    sl, ic = np.polyfit(vcov["hla_i_b2m_mrna"], vcov["angiogenesis_target"], 1)
    ax_c.plot(x_line, sl * x_line + ic, color="#111111", lw=1.0, ls="--")
    r_hlai = vcol.loc[vcol["covariate"] == "hla_i_b2m_mrna", "pearson_r_target"].iloc[0]
    r_endo = vcol.loc[vcol["covariate"] == "endothelial_mrna", "pearson_r_target"].iloc[0]
    r_myel = vcol.loc[vcol["covariate"] == "myeloid_mrna", "pearson_r_target"].iloc[0]
    ax_c.text(
        0.04,
        0.96,
        f"HLA-I/B2M: $r={r_hlai:.2f}$\nEndo. ${r_endo:.2f}$ | Myel. ${r_myel:.2f}$",
        transform=ax_c.transAxes,
        va="top",
        fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.16", fc="#F8FAFC", ec="#CBD5E1", lw=0.55),
    )
    ax_c.set_ylim(1.36, 3.45)
    ax_c.set_xlabel("Biopsy HLA-I / B2M ($\\log_2$)", fontsize=7.6)
    ax_c.set_ylabel("Angiogenesis score", fontsize=7.6)
    ax_c.tick_params(labelsize=6.8)
    ax_c.legend(frameon=False, loc="lower right", ncol=2, fontsize=6.5, handletextpad=0.2, columnspacing=0.6)
    despine(ax_c)

    # Panel D: All 8 proteome-matched perturbation contrasts
    ax_d = fig.add_subplot(gs_bot[0, 2])
    add_panel_label(ax_d, "D", x=-0.20, y=1.08)
    elast = pd.read_csv(RES / "exp1_concentration_decoupling/perturbation_elasticity_summary.tsv", sep="\t")
    f7a = pd.read_csv(RES / "exp7_distributional_sensing/exp7a_followup_elasticity_and_ub.tsv", sep="\t")
    r_sk5 = f7a[f7a["cohort"] == "Stopfer2022_SKMEL5_MEKi_100nM_72h"].iloc[0]
    r_mm24 = f7a[f7a["cohort"] == "Bernhardt2025_MaMel63a_IFNg_24h"].iloc[0]
    r_mm72 = f7a[f7a["cohort"] == "Bernhardt2025_MaMel63a_IFNg_72h"].iloc[0]
    plot_rows = [
        ("ITO57T", float(elast.iloc[0]["r_squared"]), float(elast.iloc[0]["frac_genes_diverging_gt_2fold"])),
        ("ITO58M", float(elast.iloc[1]["r_squared"]), float(elast.iloc[1]["frac_genes_diverging_gt_2fold"])),
        ("ITO66T", float(elast.iloc[2]["r_squared"]), float(elast.iloc[2]["frac_genes_diverging_gt_2fold"])),
        ("UWB1.289", float(elast.iloc[3]["r_squared"]), float(elast.iloc[3]["frac_genes_diverging_gt_2fold"])),
        ("MaMel 24h", float(r_mm24["r_squared"]), float(r_mm24["frac_multi_genes_gt_2fold"])),
        ("MaMel 72h", float(r_mm72["r_squared"]), float(r_mm72["frac_multi_genes_gt_2fold"])),
        ("A375 TGF$\\beta$", float(elast.iloc[4]["r_squared"]), float(elast.iloc[4]["frac_genes_diverging_gt_2fold"])),
        ("SKMEL5 MEKi", float(r_sk5["r_squared"]), float(r_sk5["frac_multi_genes_gt_2fold"])),
    ]
    x_idx = np.arange(len(plot_rows))
    w = 0.38
    r2_vals = [100.0 * r[1] for r in plot_rows]
    dv_vals = [100.0 * r[2] for r in plot_rows]
    ax_d.bar(x_idx - w / 2, r2_vals, width=w, color=C_VERM, label="Proteome $R^2$ (%)")
    ax_d.bar(x_idx + w / 2, dv_vals, width=w, color=C_BLUE, label="Sibling $>2\\times$ (%)")
    ax_d.axvline(5.5, ls=":", lw=0.8, color=C_GRAY)
    ax_d.set_xticks(x_idx)
    ax_d.set_xticklabels([r[0] for r in plot_rows], fontsize=6.5, rotation=38, ha="right")
    ax_d.set_ylim(0, 88)
    ax_d.set_ylabel("Percentage (%)", fontsize=7.6)
    ax_d.tick_params(axis="y", labelsize=6.8)
    ax_d.legend(frameon=False, loc="upper right", fontsize=6.5)
    despine(ax_d)

    save_fig(fig, "fig1_concentration_decoupling", svg=True)


def make_fig_distributional() -> None:
    """Figure 2: LOLO & Strict Cross-Study Transfer, Kitagawa-Oaxaca Decomposition, External Experimental Bridge & SK-Mel-28."""
    fig = plt.figure(figsize=(6.55, 5.85))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.40)
    r8 = RES / "exp8_immunity_bridge"

    # Panel A: 0-1 AUROC axis comparing LOLO vs Strict Study+Lineage-Purged LOSO across representations
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "A")
    tr = pd.read_csv(r8 / "cross_transfer_all_representations_db_vs_il.csv")
    tr_db = tr[tr["sequence_coding"] == "Database_20AA"].set_index("representation")
    tr_il = tr[tr["sequence_coding"] == "Isobaric_IL_Harmonized"].set_index("representation")

    items_a = [
        ("P4–P7 pairs", tr_db.loc["R1_Coarse96", "lolo_auroc"], tr_db.loc["R1_Coarse96", "loso_strict_auroc"]),
        ("P2, P9 anchors", tr_db.loc["R2c_Anchor40", "lolo_auroc"], tr_db.loc["R2c_Anchor40", "loso_strict_auroc"]),
        ("Exact 9-mer", tr_il.loc["R4_Peptide_Identity", "lolo_auroc"], tr_il.loc["R4_Peptide_Identity", "loso_strict_auroc"]),
        ("Parent gene", tr_db.loc["R5_SourceGene_Oracle", "lolo_auroc"], tr_db.loc["R5_SourceGene_Oracle", "loso_strict_auroc"]),
        ("P1–P9 (20-AA)", tr_db.loc["R2_PosAA180", "lolo_auroc"], tr_db.loc["R2_PosAA180", "loso_strict_auroc"]),
        ("P1–P9 (I=L)", tr_il.loc["R2_PosAA180", "lolo_auroc"], tr_il.loc["R2_PosAA180", "loso_strict_auroc"]),
    ]
    xa = np.arange(len(items_a))
    wb = 0.36
    lolo_v = [float(x[1]) for x in items_a]
    loso_v = [float(x[2]) for x in items_a]
    ax_a.bar(xa - wb / 2, lolo_v, width=wb, color=C_BLUE, alpha=0.92, label="LOLO (16 lineages)")
    ax_a.bar(xa + wb / 2, loso_v, width=wb, color=C_GREEN, alpha=0.92, label="Strict cross-study LOSO")
    ax_a.axhline(0.50, ls="--", lw=0.8, color=C_GRAY)
    for xi, sv in enumerate(loso_v):
        if xi >= 3:
            ax_a.text(xi + wb * 0.55, sv + 0.02, f"{sv:.2f}", ha="center", fontsize=7.0, fontweight="bold", color=C_GREEN)
    ax_a.set_xticks(xa)
    ax_a.set_xticklabels([x[0] for x in items_a], fontsize=7.1, rotation=26, ha="right")
    ax_a.set_ylim(0.0, 1.28)
    ax_a.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_a.set_ylabel("Held-out AUROC (35 contrasts)")
    ax_a.legend(frameon=False, loc="upper left", fontsize=7.2)
    despine(ax_a)

    # Panel B: Empirical Kitagawa-Oaxaca Locus Allocation vs Within-Locus Motif Decomposition (MHCflurry35 + Refits)
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "B")
    ko = pd.read_csv(r8 / "kitagawa_oaxaca_locus_vs_motif_decomposition.csv")
    ko_lolo = ko[(ko["regime"] == "MHCflurry35_Hard_500nM") & (ko["sequence_coding"] == "Isobaric_IL_Harmonized") & (ko["eval_split"] == "LOLO")].iloc[0]
    ko_loso = ko[(ko["regime"] == "MHCflurry35_Hard_500nM") & (ko["sequence_coding"] == "Isobaric_IL_Harmonized") & (ko["eval_split"] == "LOSO")].iloc[0]
    ko_cov = ko[(ko["regime"] == "MHCflurry35_Hard_500nM") & (ko["sequence_coding"] == "Refit_Assigned_Coverage_Only_1df")].iloc[0]
    ko_abc = ko[(ko["regime"] == "MHCflurry35_Hard_500nM") & (ko["sequence_coding"] == "Refit_Conditional_ABC_Given_Assigned_2df")].iloc[0]

    bars_b = [
        ("Assigned cov.", float(ko_cov["auroc_projected_between_locus"]), float(ko_cov["auroc_projected_within_locus"])),
        ("Cond. A/B/C", float(ko_abc["auroc_projected_between_locus"]), float(ko_abc["auroc_projected_within_locus"])),
        ("Between-locus", float(ko_lolo["auroc_projected_between_locus"]), float(ko_loso["auroc_projected_between_locus"])),
        ("Within-locus", float(ko_lolo["auroc_projected_within_locus"]), float(ko_loso["auroc_projected_within_locus"])),
    ]
    xb = np.arange(len(bars_b))
    ax_b.bar(xb - wb / 2, [r[1] for r in bars_b], width=wb, color=C_BLUE, label="LOLO")
    ax_b.bar(xb + wb / 2, [r[2] for r in bars_b], width=wb, color=C_GOLD, label="Strict LOSO")
    ax_b.axhline(0.50, ls="--", lw=0.8, color=C_GRAY)
    ax_b.text(
        0.03,
        0.97,
        f"LOSO gap:\n{ko_loso['pct_linear_gap_from_within_locus']:.1f}% within\n{ko_loso['pct_linear_gap_from_between_locus']:.1f}% between",
        transform=ax_b.transAxes,
        va="top",
        fontsize=7.0,
        bbox=dict(boxstyle="round,pad=0.16", fc="#F8FAFC", ec="#CBD5E1", lw=0.6),
    )
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([r[0] for r in bars_b], fontsize=7.4, rotation=24, ha="right")
    ax_b.set_ylim(0.0, 1.32)
    ax_b.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_b.set_ylabel("Component AUROC (35 contrasts)")
    ax_b.legend(frameon=False, loc="upper right", fontsize=7.2)
    despine(ax_b)

    # Panel C: Frozen External Experimental Bridge (Aranha 2026 PXD085470 HAP1 +/- IFNg & Carre 2025 PXD045265 ECN90 +/- IFNa)
    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "C")
    ext = pd.read_csv(r8 / "external_frozen_decoder_bridge_results.csv")
    ext_items = [
        ("HAP1 WT", "Aranha2026_PXD085470_SpecCount", "HAP1_WT_IFNg_vs_UNT_Pooled", "HAP1_WT_IFNg_vs_UNT_Rep"),
        ("ERAP1-KO", "Aranha2026_PXD085470_SpecCount", "HAP1_ERAP1_KO_IFNg_vs_UNT_Pooled", "HAP1_ERAP1_KO_IFNg_vs_UNT_Rep"),
        ("TAP1-KO", "Aranha2026_PXD085470_SpecCount", "HAP1_TAP1_KO_IFNg_vs_UNT_Pooled", "HAP1_TAP1_KO_IFNg_vs_UNT_Rep"),
        ("TAP2-KO", "Aranha2026_PXD085470_SpecCount", "HAP1_TAP2_KO_IFNg_vs_UNT_Pooled", "HAP1_TAP2_KO_IFNg_vs_UNT_Rep"),
        ("ECN90 +IFN-α", "Carre2025_PXD045265_RepFreq_AuthorHLA", "ECN90_BetaCell_IFNa_vs_Basal_RepFreq351", None),
    ]
    xc = np.arange(len(ext_items))
    w_ext = 0.27
    bet_vals, wit_vals, tot_vals = [], [], []
    for _, d_lbl, c_lbl, rep_pref in ext_items:
        row_e = ext[(ext["dataset"] == d_lbl) & (ext["contrast"] == c_lbl)].iloc[0]
        bet_vals.append(float(row_e["frozen_between_locus_IL"]))
        wit_vals.append(float(row_e["frozen_within_locus_IL"]))
        tot_vals.append(float(row_e["frozen_linear_shift_IL"]))
    ax_c.bar(xc - w_ext, wit_vals, width=w_ext, color=C_BLUE, label="Within-locus $\\hat{\\beta}^\\top(\\bar{w}\\Delta\\mu)$")
    ax_c.bar(xc, bet_vals, width=w_ext, color=C_GOLD, label="Between-locus $\\hat{\\beta}^\\top(\\Delta w\\bar{\\mu})$")
    ax_c.bar(xc + w_ext, tot_vals, width=w_ext, color=C_GREEN, label="Net shift $\\hat{\\beta}^\\top\\Delta q$")
    for xi, (_, _, _, rep_pref) in enumerate(ext_items):
        if rep_pref is not None:
            reps = ext[ext["contrast"].str.startswith(rep_pref)]["frozen_linear_shift_IL"].values
            ax_c.scatter(np.full(len(reps), xi + w_ext) + np.linspace(-0.05, 0.05, len(reps)), reps, s=16, color="#111111", zorder=5)
    ax_c.axhline(0.0, ls="--", lw=0.8, color=C_GRAY)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([x[0] for x in ext_items], fontsize=7.4, rotation=24, ha="right")
    ax_c.set_ylim(-1.8, 13.2)
    ax_c.set_ylabel("Frozen 9-mer motif (I=L) shift")
    ax_c.legend(frameon=False, loc="upper left", fontsize=7.1)
    despine(ax_c)

    # Panel D: SK-Mel-28 Sensitive (wt) vs Resistant (dr) with Mock-Adjusted DiD & Unseen-9mer Transfer
    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "D")
    skp = pd.read_csv(r8 / "skmel28_predictions_with_il_and_unseen_9mer_purge.csv")
    did = pd.read_csv(r8 / "skmel28_mock_adjusted_did.csv")
    did_il = did[did["readout"] == "PosAA180 (I=L Harmonized)"].set_index("time_h")

    order_cats = [
        ("SKMel28_wt", 1, "wt +\nMAPKi", C_GREEN),
        ("SKMel28_dr", 1, "dr +\nMAPKi", C_VERM),
        ("SKMel28_wt", 0, "wt +\nMock", C_SKY),
        ("SKMel28_dr", 0, "dr +\nMock", C_GRAY),
    ]
    xd = np.arange(len(order_cats))
    for xi, (subl, is_drg, _, col) in enumerate(order_cats):
        g = skp[(skp["cell_subline"] == subl) & (skp["is_drug_exposed"] == is_drg)].sort_values("time_h")
        sc_vals = g["score_R2_PosAA180_IL"].values
        ax_d.bar(xi, np.mean(sc_vals), width=0.52, color=col, alpha=0.32, edgecolor=col, lw=1.0)
        for (_, row_g), off_x, mk in zip(g.iterrows(), [-0.09, 0.09], ["o", "^"]):
            ax_d.scatter([xi + off_x], [row_g["score_R2_PosAA180_IL"]], s=38, marker=mk, color=col, edgecolors="#111111", lw=0.6, zorder=5)
    ax_d.scatter([], [], s=32, marker="o", color="#333333", label="24 h")
    ax_d.scatter([], [], s=32, marker="^", color="#333333", label="72 h")
    ax_d.axhline(0.0, ls=":", lw=0.8, color=C_GRAY)
    ax_d.text(
        0.98,
        0.97,
        f"0 shared 4-digit HLAs\nMock-adj. DiD: +{did_il.loc[24, 'mock_adjusted_did']:.2f} (24h),\n+{did_il.loc[72, 'mock_adjusted_did']:.2f} (72h); Unseen-9mer AUROC 1.00",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.18", fc="#F8FAFC", ec="#CBD5E1", lw=0.6),
    )
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([c[2] for c in order_cats], fontsize=7.6)
    ax_d.set_ylim(-11.5, 19.5)
    ax_d.set_ylabel("Full 9-mer motif (I=L) score")
    ax_d.legend(frameon=False, loc="lower left", fontsize=7.4)
    despine(ax_d)

    save_fig(fig, "fig_distributional_sensing_scope")


# Fixed chemistry-grouped ordering of the 19 I=L-harmonized amino acids (for Figure 3B & Figure S6)
CHEM_AA_ORDER = [
    ("L", "L/I", "Hydrophobic", C_BLUE),
    ("V", "V", "Hydrophobic", C_BLUE),
    ("M", "M", "Hydrophobic", C_BLUE),
    ("A", "A", "Hydrophobic", C_BLUE),
    ("F", "F", "Hydrophobic", C_BLUE),
    ("Y", "Y", "Hydrophobic", C_BLUE),
    ("W", "W", "Hydrophobic", C_BLUE),
    ("P", "P", "Proline", C_GOLD),
    ("G", "G", "Polar", C_GREEN),
    ("S", "S", "Polar", C_GREEN),
    ("T", "T", "Polar", C_GREEN),
    ("N", "N", "Polar", C_GREEN),
    ("Q", "Q", "Polar", C_GREEN),
    ("C", "C", "Polar", C_GREEN),
    ("R", "R", "Basic", C_VERM),
    ("K", "K", "Basic", C_VERM),
    ("H", "H", "Basic", C_VERM),
    ("E", "E", "Acidic", C_PURP),
    ("D", "D", "Acidic", C_PURP),
]
CHEM_DIVIDERS = [6.5, 7.5, 13.5, 16.5]


def draw_pssm_heatmap(
    ax: plt.Axes,
    df_logo: pd.DataFrame,
    val_col: str,
    vmax: float,
    annotate_thresh: float,
    fmt: str = "{:+.1f}",
    show_ylabels: bool = True,
    cbar_label: str = "",
) -> None:
    """Render a clean 19 x 9 chemistry-grouped diverging heatmap with selective top-cell annotations."""
    aa_keys = [x[0] for x in CHEM_AA_ORDER]
    aa_disp = [x[1] for x in CHEM_AA_ORDER]
    piv = df_logo.pivot(index="aa", columns="position", values=val_col).reindex(index=aa_keys, columns=range(1, 10))
    mat = piv.values

    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax, interpolation="nearest")
    for y_div in CHEM_DIVIDERS:
        ax.axhline(y_div, color="#334155", lw=0.7, alpha=0.75)

    # Thin chemistry color strip on left edge of heatmap
    for ri, (_, _, _, c_hex) in enumerate(CHEM_AA_ORDER):
        ax.add_patch(
            mpatches.Rectangle(
                (-0.78, ri - 0.5),
                0.28,
                1.0,
                fc=c_hex,
                ec="none",
                clip_on=False,
                zorder=4,
            )
        )

    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[ri, ci]
            if np.isfinite(v) and abs(v) >= annotate_thresh:
                txt_col = "white" if abs(v) > 0.56 * vmax else "#111111"
                ax.text(ci, ri, fmt.format(v), ha="center", va="center", fontsize=6.5, fontweight="bold", color=txt_col)

    ax.set_xticks(range(9))
    ax.set_xticklabels([f"P{k}" for k in range(1, 10)], fontsize=7.0)
    ax.set_yticks(range(len(aa_disp)))
    if show_ylabels:
        ax.set_yticklabels(aa_disp, fontsize=6.7)
        ax.tick_params(axis="y", pad=5.5, length=0)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.044, pad=0.025)
    cbar.ax.tick_params(labelsize=6.6)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=7.1)


def make_fig_anatomy_and_logos() -> None:
    """Figure 3 in manuscript: Visual Representation x Position Matrix, Aligned 19x9 Chemistry-Grouped
    Diverging Shift & Decoder Weight Heatmaps, Per-Position Ablations, and Multi-Allele Supertype Convergence."""
    fig = plt.figure(figsize=(6.80, 7.05))
    gs_outer = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.50, 0.80], hspace=0.40)

    r8 = RES / "exp8_immunity_bridge"
    r9 = RES / "exp9_post_meeting"
    ab = pd.read_csv(r9 / "positional_ablation_and_subsets.csv")
    ab_il = ab[ab["sequence_coding"] == "Isobaric_IL_Harmonized"].set_index("subset_key")
    tr_il = pd.read_csv(r8 / "cross_transfer_all_representations_db_vs_il.csv")
    tr_il_idx = tr_il[tr_il["sequence_coding"] == "Isobaric_IL_Harmonized"].set_index("representation")
    tr_db_idx = tr_il[tr_il["sequence_coding"] == "Database_20AA"].set_index("representation")

    # =========================================================================
    # Row 1: Panel A (Visual Representation x Position P1..P9 Grid + 3-Benchmark Dot Plot)
    # =========================================================================
    gs_top = gs_outer[0].subgridspec(1, 2, width_ratios=[1.16, 1.0], wspace=0.08)
    ax_grid = fig.add_subplot(gs_top[0, 0])
    ax_auc = fig.add_subplot(gs_top[0, 1], sharey=ax_grid)
    add_panel_label(ax_grid, "A", x=-0.34, y=1.04)

    rows_spec = [
        ("Full (P1–P9)", [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         float(ab_il.loc["Full_9mer_Motif_P1_to_P9", "lolo_auroc"]),
         float(ab_il.loc["Full_9mer_Motif_P1_to_P9", "loso_strict_auroc"]),
         float(ab_il.loc["Full_9mer_Motif_P1_to_P9", "skmel28_8run_auroc"])),
        ("Locus (A/B/C)", [4, 4, 4, 4, 4, 4, 4, 4, 4, 0],
         float(ab_il.loc["Locus_Stratified_ABC_Mass_3x180", "lolo_auroc"]),
         float(ab_il.loc["Locus_Stratified_ABC_Mass_3x180", "loso_strict_auroc"]),
         float(ab_il.loc["Locus_Stratified_ABC_Mass_3x180", "skmel28_8run_auroc"])),
        ("Non-Anchor (P1, P3–P8)", [1, 0, 1, 1, 1, 1, 1, 1, 0, 0],
         float(ab_il.loc["All_NonAnchor_P1_P3_to_P8", "lolo_auroc"]),
         float(ab_il.loc["All_NonAnchor_P1_P3_to_P8", "loso_strict_auroc"]),
         float(ab_il.loc["All_NonAnchor_P1_P3_to_P8", "skmel28_8run_auroc"])),
        ("Anchors (P2, P9)", [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
         float(ab_il.loc["HLA_Anchors_P2_P9", "lolo_auroc"]),
         float(ab_il.loc["HLA_Anchors_P2_P9", "loso_strict_auroc"]),
         float(ab_il.loc["HLA_Anchors_P2_P9", "skmel28_8run_auroc"])),
        ("Bulge (P4–P7)", [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
         float(ab_il.loc["Central_Bulge_Residues_P4_to_P7", "lolo_auroc"]),
         float(ab_il.loc["Central_Bulge_Residues_P4_to_P7", "loso_strict_auroc"]),
         float(ab_il.loc["Central_Bulge_Residues_P4_to_P7", "skmel28_8run_auroc"])),
        ("Bulge-4c (P4–P7)", [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
         float(ab_il.loc["Central_Bulge_4Class_Pairs", "lolo_auroc"]),
         float(ab_il.loc["Central_Bulge_4Class_Pairs", "loso_strict_auroc"]),
         float(ab_il.loc["Central_Bulge_4Class_Pairs", "skmel28_8run_auroc"])),
        ("Flank (P1, P3, P8)", [1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
         float(ab_il.loc["Trimming_Secondary_P1_P3_P8", "lolo_auroc"]),
         float(ab_il.loc["Trimming_Secondary_P1_P3_P8", "loso_strict_auroc"]),
         float(ab_il.loc["Trimming_Secondary_P1_P3_P8", "skmel28_8run_auroc"])),
        ("Pooled-AA (avg.)", [3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
         float(ab_il.loc["Position_Pooled_AA20", "lolo_auroc"]),
         float(ab_il.loc["Position_Pooled_AA20", "loso_strict_auroc"]),
         float(ab_il.loc["Position_Pooled_AA20", "skmel28_8run_auroc"])),
        ("Seq-ID (exact 9-mer)", [0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
         float(tr_il_idx.loc["R4_Peptide_Identity", "lolo_auroc"]),
         float(tr_il_idx.loc["R4_Peptide_Identity", "loso_strict_auroc"]),
         0.333333),
        ("Gene-Share", [0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
         float(tr_db_idx.loc["R5_SourceGene_Oracle", "lolo_auroc"]),
         float(tr_db_idx.loc["R5_SourceGene_Oracle", "loso_strict_auroc"]),
         np.nan),
    ]

    tile_palette = {
        0: ("#F1F5F9", "#CBD5E1", ""),
        1: (C_BLUE, "#004C78", "AA"),
        2: (C_GOLD, "#92400E", "4c"),
        3: (C_SKY, "#0369A1", "avg"),
        4: (C_GREEN, "#065F46", "A/B/C"),
        5: (C_PURP, "#701A75", "Seq"),
        6: (C_VERM, "#9A3412", "Gene"),
    }

    n_rows = len(rows_spec)
    y_rows = np.arange(n_rows)[::-1]
    for r_idx, (rlab, tiles, v_lolo, v_loso, v_sk) in enumerate(rows_spec):
        yv = y_rows[r_idx]
        for c_idx, tcode in enumerate(tiles):
            fc, ec, short_txt = tile_palette[tcode]
            x_left = c_idx + (0.22 if c_idx == 9 else 0.0)
            rect = mpatches.FancyBboxPatch(
                (x_left + 0.07, yv - 0.36),
                0.86,
                0.72,
                boxstyle="round,pad=0.03,rounding_size=0.10",
                fc=fc,
                ec=ec,
                lw=0.55,
            )
            ax_grid.add_patch(rect)
            if short_txt and (c_idx == 9 or tcode in (2, 3)):
                ax_grid.text(
                    x_left + 0.50,
                    yv,
                    short_txt,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    fontweight="bold",
                    color="white" if tcode in (1, 4, 5, 6) else "#111111",
                )

        ax_auc.plot([0.50, v_loso], [yv, yv], lw=1.4, color=C_GREEN, alpha=0.45, zorder=1)
        ax_auc.scatter([v_lolo], [yv + 0.14], s=25, marker="o", color=C_BLUE, zorder=4, label="LOLO (16 lin.)" if r_idx == 0 else None)
        ax_auc.scatter([v_loso], [yv], s=29, marker="s", color=C_GREEN, zorder=5, label="Strict LOSO" if r_idx == 0 else None)
        if np.isfinite(v_sk):
            ax_auc.scatter([v_sk], [yv - 0.14], s=27, marker="^", color=C_VERM, zorder=4, label="SK-Mel-28 (0 HLA)" if r_idx == 0 else None)

    ax_grid.axvline(9.11, color="#94A3B8", ls=":", lw=0.75)
    col_xticks = [k + 0.5 for k in range(9)] + [9.72]
    ax_grid.set_xticks(col_xticks)
    ax_grid.set_xticklabels([f"P{k}" for k in range(1, 10)] + ["ID"], fontsize=7.0)
    ax_grid.xaxis.tick_top()
    ax_grid.set_yticks(y_rows)
    ax_grid.set_yticklabels([r[0] for r in rows_spec], fontsize=7.0)
    ax_grid.set_xlim(-0.05, 10.25)
    ax_grid.set_ylim(-0.65, n_rows - 0.25)
    for sp in ("top", "right", "bottom", "left"):
        ax_grid.spines[sp].set_visible(False)
    ax_grid.tick_params(axis="both", length=0)

    leg_patches = [
        mpatches.Patch(fc=C_BLUE, ec="#004C78", label="19-AA (I=L)"),
        mpatches.Patch(fc=C_GREEN, ec="#065F46", label="Per-locus AA"),
        mpatches.Patch(fc=C_GOLD, ec="#92400E", label="4-class pairs"),
        mpatches.Patch(fc=C_SKY, ec="#0369A1", label="Pos. avg."),
    ]
    ax_grid.legend(handles=leg_patches, frameon=False, loc="upper center", bbox_to_anchor=(0.48, -0.02), ncol=4, fontsize=6.5, handlelength=0.9, columnspacing=0.6)

    ax_auc.axvline(0.50, ls="--", lw=0.8, color=C_GRAY)
    ax_auc.set_xlim(0.10, 1.08)
    ax_auc.set_xticks([0.25, 0.50, 0.75, 1.00])
    ax_auc.set_xlabel("Held-out AUROC", fontsize=7.6)
    ax_auc.tick_params(axis="y", left=False, labelleft=False)
    ax_auc.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.48, 1.01),
        ncol=2,
        fontsize=6.5,
        handletextpad=0.25,
        columnspacing=0.65,
    )
    despine(ax_auc)

    # =========================================================================
    # Row 2: Panel B — Aligned 19x9 Chemistry-Grouped Diverging Heatmaps (IFN-g shift & Decoder Weights)
    # =========================================================================
    gs_mid = gs_outer[1].subgridspec(1, 2, wspace=0.26)
    ax_hm1 = fig.add_subplot(gs_mid[0, 0])
    ax_hm2 = fig.add_subplot(gs_mid[0, 1])
    add_panel_label(ax_hm1, "B", x=-0.16, y=1.08)

    logo_df = pd.read_csv(r9 / "pssm_shift_and_decoder_logo_matrices.csv")
    logo_il = logo_df[logo_df["sequence_coding"] == "Isobaric_IL_Harmonized"].copy()

    draw_pssm_heatmap(ax_hm1, logo_il, "mean_dq_ifng_pp", vmax=5.8, annotate_thresh=1.55, fmt="{:+.1f}", show_ylabels=True, cbar_label="IFN-$\\gamma$ $\\Delta q$ (pp)")
    draw_pssm_heatmap(ax_hm2, logo_il, "frozen_beta_std", vmax=0.18, annotate_thresh=0.132, fmt="{:+.2f}", show_ylabels=False, cbar_label="Decoder $\\hat{\\beta}\\cdot\\mathrm{SD}$")

    chem_leg = [
        mpatches.Patch(fc=C_BLUE, label="Hydrophobic"),
        mpatches.Patch(fc=C_GOLD, label="Proline"),
        mpatches.Patch(fc=C_GREEN, label="Polar"),
        mpatches.Patch(fc=C_VERM, label="Basic"),
        mpatches.Patch(fc=C_PURP, label="Acidic"),
    ]
    ax_hm1.legend(
        handles=chem_leg,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(1.12, 1.01),
        ncol=5,
        fontsize=6.5,
        handlelength=0.85,
        columnspacing=0.75,
    )

    # =========================================================================
    # Row 3: Panel C (Per-Position Ablation) + Panel D (Multi-Allele HLA Supertype Convergence Only)
    # =========================================================================
    gs_bot = gs_outer[2].subgridspec(1, 2, width_ratios=[1.04, 1.0], wspace=0.34)
    ax_c = fig.add_subplot(gs_bot[0, 0])
    ax_d = fig.add_subplot(gs_bot[0, 1])
    add_panel_label(ax_c, "C", x=-0.18, y=1.05)
    add_panel_label(ax_d, "D", x=-0.18, y=1.05)

    pos_x = np.arange(1, 10)
    drop_loso = [float(ab_il.loc[f"Drop_P{p}", "loso_strict_auroc"]) for p in pos_x]
    only_loso = [float(ab_il.loc[f"Only_P{p}", "loso_strict_auroc"]) for p in pos_x]
    drop_sk = [float(ab_il.loc[f"Drop_P{p}", "skmel28_8run_auroc"]) for p in pos_x]

    ax_c.axhline(0.788, ls=":", lw=0.9, color=C_GREEN, alpha=0.8)
    ax_c.axhline(0.500, ls="--", lw=0.75, color=C_GRAY)
    ax_c.plot(pos_x, drop_loso, "-s", color=C_GREEN, lw=1.4, ms=4.2, label="Drop $P_k$ (LOSO)")
    ax_c.plot(pos_x, only_loso, "--o", color=C_BLUE, lw=1.2, ms=3.8, label="Only $P_k$ (LOSO)")
    ax_c.plot(pos_x, drop_sk, "-^", color=C_VERM, lw=1.3, ms=4.2, label="Drop $P_k$ (SK-Mel-28)")
    ax_c.set_xticks(pos_x)
    ax_c.set_xticklabels([f"P{p}" for p in pos_x], fontsize=7.2)
    ax_c.set_ylim(0.04, 1.50)
    ax_c.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax_c.set_ylabel("Held-out AUROC", fontsize=7.6)
    ax_c.set_xlabel("Peptide residue position", fontsize=7.6)
    ax_c.legend(frameon=False, loc="upper right", ncol=2, fontsize=6.5, columnspacing=0.6, handletextpad=0.3)
    despine(ax_c)

    # Panel D: Focused Baseline P2/P9 Anchor JSD Contraction (Single Alleles -> Disjoint Loci -> Multi-Allele Donors)
    jsd_df = pd.read_csv(r9 / "baseline_supertype_convergence_jsd.csv").set_index("comparison_level")
    d_items = [
        ("Single\nalleles", float(jsd_df.loc["1_Single_4Digit_Allele_Pairs", "mean_jsd_Anchors_P2P9_bits"]), C_GOLD),
        ("Disjoint\nHLA-B", float(jsd_df.loc["2b_PerLocus_HLA_B_Disjoint_Pairs", "mean_jsd_Anchors_P2P9_bits"]), C_VERM),
        ("Disjoint\nHLA-A", float(jsd_df.loc["2a_PerLocus_HLA_A_Disjoint_Pairs", "mean_jsd_Anchors_P2P9_bits"]), C_SKY),
        ("0-shared\ndonors", float(jsd_df.loc["4_Full_Donor_Repertoire_ZeroShared4DigitHLA", "mean_jsd_Anchors_P2P9_bits"]), C_BLUE),
        ("All 16\ndonors", float(jsd_df.loc["3_Full_Donor_Repertoire_All16Lineages", "mean_jsd_Anchors_P2P9_bits"]), C_GREEN),
    ]
    xd = np.arange(len(d_items))
    vals_d = [x[1] for x in d_items]
    ax_d.bar(xd, vals_d, width=0.60, color=[x[2] for x in d_items], alpha=0.90)
    for xi, vv in enumerate(vals_d):
        ax_d.text(xi, vv + 0.018, f"{vv:.2f}", ha="center", fontsize=6.7, fontweight="bold" if xi in (0, 3) else "normal", color="#111111")
    ax_d.annotate(
        "−39.8% (95.5% share ≥1 supertype)",
        xy=(3, vals_d[3] + 0.06),
        xytext=(2.35, 0.72),
        ha="center",
        fontsize=6.7,
        fontweight="bold",
        color=C_BLUE,
        arrowprops=dict(arrowstyle="->", lw=0.8, color=C_BLUE),
    )
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([x[0] for x in d_items], fontsize=6.7)
    ax_d.set_ylim(0.0, 0.84)
    ax_d.set_ylabel("Baseline P2+P9 JSD (bits)", fontsize=7.6)
    ax_d.set_xlabel("Comparison stratum", fontsize=7.6)
    despine(ax_d)

    save_fig(fig, "fig_representation_anatomy_and_logos", svg=True)


def make_fig2() -> None:
    """Figure 3 in manuscript: MONTE 6-Omic Degradative Flux, PTM Compartment Bifurcation & In Vivo APPM Readout."""
    fig = plt.figure(figsize=(6.55, 5.45))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.42)

    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "A")
    reg = pd.read_csv(RES / "exp2_monte_degradative_flux/across_gene_ptm_regression.tsv", sep="\t")
    preds = ["log2_ub_sites", "log2_phos_sites", "log2_acetyl_sites", "secretory_er_membrane", "log2_spec_abundance"]
    pred_labels = ["K-ε-GG ubiquitylation", "Phosphorylation", "Lysine acetylation", "Secretory / ER / Memb.", "MS spectral abundance"]
    y_idx = np.arange(len(preds))[::-1]
    for offset, out_col, col, lab in [(0.18, "log2_hlai_peptides", C_BLUE, "HLA-I yield"), (-0.18, "log2_hlaii_peptides", C_PURP, "HLA-II yield")]:
        sub = reg[reg["outcome"] == out_col].set_index("predictor").loc[preds]
        ax_a.errorbar(sub["std_beta"], y_idx + offset, xerr=1.96 * sub["hc3_se"], fmt="o", color=col, ms=4.8, lw=1.2, capsize=2.3, label=lab)
    ax_a.axvline(0, ls="--", lw=0.8, color=C_GRAY)
    ax_a.set_yticks(y_idx)
    ax_a.set_yticklabels(pred_labels, fontsize=7.6)
    ax_a.set_xlim(-0.08, 0.44)
    ax_a.set_ylim(-0.6, 5.1)
    ax_a.set_xlabel("Std. partial β (±95% HC3 CI, N=7,594)")
    ax_a.legend(frameon=False, loc="upper right", fontsize=7.5)
    despine(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "B")
    flux = pd.read_csv(RES / "exp2_monte_degradative_flux/within_gene_across_patient_flux.tsv", sep="\t")
    cats = ["HLA-I\n(FT)", "HLA-I\n(noIP)", "HLA-II\n(FT)", "HLA-II\n(noIP)"]
    x_pos = np.arange(len(cats))
    ax_b.errorbar(x_pos - 0.14, flux["beta_prot"].values, yerr=1.96 * flux["se_prot_gene_clustered"].values, fmt="s", color=C_GOLD, ms=5.0, lw=1.2, capsize=2.6, label="Proteome ($\\beta_{\\mathrm{Prot}}$)")
    ax_b.errorbar(x_pos + 0.14, flux["beta_ub"].values, yerr=1.96 * flux["se_ub_gene_clustered"].values, fmt="o", color=C_BLUE, ms=5.0, lw=1.2, capsize=2.6, label="Ubiquitylome ($\\beta_{\\mathrm{Ub}}$)")
    ax_b.axhline(0, ls="--", lw=0.8, color=C_GRAY)
    ax_b.text(
        0.03,
        0.95,
        f"Identical peptide ($N=1{{,}}609$):\nJoint Prot + Ub $R^2 = {100*flux['identical_peptide_r2'].iloc[0]:.2f}\\%$",
        transform=ax_b.transAxes,
        va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.20", fc="#FFF7ED", ec=C_GOLD, lw=0.6),
    )
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(cats)
    ax_b.set_ylim(-0.09, 0.22)
    ax_b.set_ylabel("Within-gene slope (±95% CI)")
    ax_b.legend(frameon=False, loc="lower left", fontsize=7.5)
    despine(ax_b)

    pt = pd.read_csv(RES / "exp2_monte_degradative_flux/monte_patient_appm_and_ligandome.tsv", sep="\t")
    tests = pd.read_csv(RES / "exp2_monte_degradative_flux/monte_appm_readout_tests.tsv", sep="\t")

    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "C")
    ax_c.scatter(pt["appm_core_prot_noIP"], pt["log2_n_hlai_peptides"], s=42, color=C_GREEN, edgecolors="white", lw=0.5, zorder=3)
    c_offsets = {"01024": (-28, -10), "00547": (5, 2), "01416": (5, -5), "01632": (5, -8), "00579": (-28, -9)}
    for _, r in pt.iterrows():
        short_id = r["sample_id"].replace("C3N-", "").replace("C3L-", "")
        ox, oy = c_offsets.get(short_id, (4, 3))
        ax_c.annotate(short_id, (r["appm_core_prot_noIP"], r["log2_n_hlai_peptides"]), xytext=(ox, oy), textcoords="offset points", fontsize=7.4, color="#334155")
    xg = np.linspace(pt["appm_core_prot_noIP"].min() - 0.3, pt["appm_core_prot_noIP"].max() + 0.3, 100)
    s1, i1 = np.polyfit(pt["appm_core_prot_noIP"], pt["log2_n_hlai_peptides"], 1)
    ax_c.plot(xg, s1 * xg + i1, ls="--", lw=1.1, color="#111111")
    row_ap = tests[(tests["hypothesis"] == "APPM core protein vs log2(HLA-I breadth)") & (tests["workflow"] == "noIP")].iloc[0]
    row_ft = tests[(tests["hypothesis"] == "APPM core protein vs log2(HLA-I breadth)") & (tests["workflow"] == "FT")].iloc[0]
    ax_c.text(
        0.04,
        0.95,
        f"noIP: $r = {row_ap['pearson_r']:.2f}$ ($P = {row_ap['pearson_p']:.3f}$)\nFT: $r = {row_ft['pearson_r']:.2f}$ ($P = {row_ft['pearson_p']:.3f}$)",
        transform=ax_c.transAxes,
        va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.20", fc="#F8FAFC", ec="#CBD5E1", lw=0.6),
    )
    ax_c.set_xlim(-3.1, 7.8)
    ax_c.set_ylim(12.22, 13.15)
    ax_c.set_xlabel("Tumor APPM core ($\log_2$ TMT, noIP)")
    ax_c.set_ylabel("HLA-I breadth ($\log_2$ peptides)")
    despine(ax_c)

    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "D")
    ax_d.scatter(pt["erap1_prot_noIP"], pt["mean_peptide_length"], s=42, color=C_BLUE, edgecolors="white", lw=0.5, zorder=3)
    d_offsets = {"00169": (-30, -8), "01632": (5, -4), "02549": (-18, 7), "00579": (-18, -11), "00547": (5, 3), "01016": (5, -8)}
    for _, r in pt.iterrows():
        short_id = r["sample_id"].replace("C3N-", "").replace("C3L-", "")
        ox, oy = d_offsets.get(short_id, (4, 3))
        ax_d.annotate(short_id, (r["erap1_prot_noIP"], r["mean_peptide_length"]), xytext=(ox, oy), textcoords="offset points", fontsize=7.4, color="#334155")
    xg2 = np.linspace(pt["erap1_prot_noIP"].min() - 0.3, pt["erap1_prot_noIP"].max() + 0.3, 100)
    s2, i2 = np.polyfit(pt["erap1_prot_noIP"], pt["mean_peptide_length"], 1)
    ax_d.plot(xg2, s2 * xg2 + i2, ls="--", lw=1.1, color="#111111")
    row_er = tests[(tests["hypothesis"] == "ERAP1 protein vs mean peptide length") & (tests["workflow"] == "noIP")].iloc[0]
    row_lg = tests[(tests["hypothesis"] == "ERAP1 protein vs untrimmed 10-12mer fraction") & (tests["workflow"] == "noIP")].iloc[0]
    ax_d.text(
        0.04,
        0.05,
        f"Mean length: $r = {row_er['pearson_r']:.2f}$ ($P = {row_er['pearson_p']:.3f}$)\n10–12mer: $r = {row_lg['pearson_r']:.2f}$ ($P = {row_lg['pearson_p']:.3f}$)",
        transform=ax_d.transAxes,
        va="bottom",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.20", fc="#F8FAFC", ec="#CBD5E1", lw=0.6),
    )
    ax_d.set_xlim(-3.2, 4.5)
    ax_d.set_ylim(8.84, 9.64)
    ax_d.set_xlabel("Tumor ERAP1 ($\log_2$ TMT, noIP)")
    ax_d.set_ylabel("Mean HLA-I length (residues)")
    despine(ax_d)

    save_fig(fig, "fig2_monte_degradative_flux")


def make_fig3() -> None:
    """Figure 4 in manuscript: Left-Censoring Sensitivity Scenarios, Allotype Routing & Proteolytic Shifts."""
    fig = plt.figure(figsize=(6.55, 5.55))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.44)

    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "A")
    cens = pd.read_csv(RES / "exp3_inflammatory_routing/censoring_bounds_all_programs.tsv", sep="\t")
    e2f = cens[cens["program"] == "E2F_TARGETS"].copy().set_index("background").loc[["ITO57T", "ITO58M", "ITO66T", "UWB1.289"]].reset_index()
    y_p = np.arange(len(e2f))[::-1]
    for yi, (_, r) in enumerate(e2f.iterrows()):
        ypos = y_p[yi]
        lo = min(r["theta_lod_min_bound"], r["theta_survivor"])
        hi = max(r["theta_lod_min_bound"], r["theta_survivor"])
        ax_a.plot([lo, hi], [ypos, ypos], lw=2.8, color=C_LIGHT if r["qualified_10gene"] else "#FDE68A", zorder=1)
        ax_a.scatter([r["theta_survivor"]], [ypos], s=46, color=C_BLUE if r["qualified_10gene"] else C_GRAY, zorder=3, label="Survivor-conditional θ" if yi == 0 else None)
        ax_a.scatter([r["theta_lod_p01_bound"]], [ypos], s=40, marker="D", color=C_GOLD, zorder=4, label="1st-percentile floor scenario" if yi == 0 else None)
        ax_a.scatter([r["theta_lod_min_bound"]], [ypos], s=42, marker="v", color=C_VERM, zorder=4, label="Run-minimum floor scenario" if yi == 0 else None)
    ax_a.axvline(0.0, ls="-", lw=0.8, color="#111111")
    ax_a.axvline(-0.5, ls="--", lw=0.8, color=C_VERM, alpha=0.7)
    ylabs = [f"{r['background']}{'' if r['qualified_10gene'] else ' (<10g)'}\n({100*r['pair_attrition_fraction']:.1f}% drop)" for _, r in e2f.iterrows()]
    ax_a.set_yticks(y_p)
    ax_a.set_yticklabels(ylabs, fontsize=7.5)
    ax_a.set_xlim(-0.62, 1.70)
    ax_a.set_ylim(-0.6, 3.85)
    ax_a.set_xlabel("Source-adjusted E2F contrast ($\log_2$)")
    ax_a.legend(frameon=False, loc="upper right", fontsize=7.2)
    despine(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "B")
    proc = pd.read_csv(RES / "exp3_inflammatory_routing/appm_positive_control_proteomics.tsv", sep="\t")
    genes_show = ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "PSMB8", "PSMB9", "PSMB10", "ERAP1"]
    bgs_show = ["ITO57T", "ITO58M", "ITO66T", "UWB1.289"]
    piv = proc[proc["gene"].isin(genes_show)].pivot(index="background", columns="gene", values="dB").reindex(index=bgs_show, columns=genes_show)
    supp_piv = proc[proc["gene"].isin(genes_show)].pivot(index="background", columns="gene", values="supported").reindex(index=bgs_show, columns=genes_show)
    im = ax_b.imshow(piv.values, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=4.0)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            is_supp = bool(supp_piv.values[i, j]) if pd.notna(supp_piv.values[i, j]) else False
            if np.isfinite(v):
                v_clean = 0.0 if abs(v) < 0.05 else v
                txt = f"{v_clean:.1f}" + ("" if is_supp else "*")
            else:
                txt = "—"
            ax_b.text(j, i, txt, ha="center", va="center", fontsize=7.3, color="white" if (np.isfinite(v) and v > 2.2) else "#111111")
    ax_b.set_xticks(np.arange(len(genes_show)))
    ax_b.set_xticklabels(genes_show, rotation=42, ha="right", fontsize=7.4)
    ax_b.set_yticks(np.arange(len(bgs_show)))
    ax_b.set_yticklabels(bgs_show, fontsize=7.6)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.038, pad=0.03)
    cbar.set_label("Protein $\Delta\log_2$", fontsize=8.2)

    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "C")
    loc = pd.read_csv(RES / "exp3_inflammatory_routing/locus_channel_reallocation.tsv", sep="\t")
    x_bg = np.arange(len(bgs_show))
    w = 0.25
    for k_i, (lname, col) in enumerate([("HLA-A", C_GOLD), ("HLA-B", C_GREEN), ("HLA-C", C_SKY)]):
        vals = [loc[(loc["background"] == b) & (loc["locus"] == lname)]["denovo_to_baseline_ratio"].iloc[0] for b in bgs_show]
        ax_c.bar(x_bg + (k_i - 1) * w, vals, width=w, color=col, label=lname)
    ax_c.axhline(1.0, ls="--", lw=0.8, color=C_GRAY)
    ax_c.set_xticks(x_bg)
    ax_c.set_xticklabels(bgs_show, fontsize=7.6)
    ax_c.set_ylim(0, 3.35)
    ax_c.set_ylabel("De novo / baseline share ratio")
    ax_c.legend(frameon=False, loc="upper left", fontsize=7.5)
    despine(ax_c)

    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "D")
    mot = pd.read_csv(RES / "exp3_inflammatory_routing/proteolytic_motif_shifts_8_backgrounds.tsv", sep="\t")
    y8 = np.arange(len(mot))[::-1]
    h_bar = 0.36
    ax_d.barh(y8 + h_bar / 2, mot["hydro_minus_nonhydro_dPep"], height=h_bar, color=C_BLUE, label="Hydrophobic C-term")
    ax_d.barh(y8 - h_bar / 2, mot["core_pro_minus_nonpro_dPep"], height=h_bar, color=C_VERM, label="Core P4–P7 Proline")
    ax_d.axvline(0.0, ls="-", lw=0.8, color="#111111")
    ax_d.set_yticks(y8)
    ax_d.set_yticklabels(mot["background"], fontsize=7.5)
    ax_d.set_xlim(-0.42, 1.24)
    ax_d.set_ylim(-0.6, 10.1)
    ax_d.set_xlabel("Mean $\Delta\log_2$ shift under IFN-γ")
    ax_d.legend(frameon=False, loc="upper right", fontsize=7.3)
    despine(ax_d)

    save_fig(fig, "fig3_inflammatory_censoring_routing")


def make_fig4() -> None:
    """Figure 5 in manuscript: HAP1 11-Knockout APPM Gatekeeper Atlas & Signal-Peptide Cleavage-Span Sparing."""
    fig = plt.figure(figsize=(6.55, 5.55))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.42)

    ko_df = pd.read_csv(RES / "exp4_hap1_gatekeepers/hap1_11ko_two_margin_summary.tsv", sep="\t").sort_values("overall_retention", ascending=True)
    ldr_df = pd.read_csv(RES / "exp4_hap1_gatekeepers/hap1_leader_signal_peptide_sparing.tsv", sep="\t")

    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "A")
    yk = np.arange(len(ko_df))
    ax_a.barh(yk, 100.0 * ko_df["overall_retention"], height=0.56, color=C_LIGHT, edgecolor="#94A3B8", lw=0.5, label="Overall")
    ax_a.scatter(100.0 * ko_df["ret_HLA_A0201"], yk, s=22, color=C_GOLD, label="HLA-A*02:01", zorder=3)
    ax_a.scatter(100.0 * ko_df["ret_HLA_B4001"], yk, s=22, color=C_GREEN, marker="s", label="HLA-B*40:01", zorder=3)
    ax_a.scatter(100.0 * ko_df["ret_HLA_C0304"], yk, s=24, color=C_SKY, marker="^", label="HLA-C*03:04", zorder=3)
    ylabs = [f"{r['knockout']}{' *' if r['same_acquisition_date_as_wt'] else ''}" for _, r in ko_df.iterrows()]
    ax_a.set_yticks(yk)
    ax_a.set_yticklabels(ylabs, fontsize=7.4)
    ax_a.set_xlabel("Baseline ligand retention (%)")
    ax_a.legend(frameon=False, loc="lower right", fontsize=7.3)
    despine(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "B")
    lengths = [8, 9, 10, 11]
    ko_idx = ko_df.set_index("knockout")
    for kname, col, mk, lab in [
        ("ERAP1", C_BLUE, "o", "ERAP1 KO"),
        ("TAPBP", C_VERM, "s", "TAPBP KO"),
        ("PDIA3", C_PURP, "^", "PDIA3 KO"),
        ("GANAB", C_GOLD, "D", "GANAB KO"),
        ("TAP1", C_GREEN, "v", "TAP1 KO *"),
    ]:
        r = ko_idx.loc[kname]
        ax_b.plot(lengths, [100.0 * r[f"ret_{L}mer"] for L in lengths], marker=mk, lw=1.5, ms=4.8, color=col, label=lab)
    ax_b.set_xticks(lengths)
    ax_b.set_xticklabels(["8-mer", "9-mer", "10-mer", "11-mer"])
    ax_b.set_ylim(5, 110)
    ax_b.set_ylabel("Baseline ligand retention (%)")
    ax_b.set_xlabel("Peptide length")
    ax_b.legend(frameon=False, loc="upper right", fontsize=7.3)
    despine(ax_b)

    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "C")
    tap_l = ldr_df[ldr_df["knockout"].isin(["TAP1", "TAP2", "ERAP1", "TAPBP"])].set_index("knockout").loc[["TAP1", "TAP2", "TAPBP", "ERAP1"]]
    x4 = np.arange(4)
    w = 0.34
    ax_c.bar(x4 - w / 2, 100.0 * tap_l["ret_leader_1_35aa"], width=w, color=C_GREEN, label="N-terminal (≤35 AA)")
    ax_c.bar(x4 + w / 2, 100.0 * tap_l["ret_internal_gt35aa"], width=w, color=C_GRAY, label="Internal (>35 AA)")
    ax_c.annotate(
        "Curated signal span:\n89.2% (TAP1), 83.3% (TAP2)\nvs. 18.9% mature chain",
        xy=(1 - w / 2, 100.0 * tap_l.loc["TAP2", "ret_leader_1_35aa"]),
        xytext=(0.45, 42),
        fontsize=7.3,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#111111"),
        bbox=dict(boxstyle="round,pad=0.18", fc="#ECFDF5", ec=C_GREEN, lw=0.6),
    )
    ax_c.set_xticks(x4)
    ax_c.set_xticklabels(["TAP1*", "TAP2*", "TAPBP", "ERAP1"], fontsize=7.6)
    ax_c.set_ylim(0, 72)
    ax_c.set_ylabel("Ligand retention (%)")
    ax_c.legend(frameon=False, loc="upper right", fontsize=7.3)
    despine(ax_c)

    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "D")
    hm = pd.read_csv(RES / "exp4_hap1_gatekeepers/hallmark50_topology_in_tap1_ko.tsv", sep="\t")
    ax_d.scatter(100.0 * hm["frac_signal_peptide_genes"], 100.0 * hm["gene_mean_retention"], s=26, color=C_BLUE, alpha=0.78, edgecolors="white", lw=0.4)
    hm_highlights = [
        ("EPITHELIAL_MESENCHYMAL_TRANSITION", "EMT", C_GREEN, (-32, 6)),
        ("E2F_TARGETS", "E2F", C_VERM, (9, 8)),
        ("ANGIOGENESIS", "Angio.", C_GOLD, (-40, 14)),
        ("COAGULATION", "Coag.", C_PURP, (-46, 8)),
    ]
    for tag, lab_txt, col, off_xy in hm_highlights:
        sub_r = hm[hm["hallmark_program"] == tag]
        if len(sub_r):
            rx = 100.0 * sub_r["frac_signal_peptide_genes"].iloc[0]
            ry = 100.0 * sub_r["gene_mean_retention"].iloc[0]
            ax_d.scatter([rx], [ry], s=48, color=col, edgecolors="#111111", lw=0.7, zorder=5)
            ax_d.annotate(lab_txt, (rx, ry), xytext=off_xy, textcoords="offset points", fontsize=7.5, fontweight="bold", color=col, arrowprops=dict(arrowstyle="-", lw=0.6, color=col, alpha=0.8))
    xg = np.linspace(0, 100.0 * hm["frac_signal_peptide_genes"].max() + 2, 100)
    sl, ic = np.polyfit(100.0 * hm["frac_signal_peptide_genes"], 100.0 * hm["gene_mean_retention"], 1)
    ax_d.plot(xg, sl * xg + ic, ls="--", lw=1.0, color="#111111")
    pa = pd.read_csv(RES / "exp4_hap1_gatekeepers/tap_ko_topology_matched_reanalysis.tsv", sep="\t")
    t2_emt = pa[(pa["knockout"] == "TAP2") & (pa["program"] == "EPITHELIAL_MESENCHYMAL_TRANSITION")].iloc[0]
    ax_d.text(
        0.04,
        0.05,
        f"49 Hallmarks: $r = 0.45$ ($P = 0.0010$)\nTAP2 EMT: $+4.24 \\to +{t2_emt['topology_matched_mean_pair_u']:.2f}$ matched",
        transform=ax_d.transAxes,
        va="bottom",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.18", fc="#F8FAFC", ec="#CBD5E1", lw=0.6),
    )
    ax_d.set_xlim(-2, 68)
    ax_d.set_ylim(14, 62)
    ax_d.set_xlabel("Curated signal-peptide genes (%)")
    ax_d.set_ylabel("TAP1-KO pathway retention (%)")
    despine(ax_d)

    save_fig(fig, "fig4_hap1_appm_gatekeepers")


def make_fig5() -> None:
    """Figure 6 in manuscript: Cross-Cohort Persistence Prediction with 0-1 AUROC Axes & Exact Method Labels."""
    fig = plt.figure(figsize=(6.55, 5.55))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.42)

    ev = pd.read_csv(RES / "exp5_persistence_prediction/lobo_persistence_metrics.tsv", sep="\t")
    port = pd.read_csv(RES / "exp5_persistence_prediction/lobo_portfolio_curves.tsv", sep="\t")
    coefs = pd.read_csv(RES / "exp5_persistence_prediction/simpson_paradox_coefficients.tsv", sep="\t")

    # Panel A: 0-1 AUROC axis across the 4 held-out backgrounds for Detection Retention
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "A")
    bgs = ["ITO57T", "ITO58M", "ITO66T", "UWB1.289"]
    methods_show = [
        ("Mechanistic_Censored", "Floor-imputed ridge", C_BLUE),
        ("Baseline_MS_Pep0", "Baseline HLA-I MS", C_GREEN),
        ("MHCflurry_Affinity", "Predicted HLA affinity", C_GOLD),
        ("Intracellular_Prot0", "Intracellular proteome", C_VERM),
    ]
    sub_ret = ev[ev["endpoint"] == "retained"]
    x4 = np.arange(len(bgs))
    w = 0.19
    for mi, (mkey, mlab, col) in enumerate(methods_show):
        m_df = sub_ret[sub_ret["method"] == mkey].set_index("held_out_background").loc[bgs]
        ax_a.bar(x4 + (mi - 1.5) * w, m_df["auroc"], width=w, color=col, label=mlab)
    ax_a.axhline(0.5, ls="--", lw=0.8, color=C_GRAY)
    ax_a.set_xticks(x4)
    ax_a.set_xticklabels(["ITO57T", "ITO58M", "ITO66T", "UWB1.289"], fontsize=7.6)
    ax_a.set_ylim(0.0, 1.34)
    ax_a.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_a.set_ylabel("Held-out retention AUROC")
    ax_a.legend(frameon=False, loc="upper right", fontsize=7.1, ncol=1)
    despine(ax_a)

    # Panel B: Portfolio Persistence Curves across Budget K (High-Occupancy Persistence)
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "B")
    p_hp = port[port["endpoint"] == "high_persist"].groupby(["method", "budget_K"])["portfolio_success_rate"].mean().unstack(0)
    ks = p_hp.index.values
    for mkey, mlab, col, mk in [
        ("Mechanistic_Censored", "Floor-imputed ridge", C_BLUE, "o"),
        ("Baseline_MS_Pep0", "Baseline HLA-I MS", C_GREEN, "s"),
        ("Virtual_Prot0_plus_Aff", "Prot0 + affinity + motif", C_PURP, "^"),
        ("MHCflurry_Affinity", "Predicted HLA affinity", C_GOLD, "D"),
        ("Intracellular_Prot0", "Intracellular proteome", C_VERM, "v"),
    ]:
        ax_b.plot(ks, 100.0 * p_hp[mkey].values, marker=mk, lw=1.5, ms=4.5, color=col, label=mlab)
    ax_b.set_xlabel("Selected baseline portfolio size ($K$)")
    ax_b.set_ylabel("Sustained high display (%)")
    ax_b.set_ylim(0, 105)
    ax_b.legend(frameon=False, loc="lower right", fontsize=7.1)
    despine(ax_b)

    # Panel C: Cross-Platform Simpson's Paradox in Unstratified vs Within-Cohort Standardized Weights
    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "C")
    cd = coefs[coefs["endpoint"] == "durable"].set_index("feature")
    f_order = ["aff", "locus_B", "full_detect0", "cterm_acidic", "prot0", "pep0"]
    f_labs = ["Predicted affinity", "HLA-B routing", "Baseline 3/3 detect.", "Acidic C-terminus", "Intracellular Prot0", "Baseline MS level"]
    yc = np.arange(len(f_order))[::-1]
    h = 0.34
    ax_c.barh(yc + h / 2, cd.loc[f_order, "within_cohort_censored_coef"], height=h, color=C_BLUE, label="Floor-imputed ridge")
    ax_c.barh(yc - h / 2, cd.loc[f_order, "unstratified_pooled_logit_coef"], height=h, color=C_VERM, alpha=0.82, label="Unstratified pooled logit")
    ax_c.axvline(0.0, ls="-", lw=0.8, color="#111111")
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels(f_labs, fontsize=7.5)
    ax_c.set_xlim(-0.18, 0.28)
    ax_c.set_ylim(-1.75, 5.55)
    ax_c.set_xlabel("Std. durability coefficient")
    ax_c.legend(frameon=False, loc="lower right", fontsize=7.2)
    despine(ax_c)

    # Panel D: 0-1 AUROC Comparison Across All 3 Persistence Endpoints (showing Affinity 0.635 lead on Durability!)
    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "D")
    mean_ev = ev.groupby(["endpoint", "method"])["auroc"].mean().unstack(0)
    ends = [("retained", "Retention"), ("high_persist", "High-\noccupancy"), ("durable", "Fold-change\ndurability")]
    xd = np.arange(len(ends))
    wd = 0.19
    for mi, (mkey, mlab, col) in enumerate(methods_show):
        vals_d = [float(mean_ev.loc[mkey, ek]) for ek, _ in ends]
        ax_d.bar(xd + (mi - 1.5) * wd, vals_d, width=wd, color=col, label=mlab)
        if mkey == "MHCflurry_Affinity":
            ax_d.text(xd[2] + (mi - 1.5) * wd, vals_d[2] + 0.02, f"{vals_d[2]:.2f}", ha="center", fontsize=7.2, fontweight="bold", color="#92400E")
    ax_d.axhline(0.5, ls="--", lw=0.8, color=C_GRAY)
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([el for _, el in ends], fontsize=7.6)
    ax_d.set_ylim(0.0, 1.34)
    ax_d.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_d.set_ylabel("Mean LOBO AUROC (4 cohorts)")
    ax_d.legend(frameon=False, loc="upper right", fontsize=7.1)
    despine(ax_d)

    save_fig(fig, "fig5_persistence_prediction")


def run(res_root: Path, fig_dir: Path) -> dict:
    global RES, FIG_DIR
    RES = Path(res_root)
    FIG_DIR = Path(fig_dir)
    make_fig1()
    make_graphical_abstract()
    make_fig_distributional()
    make_fig_anatomy_and_logos()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    manifest = {
        "figures": sorted([p.name for p in FIG_DIR.glob("*.pdf")]),
        "svg_figures": sorted([p.name for p in FIG_DIR.glob("*.svg")]),
        "png_previews": sorted([p.name for p in FIG_DIR.glob("*.png")]),
    }
    (FIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    from presentation_state.config import resolve_config
    cfg = resolve_config()
    print(json.dumps(run(cfg.results_dir, cfg.fig_dir), indent=2))
