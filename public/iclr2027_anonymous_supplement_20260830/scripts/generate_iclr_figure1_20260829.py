#!/usr/bin/env python3
"""Generate editable SVG variants and a paper-ready PDF of ICLR Figure 1."""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/_runtime_cache/matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})


OUT = ROOT / "paper" / "figures"
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
BLUE = "#0072B2"
ORANGE = "#A65E00"
GREEN = "#007A5A"
RED = "#D55E00"
DARK = "#333333"
MID = "#666666"
LIGHT = "#F3F5F7"


def text(x, y, value, size=2.4, weight="normal", fill=DARK, anchor="start", italic=False):
    style = ' font-style="italic"' if italic else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
        f'font-size="{size:.1f}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{style}>{value}</text>'
    )


def rect(x, y, w, h, fill="#FFFFFF", stroke="#B5BCC3", width=0.35, radius=1.5, dash=None):
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width:.2f}" '
        f'rx="{radius:.1f}" ry="{radius:.1f}"{dashed}/>'
    )


def line(x1, y1, x2, y2, stroke=DARK, width=0.45, dash=None, arrow=False):
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.2f}"{dashed}{marker}/>'
    )


def path(d, stroke=DARK, width=0.45, fill="none", arrow=False, dash=None):
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{width:.2f}"{dashed}{marker}/>'
    )


def header(width, height, warm=False):
    bg = "#FAF8F4" if warm else "#FFFFFF"
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="{width:.1f}mm" height="{height:.1f}mm"
     viewBox="0 0 {width:.1f} {height:.1f}" version="1.1">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="4" markerHeight="4" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{DARK}"/>
    </marker>
  </defs>
  <rect id="canvas-background" x="0.0" y="0.0" width="{width:.1f}" height="{height:.1f}" fill="{bg}"/>
'''


def latent_content(x, y, w, h, prefix):
    parts = [f'<g id="{prefix}-latent-content">']
    parts.append(text(x + w / 2, y + 5.0, "Raw-q ambiguity", 2.3, "bold", anchor="middle"))
    for idx, (cx, color, angle) in enumerate(((x + 8.0, BLUE, 0), (x + w - 8.0, ORANGE, 28))):
        parts.append(f'<g id="{prefix}-chart-{idx + 1}">')
        parts.append(rect(cx - 5.5, y + 9.0, 11.0, 12.0, "#FFFFFF", color, 0.45, 0.8))
        parts.append(line(cx - 4.5, y + 19.0, cx + 4.5, y + 11.0, color, 0.25))
        parts.append(line(cx - 4.5, y + 11.0, cx + 4.5, y + 19.0, color, 0.25))
        parts.append(f'<circle cx="{cx:.1f}" cy="{y + 14.5:.1f}" r="1.1" fill="{color}"/>')
        parts.append(text(cx, y + 24.5, f"q chart {idx + 1}", 1.9, fill=color, anchor="middle"))
        parts.append(f'</g>')
    parts.append(path(f"M {x + 5.0:.1f} {y + 34.0:.1f} C {x + 10.0:.1f} {y + 25.0:.1f}, {x + 18.0:.1f} {y + 42.0:.1f}, {x + w - 5.0:.1f} {y + 30.0:.1f}", BLUE, 0.6))
    parts.append(path(f"M {x + 5.0:.1f} {y + 34.0:.1f} C {x + 10.0:.1f} {y + 25.0:.1f}, {x + 18.0:.1f} {y + 42.0:.1f}, {x + w - 5.0:.1f} {y + 30.0:.1f}", ORANGE, 0.25, dash="1.4,1.0"))
    parts.append(text(x + w / 2, y + h - 6.0, "identical physical curve", 2.0, fill=MID, anchor="middle"))
    parts.append(text(x + w / 2, y + h - 2.5, "q depends on chart", 2.1, "bold", RED, "middle"))
    parts.append('</g>')
    return "\n".join(parts)


def quotient_content(x, y, w, h, prefix):
    parts = [f'<g id="{prefix}-quotient-content">']
    parts.append(text(x + w / 2, y + 5.0, "Response quotient", 2.3, "bold", anchor="middle"))
    for i in range(5):
        px = x + 5.0 + i * (w - 10.0) / 4.0
        parts.append(line(px, y + 13.0, px, y + 25.0, "#B5BCC3", 0.25, dash="0.8,0.8"))
        parts.append(f'<circle cx="{px:.1f}" cy="{y + 22.0:.1f}" r="0.8" fill="{GREEN}"/>')
    parts.append(path(f"M {x + 4.0:.1f} {y + 22.0:.1f} C {x + 9.0:.1f} {y + 13.0:.1f}, {x + w - 10.0:.1f} {y + 30.0:.1f}, {x + w - 4.0:.1f} {y + 17.0:.1f}", GREEN, 0.65))
    parts.append(text(x + w / 2, y + 30.0, "fixed physical probes P", 2.0, fill=GREEN, anchor="middle"))
    parts.append(rect(x + 5.0, y + 34.0, w - 10.0, 10.0, "#E7F5EF", GREEN, 0.4, 3.0))
    parts.append(text(x + w / 2, y + 40.2, "[f(P)]", 3.0, "bold", GREEN, "middle"))
    parts.append(text(x + w / 2, y + h - 2.5, "one response object", 2.0, fill=MID, anchor="middle"))
    parts.append('</g>')
    return "\n".join(parts)


def coordinates_content(x, y, w, h, prefix):
    parts = [f'<g id="{prefix}-coordinates-content">']
    parts.append(text(x + w / 2, y + 5.0, "Named coordinates", 2.3, "bold", anchor="middle"))
    colors = (BLUE, ORANGE, GREEN)
    labels = ("level", "slope", "curve")
    for i, (color, label) in enumerate(zip(colors, labels)):
        bx = x + 4.0 + i * (w - 8.0) / 3.0
        bw = (w - 11.0) / 3.0
        parts.append(rect(bx, y + 10.0, bw, 14.0, "#FFFFFF", color, 0.4, 1.0))
        if i == 0:
            parts.append(line(bx + 2.0, y + 18.0, bx + bw - 2.0, y + 18.0, color, 0.55))
        elif i == 1:
            parts.append(line(bx + 2.0, y + 21.0, bx + bw - 2.0, y + 13.0, color, 0.55))
        else:
            parts.append(path(f"M {bx + 2.0:.1f} {y + 20.0:.1f} Q {bx + bw / 2:.1f} {y + 11.0:.1f} {bx + bw - 2.0:.1f} {y + 20.0:.1f}", color, 0.55))
        parts.append(text(bx + bw / 2, y + 28.5, label, 1.8, fill=color, anchor="middle"))
    parts.append(rect(x + 4.0, y + 33.0, w - 8.0, 8.5, "#EAF3F8", BLUE, 0.35, 1.5))
    parts.append(text(x + w / 2, y + 38.5, "support-only re-q", 1.9, "bold", BLUE, "middle"))
    parts.append(rect(x + 4.0, y + 44.5, w - 8.0, 8.5, "#FFF4DA", ORANGE, 0.35, 1.5, "1.2,0.8"))
    parts.append(text(x + w / 2, y + 50.0, "rank-poor: decoder prior", 1.7, fill=ORANGE, anchor="middle"))
    parts.append(text(x + w / 2, y + h - 2.5, "c = (level, slope, curve)", 1.75, fill=MID, anchor="middle"))
    parts.append('</g>')
    return "\n".join(parts)


def evidence_content(x, y, w, h, prefix):
    parts = [f'<g id="{prefix}-evidence-content">']
    parts.append(text(x + w / 2, y + 5.0, "Temporal transfer", 2.3, "bold", anchor="middle"))
    box_h = 16.0
    for i, (label, metric, color) in enumerate((("ZT · quadratic", "R² = 0.9888", BLUE), ("Vapor P · thermo", "P-space R² = 0.9996", GREEN))):
        by = y + 9.0 + i * 21.0
        parts.append(rect(x + 3.0, by, w - 6.0, box_h, LIGHT, "#A6AFB8", 0.3, 1.0, "1.6,1.0"))
        parts.append(path(f"M {x + 6.0:.1f} {by + 12.5:.1f} C {x + 10.0:.1f} {by + 3.5:.1f}, {x + 13.5:.1f} {by + 15.0:.1f}, {x + 17.0:.1f} {by + 6.0:.1f}", color, 0.65))
        parts.append(f'<circle cx="{x + 8.0:.1f}" cy="{by + 10.0:.1f}" r="0.8" fill="{ORANGE}"/>')
        parts.append(f'<circle cx="{x + 14.0:.1f}" cy="{by + 8.0:.1f}" r="0.8" fill="{ORANGE}"/>')
        parts.append(text(x + w - 4.0, by + 6.5, label, 1.8, "bold", anchor="end"))
        parts.append(text(x + w - 4.0, by + 12.0, metric, 2.1, "bold", color, "end"))
        parts.append(text(x + w - 4.0, by + 15.0, "support → hidden query", 1.6, fill=MID, anchor="end"))
    parts.append(rect(x + 3.0, y + h - 11.0, w - 6.0, 7.0, "#F0F0F0", "#B5BCC3", 0.3, 1.0))
    parts.append(text(x + w / 2, y + h - 7.5, "accuracy is baseline-dependent", 1.6, fill=MID, anchor="middle"))
    parts.append(text(x + w / 2, y + h - 5.0, "expression is compact and named", 1.6, fill=MID, anchor="middle"))
    parts.append('</g>')
    return "\n".join(parts)


def panel(label, x, y, w, h, content, prefix, fill="#FFFFFF", stroke="#C5CBD1"):
    return "\n".join((
        f'<g id="{prefix}-panel-{label}" inkscape:groupmode="layer" inkscape:label="Panel {label}">',
        text(x, y - 2.5, label, 3.5, "bold", "#000000"),
        rect(x, y, w, h, fill, stroke, 0.35, 1.8),
        content(x, y, w, h, prefix),
        '</g>',
    ))


def variant_a():
    parts = [header(183.0, 76.0)]
    specs = (
        ("a", 4.0, 10.0, 28.0, 58.0, latent_content),
        ("b", 50.0, 10.0, 26.0, 58.0, quotient_content),
        ("c", 94.0, 10.0, 31.0, 58.0, coordinates_content),
        ("d", 143.0, 10.0, 36.0, 58.0, evidence_content),
    )
    for label, x, y, w, h, content in specs:
        parts.append(panel(label, x, y, w, h, content, "variant-a"))
    parts.append('<g id="variant-a-arrows" inkscape:groupmode="layer" inkscape:label="Flow arrows">')
    parts.append(line(35.0, 47.7, 47.0, 47.7, DARK, 0.55, arrow=True))
    parts.append(line(79.0, 47.7, 91.0, 47.7, DARK, 0.55, arrow=True))
    parts.append(line(128.0, 47.7, 140.0, 47.7, DARK, 0.55, arrow=True))
    parts.append('</g>\n</svg>')
    return "\n".join(parts)


def variant_b():
    parts = [header(183.0, 154.0, warm=True)]
    specs = (
        ("a", 5.0, 10.0, 75.0, 58.0, latent_content, "#FFFFFF"),
        ("b", 103.0, 10.0, 75.0, 58.0, quotient_content, "#EFF8F4"),
        ("c", 5.0, 88.0, 75.0, 58.0, coordinates_content, "#EEF5F9"),
        ("d", 103.0, 88.0, 75.0, 58.0, evidence_content, "#FFFFFF"),
    )
    for label, x, y, w, h, content, fill in specs:
        parts.append(panel(label, x, y, w, h, content, "variant-b", fill))
    parts.append('<g id="variant-b-arrows" inkscape:groupmode="layer" inkscape:label="Flow arrows">')
    parts.append(line(83.0, 37.3, 100.0, 37.3, DARK, 0.55, arrow=True))
    parts.append(path("M 140.5 71.0 Q 91.5 78.0 42.5 85.0", DARK, 0.55, arrow=True))
    parts.append(line(83.0, 125.7, 100.0, 125.7, DARK, 0.55, arrow=True))
    parts.append('</g>\n</svg>')
    return "\n".join(parts)


def render_svg_pdf(svg_path, pdf_path):
    root = ET.parse(svg_path).getroot()
    width, height = map(float, root.attrib["viewBox"].split()[2:])
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    scale = 72.0 / 25.4
    ns = "{http://www.w3.org/2000/svg}"
    for elem in root.iter():
        tag = elem.tag.removeprefix(ns)
        if tag == "rect":
            x, y = float(elem.attrib["x"]), float(elem.attrib["y"])
            w, h = float(elem.attrib["width"]), float(elem.attrib["height"])
            patch = FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={float(elem.attrib.get('rx', 0))}",
                facecolor=elem.attrib.get("fill", "none"),
                edgecolor=elem.attrib.get("stroke", "none"),
                linewidth=float(elem.attrib.get("stroke-width", 0)) * scale,
            )
            if "stroke-dasharray" in elem.attrib:
                patch.set_linestyle((0, tuple(float(v) * scale for v in elem.attrib["stroke-dasharray"].split(","))))
            ax.add_patch(patch)
        elif tag == "line":
            x1, y1 = float(elem.attrib["x1"]), float(elem.attrib["y1"])
            x2, y2 = float(elem.attrib["x2"]), float(elem.attrib["y2"])
            color = elem.attrib.get("stroke", DARK)
            linewidth = float(elem.attrib.get("stroke-width", 0.45)) * scale
            if "marker-end" in elem.attrib:
                ax.add_patch(FancyArrowPatch(
                    (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=5.5,
                    color=color, linewidth=linewidth, shrinkA=0, shrinkB=0,
                ))
            else:
                plotted, = ax.plot((x1, x2), (y1, y2), color=color, linewidth=linewidth)
                if "stroke-dasharray" in elem.attrib:
                    plotted.set_dashes(tuple(float(v) * scale for v in elem.attrib["stroke-dasharray"].split(",")))
        elif tag == "circle":
            ax.add_patch(Circle(
                (float(elem.attrib["cx"]), float(elem.attrib["cy"])),
                float(elem.attrib["r"]),
                facecolor=elem.attrib.get("fill", "none"),
                edgecolor=elem.attrib.get("stroke", "none"),
                linewidth=float(elem.attrib.get("stroke-width", 0)) * scale,
            ))
        elif tag == "path" and "d" in elem.attrib and "stroke" in elem.attrib:
            tokens = re.findall(r"[MCQ]|-?\d+(?:\.\d+)?", elem.attrib["d"])
            vertices, codes = [], []
            idx = 0
            while idx < len(tokens):
                command = tokens[idx]
                idx += 1
                if command == "M":
                    vertices.append((float(tokens[idx]), float(tokens[idx + 1])))
                    codes.append(MplPath.MOVETO)
                    idx += 2
                elif command == "C":
                    for _ in range(3):
                        vertices.append((float(tokens[idx]), float(tokens[idx + 1])))
                        codes.append(MplPath.CURVE4)
                        idx += 2
                elif command == "Q":
                    for _ in range(2):
                        vertices.append((float(tokens[idx]), float(tokens[idx + 1])))
                        codes.append(MplPath.CURVE3)
                        idx += 2
            patch = PathPatch(
                MplPath(vertices, codes),
                facecolor=elem.attrib.get("fill", "none"),
                edgecolor=elem.attrib.get("stroke", DARK),
                linewidth=float(elem.attrib.get("stroke-width", 0.45)) * scale,
            )
            if "stroke-dasharray" in elem.attrib:
                patch.set_linestyle((0, tuple(float(v) * scale for v in elem.attrib["stroke-dasharray"].split(","))))
            ax.add_patch(patch)
        elif tag == "text":
            anchor = {"start": "left", "middle": "center", "end": "right"}[elem.attrib.get("text-anchor", "start")]
            ax.text(
                float(elem.attrib["x"]), float(elem.attrib["y"]), elem.text or "",
                fontsize=float(elem.attrib.get("font-size", 2.4)) * scale,
                fontfamily="DejaVu Sans",
                fontweight=elem.attrib.get("font-weight", "normal"),
                fontstyle=elem.attrib.get("font-style", "normal"),
                color=elem.attrib.get("fill", DARK),
                ha=anchor, va="baseline",
            )
    fig.savefig(pdf_path, format="pdf", dpi=300, metadata={"Creator": "latent_variable_search", "CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    variant_a_path = OUT / "figure1_variant_a.svg"
    variant_a_path.write_text(variant_a(), encoding="utf-8")
    (OUT / "figure1_variant_b.svg").write_text(variant_b(), encoding="utf-8")
    render_svg_pdf(variant_a_path, OUT / "figure1_variant_a.pdf")


if __name__ == "__main__":
    main()
