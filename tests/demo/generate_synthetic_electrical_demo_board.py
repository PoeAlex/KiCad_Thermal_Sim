#!/usr/bin/env python3
"""Generate a small KiCad board that demonstrates ThermalSim electrical regressions.

The board is intentionally synthetic.  It contains four labelled cases:
A. 8-turn, 0.25 mm / 0.45 mm-pitch same-net square spiral.
B. 15 um endpoint gap (positive snap case) plus a 50 um negative control.
C. Tracks ending in the void of a C-shaped zone bbox while crossing filled copper.
D. Zone/via overlap where the via centre is in the zone void but annular copper overlaps.
"""
from pathlib import Path
import uuid

OUT = Path(__file__).with_name("synthetic_electrical_connectivity_demo.kicad_pcb")
NS = uuid.UUID("4f1ea6a6-fbf0-4c4a-b890-14b7cf14fd58")


def uid(name):
    return str(uuid.uuid5(NS, name))


def fmt(v):
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def segment(name, start, end, width, layer, net):
    return f'''\t(segment\n\t\t(start {fmt(start[0])} {fmt(start[1])})\n\t\t(end {fmt(end[0])} {fmt(end[1])})\n\t\t(width {fmt(width)})\n\t\t(layer "{layer}")\n\t\t(net "{net}")\n\t\t(uuid "{uid(name)}")\n\t)\n'''


def pad_footprint(name, ref, at, layer, net, size=1.2):
    return f'''\t(footprint "ThermalSimDemo:Terminal"\n\t\t(layer "F.Cu")\n\t\t(uuid "{uid(name+'-fp')}")\n\t\t(at {fmt(at[0])} {fmt(at[1])})\n\t\t(property "Reference" "{ref}"\n\t\t\t(at 0 -1.7 0)\n\t\t\t(layer "F.SilkS")\n\t\t\t(uuid "{uid(name+'-ref')}")\n\t\t\t(effects (font (size 1 1) (thickness 0.15)))\n\t\t)\n\t\t(property "Value" "{net}"\n\t\t\t(at 0 1.7 0)\n\t\t\t(layer "F.Fab")\n\t\t\t(hide yes)\n\t\t\t(uuid "{uid(name+'-val')}")\n\t\t\t(effects (font (size 1 1) (thickness 0.15)))\n\t\t)\n\t\t(attr exclude_from_pos_files exclude_from_bom allow_missing_courtyard)\n\t\t(pad "1" smd rect\n\t\t\t(at 0 0)\n\t\t\t(size {fmt(size)} {fmt(size)})\n\t\t\t(layers "{layer}")\n\t\t\t(net "{net}")\n\t\t\t(pinfunction "1")\n\t\t\t(pintype "passive")\n\t\t\t(uuid "{uid(name+'-pad')}")\n\t\t)\n\t\t(embedded_fonts no)\n\t)\n'''


def via(name, at, size, drill, net):
    return f'''\t(via\n\t\t(at {fmt(at[0])} {fmt(at[1])})\n\t\t(size {fmt(size)})\n\t\t(drill {fmt(drill)})\n\t\t(layers "F.Cu" "B.Cu")\n\t\t(remove_unused_layers no)\n\t\t(keep_end_layers yes)\n\t\t(net "{net}")\n\t\t(uuid "{uid(name)}")\n\t)\n'''


def zone(name, pts, layer, net):
    pts_s = " ".join(f"(xy {fmt(x)} {fmt(y)})" for x, y in pts)
    return f'''\t(zone\n\t\t(net "{net}")\n\t\t(layers "{layer}")\n\t\t(uuid "{uid(name)}")\n\t\t(hatch edge 0.5)\n\t\t(connect_pads yes (clearance 0))\n\t\t(min_thickness 0.1)\n\t\t(fill yes\n\t\t\t(thermal_gap 0.3)\n\t\t\t(thermal_bridge_width 0.3)\n\t\t\t(island_removal_mode 0)\n\t\t)\n\t\t(polygon\n\t\t\t(pts {pts_s})\n\t\t)\n\t\t(filled_polygon\n\t\t\t(layer "{layer}")\n\t\t\t(pts {pts_s})\n\t\t)\n\t)\n'''


def text(name, txt, at, size=1.3, thickness=0.18, justify=""):
    just = f"\n\t\t\t(justify {justify})" if justify else ""
    return f'''\t(gr_text "{txt}"\n\t\t(at {fmt(at[0])} {fmt(at[1])} 0)\n\t\t(layer "F.SilkS")\n\t\t(uuid "{uid(name)}")\n\t\t(effects\n\t\t\t(font (size {fmt(size)} {fmt(size)}) (thickness {fmt(thickness)})){just}\n\t\t)\n\t)\n'''


def edge(name, p1, p2):
    return f'''\t(gr_line\n\t\t(start {fmt(p1[0])} {fmt(p1[1])})\n\t\t(end {fmt(p2[0])} {fmt(p2[1])})\n\t\t(stroke (width 0.05) (type default))\n\t\t(layer "Edge.Cuts")\n\t\t(uuid "{uid(name)}")\n\t)\n'''


body = []

# Board outline and title.
for i, (a, b) in enumerate([
    ((2, 2), (118, 2)), ((118, 2), (118, 74)),
    ((118, 74), (2, 74)), ((2, 74), (2, 2)),
]):
    body.append(edge(f"edge-{i}", a, b))
body += [
    text("title", "ThermalSim electrical connectivity regression demo", (60, 5), 1.6),
    text("subtitle", "Run Electrical Preview at 0.05 mm. Press B first to refill zones.", (60, 7.5), 1.0),
]

# Case A: continuous 8-turn square spiral, 0.25 mm trace, 0.45 mm pitch.
net_a = "DEMO_A_8TURN_SPIRAL"
L, R, T, B = 8.0, 36.0, 12.0, 40.0
pitch = 0.45
current = (L, B)
spiral_segments = []
for turn in range(8):
    targets = [(L, T), (R, T), (R, B), (L + pitch, B)]
    for side, target in enumerate(targets):
        spiral_segments.append((current, target))
        current = target
        if side == 0:
            L += pitch
        elif side == 1:
            T += pitch
        elif side == 2:
            R -= pitch
        elif side == 3:
            B -= pitch
body.append(text("case-a-title", "A  8-turn same-net spiral", (22, 9), 1.25))
body.append(text("case-a-note", "0.25 mm trace / 0.45 mm pitch / 0.20 mm copper gap", (22, 42.5), 0.85))
body.append(pad_footprint("a-src", "A1", spiral_segments[0][0], "F.Cu", net_a, 1.0))
body.append(pad_footprint("a-sink", "A2", current, "F.Cu", net_a, 1.0))
for i, (p1, p2) in enumerate(spiral_segments):
    body.append(segment(f"a-seg-{i}", p1, p2, 0.25, "F.Cu", net_a))

# Case B: endpoint snap and negative control.
net_b = "DEMO_B_15UM_SNAP"
body.append(text("case-b-title", "B  Endpoint snap", (74, 11), 1.25))
body.append(text("case-b-note1", "15 um gap -> accepted endpoint snap", (74, 15), 0.9))
body.append(pad_footprint("b-src", "B1", (49, 18), "F.Cu", net_b, 1.0))
body.append(pad_footprint("b-sink", "B2", (67, 18), "F.Cu", net_b, 1.0))
body.append(segment("b-left", (49, 18), (58.000, 18), 0.25, "F.Cu", net_b))
body.append(segment("b-right", (58.015, 18), (67, 18), 0.25, "F.Cu", net_b))
net_bn = "DEMO_B_50UM_OPEN"
body.append(text("case-b-note2", "50 um gap -> negative control (must stay open)", (74, 23), 0.9))
body.append(segment("bn-left", (49, 26), (58.000, 26), 0.25, "F.Cu", net_bn))
body.append(segment("bn-right", (58.050, 26), (67, 26), 0.25, "F.Cu", net_bn))

# Case C: track crosses a C-shaped zone but ends in the zone bbox void.
net_c = "DEMO_C_TRACK_ZONE"
c_pts = [(45, 44), (68, 44), (68, 48), (53, 48), (53, 60), (68, 60), (68, 64), (45, 64)]
body.append(text("case-c-title", "C  Track -> zone raster overlap", (56.5, 42), 1.25))
body.append(text("case-c-note", "Both track endpoints are in the zone bbox void", (56.5, 66), 0.85))
body.append(zone("c-zone", c_pts, "F.Cu", net_c))
body.append(pad_footprint("c-left-pad", "C1", (40, 54), "F.Cu", net_c, 1.2))
body.append(pad_footprint("c-top-pad", "C2", (56, 39), "F.Cu", net_c, 1.2))
body.append(segment("c-left-track", (40, 54), (58, 54), 0.6, "F.Cu", net_c))
body.append(segment("c-top-track", (56, 39), (56, 54), 0.6, "F.Cu", net_c))

# Case D: via centre lies in C-zone void, but via copper overlaps zone edge.
net_d = "DEMO_D_ZONE_VIA"
d_pts = [(82, 44), (100, 44), (100, 48), (90, 48), (90, 60), (100, 60), (100, 64), (82, 64)]
body.append(text("case-d-title", "D  Zone -> via raster overlap", (101, 42), 1.25))
body.append(text("case-d-note", "Via centre in void; annular copper overlaps zone", (101, 66), 0.85))
body.append(zone("d-zone", d_pts, "F.Cu", net_d))
body.append(pad_footprint("d-src-pad", "D1", (77, 54), "F.Cu", net_d, 1.2))
body.append(segment("d-src-track", (77, 54), (94, 54), 0.6, "F.Cu", net_d))
body.append(via("d-via", (90.35, 54), 1.2, 0.5, net_d))
body.append(pad_footprint("d-sink-pad", "D2", (112, 54), "In1.Cu", net_d, 1.2))
body.append(segment("d-in1-track", (90.35, 54), (112, 54), 0.6, "In1.Cu", net_d))

header = '''(kicad_pcb\n\t(version 20260206)\n\t(generator "pcbnew")\n\t(generator_version "10.0")\n\t(general\n\t\t(thickness 1.6)\n\t\t(legacy_teardrops no)\n\t)\n\t(paper "A4")\n\t(layers\n\t\t(0 "F.Cu" signal)\n\t\t(4 "In1.Cu" signal)\n\t\t(6 "In2.Cu" signal)\n\t\t(2 "B.Cu" signal)\n\t\t(5 "F.SilkS" user "F.Silkscreen")\n\t\t(7 "B.SilkS" user "B.Silkscreen")\n\t\t(1 "F.Mask" user)\n\t\t(3 "B.Mask" user)\n\t\t(25 "Edge.Cuts" user)\n\t\t(35 "F.Fab" user)\n\t)\n\t(setup\n\t\t(pad_to_mask_clearance 0)\n\t)\n'''
footer = '\t(embedded_fonts no)\n)\n'
OUT.write_text(header + ''.join(body) + footer, encoding='utf-8')
print(OUT)
