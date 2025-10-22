#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 19:53:46 2025

@author: ifraa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
NeuroEng Voltage Transient Processor - Plotting Helpers
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-10-21
Code:       VT_Module_Py_PlotHelpers.py
Version:    v1.0.0

DESCRIPTION
---------------------------
- Provides advanced, reusable helper functions for matplotlib
  plot styling, used by the VT processing pipeline.
- Includes functions for:
  - Adjusting Y-axis label positions.
  - Monkey-patching plt.show() to apply adjustments.
  - Creating standardized, frame-less legends.
  
#==============================================================================
"""

import matplotlib.pyplot as plt

#--------------------
# Y-Axis label positioning defaults
#--------------------
AUTO_YLABEL_ADJUST = False
YLABEL_X  = -0.085   # horizontal offset
YLABEL_Y  = 0.50    # vertical position

#--------------------
# [HELPER FUNCTION] Axis Labels
#--------------------

def adjust_all_figs_ylabel_position(x=YLABEL_X, y=YLABEL_Y):
    """Offset y-axis labels for all axes in all open figures."""
    for num in plt.get_fignums():
        fig = plt.figure(num)
        for ax in fig.get_axes():
            if hasattr(ax, "yaxis"):
                ax.yaxis.set_label_coords(x, y)

# Monkey-patch plt.show so labels are nudged AFTER layout is finalized
original_show = plt.show
def patched_show(*args, **kwargs):
    plt.draw()  # finalize layout (tight/constrained)
    if AUTO_YLABEL_ADJUST:
        adjust_all_figs_ylabel_position()
    return original_show(*args, **kwargs)

# Apply the patch globally when this module is imported
plt.show = patched_show

def place_ylabel_left(ax, x=None, y=None):
    """Put y-label left of the spine using axes coords (no autosnap)."""
    if x is None: x = YLABEL_X
    if y is None: y = YLABEL_Y
    lab = ax.yaxis.get_label()
    lab.set_transform(ax.transAxes)   # use axes coordinates
    lab.set_ha("right"); lab.set_va("center")
    lab.set_clip_on(False)            # allow outside drawing
    lab.set_position((x, y))          # same as set_label_coords(x, y)

def place_legend_left(ax):
    """Left-aligned legend without a title or frame."""
    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,          # no border box
        labelspacing=0.4,       # vertical spacing between entries
        handlelength=2.0,       # line length in legend
        borderpad=0.0
    )
    try:
        leg._legend_box.align = "left"   # keep labels left-aligned
    except Exception:
        pass
    return leg

# def place_legend_left(ax, title="Before & After"):
#     """ Left-aligned legend with title, without frame."""
#     leg = ax.legend(
#         title=title,
#         loc="upper left",
#         bbox_to_anchor=(0.02, 0.98),
#         frameon=False,                # ❌ no frame
#         labelspacing=0.4,
#         borderpad=0.0,
#         handlelength=2.0,
#     )
#     # Left-align both the title and labels
#     try:
#         leg._legend_box.align = "left"   # aligns all legend items
#     except Exception:
#         pass
#     leg.get_title().set_ha("left")       # aligns just the title text
#     return leg

#==============================================================================

__all__ = [
    "patched_show",
    "adjust_all_figs_ylabel_position",
    "place_ylabel_left",
    "place_legend_left",
    "AUTO_YLABEL_ADJUST",
    "YLABEL_X",
    "YLABEL_Y",
]