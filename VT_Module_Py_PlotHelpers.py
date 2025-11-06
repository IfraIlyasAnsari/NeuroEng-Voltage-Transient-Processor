"""
===============================================================================
NeuroEng Voltage Transient Processor - Plot Helpers
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-10-21
Modified:   2025-10-27
Code:       VT_Module_Py_PlotHelpers.py
Version:    v1.0.1

DESCRIPTION
---------------------------
- Provides reusable helper functions specifically for enhancing matplotlib plots
  within the VT processing pipeline.
- Includes functions for:
  - Consistently placing Y-axis labels slightly leftward for better spacing.
  - Monkey-patching `plt.show()` to automatically apply Y-label adjustments
    after plot layout is finalized (controlled by `AUTO_YLABEL_ADJUST`).
  - Creating standardized, frame-less legends positioned top-left.
==============================================================================
"""
#==============================================================================
# 📦 Imports
#==============================================================================
import matplotlib.pyplot as plt

#--------------------
# Y-Axis Label Positioning Defaults
#--------------------
AUTO_YLABEL_ADJUST = True  # Master switch to enable/disable automatic adjustment
YLABEL_X  = -0.085         # Default horizontal offset (fraction of axes width)
YLABEL_Y  = 0.50           # Default vertical position (fraction of axes height)

#==============================================================================
# 🛠️ Helper Functions
#==============================================================================

def adjust_all_figs_ylabel_position(x: float = YLABEL_X, y: float = YLABEL_Y):
    """
    Iterates through all open matplotlib figures and adjusts the position
    of the Y-axis label for each axes object.
    
    Args:
        x: The target x-coordinate for the label (in axes coordinates).
        y: The target y-coordinate for the label (in axes coordinates).
    """
    # plt.get_fignums() returns a list of IDs for all active figures
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        # Iterate through all axes within the current figure
        for ax in fig.get_axes():
            # Check if the axes object has a yaxis attribute (it usually does)
            if hasattr(ax, "yaxis"):
                # Set the label coordinates relative to the axes boundaries
                # (0,0) is bottom-left, (1,1) is top-right.
                # Negative x moves it left of the axes.
                ax.yaxis.set_label_coords(x, y)

# --- Monkey-Patching plt.show() ---
# Store the original plt.show function before we replace it
original_show = plt.show

def patched_show(*args, **kwargs):
    """
    A replacement for `plt.show()` that first finalizes plot layout
    and then applies the Y-label adjustment before calling the original show.
    """
    # Ensure layout adjustments (like constrained_layout) are applied
    plt.draw() 
    # If the global flag is enabled, apply the Y-label adjustment
    if AUTO_YLABEL_ADJUST:
        adjust_all_figs_ylabel_position()
    # Call the original plt.show() function to actually display the plots
    return original_show(*args, **kwargs)

# ❗ Crucial Step: Replace the global plt.show with our patched version.
# This happens automatically whenever this module is imported.
plt.show = patched_show

# --- Direct Label Placement Function (Less commonly needed now) ---
def place_ylabel_left(ax, x: float | None = None, y: float | None = None):
    """
    Manually sets the position of the Y-label for a *specific* axes object.
    Uses axes coordinates and adjusts alignment.
    (Less necessary now due to the automatic patching of plt.show)
    """
    # Use global defaults if specific coordinates are not provided
    x_coord = x if x is not None else YLABEL_X
    y_coord = y if y is not None else YLABEL_Y
    
    # Get the label object associated with the Y-axis
    label = ax.yaxis.get_label()
    # Set the coordinate system for positioning to be relative to the axes
    label.set_transform(ax.transAxes) 
    # Align the label text: right edge at x_coord, vertical center at y_coord
    label.set_ha("right")
    label.set_va("center")
    # Allow the label to be drawn outside the main axes area
    label.set_clip_on(False) 
    # Set the position using the chosen coordinates
    label.set_position((x_coord, y_coord)) 

# --- Standardized Legend Function ---
def place_legend_left(ax) -> plt.legend:
    """
    Creates and places a standardized, frame-less legend in the upper-left
    corner of the provided axes object.
    """
    # Create the legend with specific styling options
    leg = ax.legend(
        loc="upper left",           # Anchor point on the legend
        bbox_to_anchor=(0.02, 0.98), # Position relative to axes (x, y)
        frameon=False,              # No background box or border
        labelspacing=0.4,           # Vertical space between entries
        handlelength=2.0,           # Length of the line/marker sample
        borderpad=0.0               # No padding inside the (invisible) frame
    )
    # Attempt to force left-alignment of text within the legend
    # This might fail on older matplotlib versions, hence the try/except
    try:
        # Access internal property to align text block
        leg._legend_box.align = "left" 
    except AttributeError:
        # Ignore if the property doesn't exist
        pass 
    return leg

# --- Example of a legend with a title (Commented out) ---
# def place_legend_left_with_title(ax, title="Default Title"):
#     """ Example: Left-aligned legend with a title, without frame."""
#     leg = ax.legend(
#         title=title,
#         loc="upper left",
#         bbox_to_anchor=(0.02, 0.98),
#         frameon=False,            
#         labelspacing=0.4,
#         borderpad=0.0,
#         handlelength=2.0,
#     )
#     # Left-align both the title text and the legend entries
#     try:
#         leg._legend_box.align = "left" 
#     except AttributeError:
#         pass
#     # Explicitly align the title text itself
#     leg.get_title().set_ha("left") 
#     return leg

#==============================================================================
# 📤 Module Exports
#==============================================================================
# Define what gets imported by 'from VT_Module_Py_PlotHelpers import *'

__all__ = [
    # Global flag
    "AUTO_YLABEL_ADJUST",
    # Functions
    "adjust_all_figs_ylabel_position",
    "place_ylabel_left",
    "place_legend_left",
    # Constants for manual adjustment (if needed)
    "YLABEL_X",
    "YLABEL_Y",
    # Note: patched_show is applied automatically, no need to export its name
]