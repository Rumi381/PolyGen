import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from typing import Union, Dict, List
from .extractVoronoiEdges import VoronoiEdgeExtractor

def get_figure_size(layout='single', journal_type='large'):
    """
    Calculate figure size in inches based on journal specifications.
    
    Parameters
    ----------
    layout : str
        Layout type ('single', '1x2', '1x3', '2x2', '2x3', '3x2', '3x3').
    journal_type : str
        Journal size ('large' or 'small').
    
    Returns
    -------
    tuple
        Figure size in inches (width, height).
    """
    mm_to_inches = 1 / 25.4  # Conversion factor from mm to inches

    # Journal-specific dimensions
    if journal_type == 'large':
        single_column = 84 * mm_to_inches  # 84 mm width for single column
        double_column = 174 * mm_to_inches  # 174 mm width for double column
        max_height = 234 * mm_to_inches  # Max height: 234 mm
    elif journal_type == 'small':
        single_column = 119 * mm_to_inches  # 119 mm width for single column
        double_column = 119 * mm_to_inches  # Use single-column width for all
        max_height = 195 * mm_to_inches  # Max height: 195 mm
    else:
        raise ValueError("journal_type must be 'large' or 'small'.")

    # Aspect ratio constants
    aspect_ratios = {
        'single': (4, 3),  # Standard 4:3 for single plot
        '1x2': (8, 3),     # Wider for 1x2 (16:6 or 8:3)
        '1x3': (12, 3),    # Very wide for 1x3
        '2x2': (4, 4),     # Square for 2x2
        '2x3': (6, 4),     # Moderate height for 2x3
        '3x2': (4, 6),     # Taller for 3x2
        '3x3': (4, 4)      # Square for 3x3
    }

    # Get aspect ratio for the layout
    ratio = aspect_ratios.get(layout, aspect_ratios['single'])
    width, height = ratio

    # Scale width and height to fit the journal's column width
    if layout == 'single':
        width = single_column
    else:
        width = double_column
    height = (width / ratio[0]) * ratio[1]

    # Ensure height doesn't exceed maximum allowed
    height = min(height, max_height)
    return (width, height)
 
def set_publication_style():
    """
    Configure matplotlib for publication-quality figures following journal guidelines.
    
    Key requirements implemented:
    - Vector graphics (saved as EPS)
    - Helvetica/Arial font
    - Consistent font sizing (8-12pt)
    - Minimum line width 0.3pt
    - No titles within figures
    - Proper figure dimensions
    """
    plt.style.use('default')
    
    params = {
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],  # As per guidelines
        'text.usetex': False,  # Disable LaTeX to ensure font consistency
        
        # Font sizes (8-12pt as specified)
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        
        # Line widths (minimum 0.3pt = 0.1mm)
        'axes.linewidth': 0.3,
        'grid.linewidth': 0.3,
        'lines.linewidth': 0.3,
        'xtick.major.width': 0.3,
        'ytick.major.width': 0.3,
        
        # Figure settings
        'figure.dpi': 300,        # For line art
        'savefig.dpi': 1200,      # For line art
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Other settings
        'figure.figsize': [3.3, 2.0],  # 84mm width (single column) at 72 dpi
        'figure.autolayout': True
    }
    plt.rcParams.update(params)

# Ploting the domain with the seed points (if provided)
def plot_boundary_with_points(
    polygon,
    figure_name,
    points=None,
    show_figure=True,
    exterior_color='#1B365D',
    exterior_style='-',
    interior_color='r',
    interior_style='-',
    line_width=0.5,
    point_color='blue',
    point_size=0.5,
    point_alpha=1.0,
    point_marker='o',
    margins=0.01,
    dpi=1200,
    show_axes=False,
    output_dir='Plots'
):
    """Plot polygon boundaries with optional seed points.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        The polygon(s) to be plotted. Can be either a single polygon or multiple polygons.
    figure_name : str
        Name of the output figure. If no extension is provided, '.png' is used.
    points : numpy.ndarray, optional
        Array of points with shape (n, 2) containing x, y coordinates.
    show_figure : bool, optional
        Whether to display the figure, by default True.
    exterior_color : str, optional
        Color for exterior boundaries. Can use shorthand or full name.
        Common shorthands: 'k' (black), 'r' (red), 'b' (blue), 'g' (green),
        'y' (yellow), 'm' (magenta), 'c' (cyan), 'w' (white).
        Default is 'k'.
    exterior_style : str, optional
        Line style for exterior boundaries. Options: ['-', '--', '-.', ':'].
        Default is '-' (solid).
    interior_color : str, optional
        Color for interior boundaries. Same color options as exterior_color.
        Default is 'r'.
    interior_style : str, optional
        Line style for interior boundaries. Same options as exterior_style.
        Default is '-' (solid).
    line_width : float, optional
        Width of all boundary lines, by default 0.5.
    point_color : str, optional
        Color for points. Same color options as exterior_color.
        Default is 'blue'.
    point_size : float, optional
        Size of points, by default 0.5.
    point_alpha : float, optional
        Transparency of points (0 to 1), by default 1.0.
    point_marker : str, optional
        Marker style for points. Common options: 'o' (circle), 's' (square),
        '^' (triangle up), 'v' (triangle down), '*' (star).
        Default is 'o'.
    margins : float or tuple, optional
        Margins around the plot. Can be:
        - Single float for same margins in x and y
        - Tuple (x_margin, y_margin) for different margins
        Default is 0.01 (1% margin in both x and y).
    dpi : int, optional
        Resolution of saved figure in dots per inch, by default 1200.
    show_axes : bool, optional
        Whether to show axes and ticks, by default False.
    output_dir : str, optional
        Directory to save the figure, by default 'Plots'.

    Notes
    -----
    The function automatically creates the output directory if it doesn't exist.
    Uses the publication style settings from set_publication_style().
    
    Color specifications can use either shorthand (e.g., 'k', 'r', 'b') or 
    full names (e.g., 'black', 'red', 'blue'). Additional valid color names 
    include standard colors like 'navy', 'forestgreen', 'darkred', etc.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> import numpy as np
    >>> # Create a simple triangle
    >>> polygon = Polygon([(0, 0), (1, 0), (0.5, 1)])
    >>> # Create some random points
    >>> points = np.random.rand(10, 2)
    >>> # Plot with custom styling
    >>> plot_boundary_with_points(
    ...     polygon,
    ...     'triangle_plot',
    ...     points=points,
    ...     exterior_color='b',  # or 'blue'
    ...     interior_color='darkred',
    ...     exterior_style='--',
    ...     point_color='forestgreen',
    ...     point_size=1.0
    ... )
    """
    # Input validation
    valid_line_styles = ['-', '--', '-.', ':']
    if exterior_style not in valid_line_styles or interior_style not in valid_line_styles:
        raise ValueError(f"Line styles must be one of: {valid_line_styles}")
    
    if points is not None and (not isinstance(points, np.ndarray) or points.shape[1] != 2):
        raise ValueError("points must be a numpy array with shape (n, 2)")

    # Setup
    set_publication_style()
    fig, ax = plt.subplots(figsize=get_figure_size(layout='single', journal_type='large'))
    ax.set_aspect('equal')

    # Set margins
    if isinstance(margins, (int, float)):
        ax.margins(margins)
    else:
        ax.margins(x=margins[0], y=margins[1])

    # Convert single polygon to list for consistent handling
    polygons = [polygon] if isinstance(polygon, Polygon) else polygon.geoms

    # Plot boundaries
    for poly in polygons:
        # Plot exterior
        x, y = poly.exterior.xy
        ax.plot(x, y, 
                color=exterior_color,
                linestyle=exterior_style,
                linewidth=line_width)
        
        # Plot interiors
        for interior in poly.interiors:
            x, y = interior.xy
            ax.plot(x, y,
                   color=interior_color,
                   linestyle=interior_style,
                   linewidth=line_width)

    # Plot points if provided
    if points is not None:
        ax.scatter(
            points[:, 0], points[:, 1],
            color=point_color,
            s=point_size,
            marker=point_marker,
            alpha=point_alpha,
            linewidth=0,
            zorder=2
        )

    # Handle axes visibility
    if not show_axes:
        ax.axis('off')

    plt.tight_layout()

    # Handle figure saving
    os.makedirs(output_dir, exist_ok=True)
    base_name, ext = os.path.splitext(figure_name)
    ext = ext if ext else ".png"
    
    plt.savefig(
        f'{output_dir}/{base_name}{ext}',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0
    )
    
    if show_figure:
        plt.show()
    
    plt.close()

# Plotting the original Voronoi cells 
def plot_voronoi_cells(
    voronoi_cells,
    figure_name,
    points=None,
    show_figure=True,
    cell_color='#1B365D',
    cell_style='-',
    line_width=0.5,
    point_color='blue',
    point_size=0.5,
    point_alpha=1.0,
    point_marker='o',
    margins=0.01,
    dpi=1200,
    show_axes=False,
    output_dir='Plots'
):
    """Plot Voronoi cells with optional seed points.

    Parameters
    ----------
    voronoi_cells : list
        List of shapely.geometry.Polygon or shapely.geometry.MultiPolygon objects
        representing the Voronoi cells.
    figure_name : str
        Name of the output figure. If no extension is provided, '.png' is used.
    points : numpy.ndarray, optional
        Array of points with shape (n, 2) containing x, y coordinates.
    show_figure : bool, optional
        Whether to display the figure, by default True.
    cell_color : str, optional
        Color for Voronoi cell boundaries. Can use shorthand or full name.
        Common shorthands: 'k' (black), 'r' (red), 'b' (blue), 'g' (green),
        'y' (yellow), 'm' (magenta), 'c' (cyan), 'w' (white).
        Default is 'k'.
    cell_style : str, optional
        Line style for cell boundaries. Options: ['-', '--', '-.', ':'].
        Default is '-' (solid).
    line_width : float, optional
        Width of cell boundary lines, by default 0.5.
    point_color : str, optional
        Color for points. Same color options as cell_color.
        Default is 'blue'.
    point_size : float, optional
        Size of points, by default 0.5.
    point_alpha : float, optional
        Transparency of points (0 to 1), by default 1.0.
    point_marker : str, optional
        Marker style for points. Common options: 'o' (circle), 's' (square),
        '^' (triangle up), 'v' (triangle down), '*' (star).
        Default is 'o'.
    margins : float or tuple, optional
        Margins around the plot. Can be:
        - Single float for same margins in x and y
        - Tuple (x_margin, y_margin) for different margins
        Default is 0 (1% margin in both x and y).
    dpi : int, optional
        Resolution of saved figure in dots per inch, by default 1200.
    show_axes : bool, optional
        Whether to show axes and ticks, by default False.
    output_dir : str, optional
        Directory to save the figure, by default 'Plots'.

    Notes
    -----
    The function automatically creates the output directory if it doesn't exist.
    Uses the publication style settings from set_publication_style().
    
    Color specifications can use either shorthand (e.g., 'k', 'r', 'b') or 
    full names (e.g., 'black', 'red', 'blue'). Additional valid color names 
    include standard colors like 'navy', 'forestgreen', 'darkred', etc.

    Examples
    --------
    >>> # Create some Voronoi cells and points
    >>> cells = [Polygon(...), Polygon(...)]  # List of Voronoi cell polygons
    >>> points = np.array([[0, 0], [1, 1]])   # Seed points
    >>> # Plot with custom styling
    >>> plot_voronoi_cells(
    ...     cells,
    ...     'voronoi_plot',
    ...     points=points,
    ...     cell_color='navy',
    ...     cell_style='--',
    ...     point_color='r',
    ...     point_size=1.0
    ... )
    """
    # Input validation
    valid_line_styles = ['-', '--', '-.', ':']
    if cell_style not in valid_line_styles:
        raise ValueError(f"Line style must be one of: {valid_line_styles}")
    
    if points is not None and (not isinstance(points, np.ndarray) or points.shape[1] != 2):
        raise ValueError("points must be a numpy array with shape (n, 2)")

    # Setup
    set_publication_style()
    fig, ax = plt.subplots(figsize=get_figure_size(layout='single', journal_type='large'))
    ax.set_aspect('equal')

    # Set margins
    if isinstance(margins, (int, float)):
        ax.margins(margins)
    else:
        ax.margins(x=margins[0], y=margins[1])

    # Plot cells
    for cell in voronoi_cells:
        if isinstance(cell, MultiPolygon):
            polygons = cell.geoms
        else:
            polygons = [cell]
            
        for polygon in polygons:
            x, y = polygon.exterior.xy
            ax.plot(x, y, 
                   color=cell_color,
                   linestyle=cell_style,
                   linewidth=line_width)

    # Plot points if provided
    if points is not None:
        ax.scatter(
            points[:, 0], points[:, 1],
            color=point_color,
            s=point_size,
            marker=point_marker,
            alpha=point_alpha,
            linewidth=0,
            zorder=2
        )

    # Handle axes visibility
    if not show_axes:
        ax.axis('off')

    plt.tight_layout()

    # Handle figure saving
    os.makedirs(output_dir, exist_ok=True)
    base_name, ext = os.path.splitext(figure_name)
    ext = ext if ext else ".png"
    
    plt.savefig(
        f'{output_dir}/{base_name}{ext}',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0
    )
    
    if show_figure:
        plt.show()
    
    plt.close()

# Plot the extracted edges of the Voronoi cells
def plot_voronoi_edges(
   voronoi_cells,
   base_figure,
   original_cells=None,
   show_figure=True,
   boundary_color='r',
   boundary_style='-',
   internal_color='b',
   internal_style='-',
   line_width=0.5,
   margins=0.01,
   dpi=1200,
   show_axes=False,
   output_dir='Plots'
):
   """Plot and save figures showing the whole Voronoi region, boundary edges,
   and internal edges.

   Parameters
   ----------
   voronoi_cells : list
       List of shapely.geometry.Polygon objects representing clipped Voronoi cells.
   base_figure : str
       Base name for the output figures. If no extension provided, '.png' is used.
       Three files will be created with suffixes: '_whole', '_boundary', '_internal'.
   original_cells : list, optional
       List of shapely.geometry.Polygon objects representing original Voronoi cells.
       If None, edge extraction will be performed only on voronoi_cells.
   show_figure : bool, optional
       Whether to display the figures, by default True.
   boundary_color : str, optional
       Color for boundary edges. Can use shorthand or full name.
       Common shorthands: 'k' (black), 'r' (red), 'b' (blue), 'g' (green).
       Default is 'r'.
   boundary_style : str, optional
       Line style for boundary edges. Options: ['-', '--', '-.', ':'].
       Default is '-' (solid).
   internal_color : str, optional
       Color for internal edges. Same color options as boundary_color.
       Default is 'b'.
   internal_style : str, optional
       Line style for internal edges. Same options as boundary_style.
       Default is '-' (solid).
   line_width : float, optional
       Width of all edge lines, by default 0.5.
   margins : float or tuple, optional
       Margins around the plot. Can be:
       - Single float for same margins in x and y
       - Tuple (x_margin, y_margin) for different margins
       Default is 0.01 (1% margin in both x and y).
   dpi : int, optional
       Resolution of saved figures in dots per inch, by default 1200.
   show_axes : bool, optional
       Whether to show axes and ticks, by default False.
   output_dir : str, optional
       Directory to save the figures, by default 'Plots'.

   Notes
   -----
   The function creates three separate figures:
   1. [basename]_whole: Complete Voronoi diagram
   2. [basename]_boundary: Only boundary edges
   3. [basename]_internal: Only internal edges

   The function automatically creates the output directory if it doesn't exist.
   Uses the publication style settings from set_publication_style().

   Color specifications can use either shorthand (e.g., 'k', 'r', 'b') or 
   full names (e.g., 'black', 'red', 'blue').

   Examples
   --------
   >>> # Create Voronoi cells
   >>> voronoi_cells = [Polygon(...), Polygon(...)]
   >>> # Plot with default edge extraction
   >>> plot_voronoi_edges(
   ...     voronoi_cells,
   ...     'voronoi_diagram',
   ...     boundary_color='darkred',
   ...     internal_color='navy'
   ... )
   >>> 
   >>> # Plot with original cells for edge comparison
   >>> original_cells = [Polygon(...), Polygon(...)]
   >>> plot_voronoi_edges(
   ...     voronoi_cells,
   ...     'voronoi_diagram',
   ...     original_cells=original_cells,
   ...     boundary_color='darkred',
   ...     internal_color='navy'
   ... )
   """
   # Input validation
   valid_line_styles = ['-', '--', '-.', ':']
   if boundary_style not in valid_line_styles or internal_style not in valid_line_styles:
       raise ValueError(f"Line styles must be one of: {valid_line_styles}")

   # Setup
   set_publication_style()

   # Extract boundary and internal edges
   extractor = VoronoiEdgeExtractor(use_kdtree=True)
   boundary_edges, internal_cell_edges = extractor.extract_edges(modified_cells=voronoi_cells,
                                                                  original_cells=original_cells)

   # Prepare file names
   base_name, ext = os.path.splitext(base_figure)
   ext = ext if ext else ".png"
   os.makedirs(output_dir, exist_ok=True)

   # Plot whole region
   plot_boundary_with_points(
       polygon=MultiPolygon(voronoi_cells),
       figure_name=f"{base_name}_whole{ext}",
       show_figure=show_figure
   )

   # Plot boundary edges
   fig, ax = plt.subplots(figsize=get_figure_size(layout='single', journal_type='large'))
   ax.set_aspect('equal')
   
   # Set margins
   if isinstance(margins, (int, float)):
       ax.margins(margins)
   else:
       ax.margins(x=margins[0], y=margins[1])
   
   if not show_axes:
       ax.axis('off')
       
   for edge in boundary_edges:
       ax.plot(*edge.xy,
               color=boundary_color,
               linestyle=boundary_style,
               linewidth=line_width)
   
   plt.tight_layout()
   plt.savefig(
       f"{output_dir}/{base_name}_boundary{ext}",
       dpi=dpi,
       bbox_inches='tight',
       pad_inches=0
   )
   if show_figure:
       plt.show()
   plt.close()

   # Plot internal edges
   fig, ax = plt.subplots(figsize=get_figure_size(layout='single', journal_type='large'))
   ax.set_aspect('equal')
   
   # Set margins
   if isinstance(margins, (int, float)):
       ax.margins(margins)
   else:
       ax.margins(x=margins[0], y=margins[1])
   
   if not show_axes:
       ax.axis('off')
       
   for edge in internal_cell_edges:
       ax.plot(*edge.xy,
               color=internal_color,
               linestyle=internal_style,
               linewidth=line_width)
   
   plt.tight_layout()
   plt.savefig(
       f"{output_dir}/{base_name}_internal{ext}",
       dpi=dpi,
       bbox_inches='tight',
       pad_inches=0
   )
   if show_figure:
       plt.show()
   plt.close()

# Plot the the Voronoi cells highlighting the short edges
def plot_voronoi_cells_with_short_edges(
    voronoi_cells,
    figure_name,
    N=None,
    threshold=None,
    points=None,
    show_figure=True,
    cell_color='#1B365D',
    cell_style='-',
    cell_width=0.5,
    short_edge_color='r',
    short_edge_style='-',
    short_edge_width=0.6,
    point_color='r',
    point_size=0.5,
    point_alpha=1.0,
    point_marker='o',
    margins=0.01,
    dpi=1200,
    show_axes=False,
    output_dir='Plots'
):
    """Plot Voronoi cells with highlighted short edges based on either N shortest edges
    or a threshold length.

    Parameters
    ----------
    voronoi_cells : list
        List of shapely.geometry.Polygon or shapely.geometry.MultiPolygon objects
        representing the Voronoi cells.
    figure_name : str
        Name of the output figure. If no extension provided, '.png' is used.
    N : int, optional
        Number of shortest edges to highlight. Must specify either N or threshold,
        but not both.
    threshold : float, optional
        Length threshold below which edges are highlighted. Must specify either N
        or threshold, but not both.
    points : numpy.ndarray, optional
        Array of points with shape (n, 2) containing x, y coordinates.
    show_figure : bool, optional
        Whether to display the figure, by default True.
    cell_color : str, optional
        Color for Voronoi cell boundaries. Can use shorthand or full name.
        Common shorthands: 'k' (black), 'r' (red), 'b' (blue), 'g' (green).
        Default is 'k'.
    cell_style : str, optional
        Line style for cell boundaries. Options: ['-', '--', '-.', ':'].
        Default is '-' (solid).
    cell_width : float, optional
        Width of cell boundary lines, by default 0.5.
    short_edge_color : str, optional
        Color for highlighted short edges. Same color options as cell_color.
        Default is 'r'.
    short_edge_style : str, optional
        Line style for short edges. Same options as cell_style.
        Default is '-' (solid).
    short_edge_width : float, optional
        Width of short edge lines, by default 2.0.
    point_color : str, optional
        Color for points. Same color options as cell_color.
        Default is 'r'.
    point_size : float, optional
        Size of points, by default 0.5.
    point_alpha : float, optional
        Transparency of points (0 to 1), by default 1.0.
    point_marker : str, optional
        Marker style for points. Common options: 'o' (circle), 's' (square),
        '^' (triangle up), 'v' (triangle down), '*' (star).
        Default is 'o'.
    margins : float or tuple, optional
        Margins around the plot. Can be:
        - Single float for same margins in x and y
        - Tuple (x_margin, y_margin) for different margins
        Default is 0.01 (1% margin in both x and y).
    dpi : int, optional
        Resolution of saved figure in dots per inch, by default 1200.
    show_axes : bool, optional
        Whether to show axes and ticks, by default False.
    output_dir : str, optional
        Directory to save the figure, by default 'Plots'.

    Raises
    ------
    ValueError
        If neither or both N and threshold are specified.
        If N is negative.
        If threshold is negative.
        If points array has incorrect shape.

    Notes
    -----
    The function automatically creates the output directory if it doesn't exist.
    Uses the publication style settings from set_publication_style().

    Color specifications can use either shorthand (e.g., 'k', 'r', 'b') or 
    full names (e.g., 'black', 'red', 'blue').

    Examples
    --------
    >>> # Highlight 5 shortest edges
    >>> plot_voronoi_cells_with_short_edges(
    ...     voronoi_cells,
    ...     'short_edges',
    ...     N=5,
    ...     short_edge_color='red',
    ...     short_edge_width=3.0
    ... )
    >>> 
    >>> # Highlight edges shorter than 0.1 units
    >>> plot_voronoi_cells_with_short_edges(
    ...     voronoi_cells,
    ...     'short_edges',
    ...     threshold=0.1,
    ...     cell_color='navy',
    ...     short_edge_color='darkred'
    ... )
    """
    # Input validation
    if (N is None) == (threshold is None):
        raise ValueError("Specify exactly one of 'N' or 'threshold'")
    
    if N is not None and N < 0:
        raise ValueError("N must be non-negative")
        
    if threshold is not None and threshold < 0:
        raise ValueError("threshold must be non-negative")
    
    if points is not None and (not isinstance(points, np.ndarray) or points.shape[1] != 2):
        raise ValueError("points must be a numpy array with shape (n, 2)")

    valid_line_styles = ['-', '--', '-.', ':']
    if cell_style not in valid_line_styles or short_edge_style not in valid_line_styles:
        raise ValueError(f"Line styles must be one of: {valid_line_styles}")

    # Setup
    set_publication_style()
    fig, ax = plt.subplots(figsize=get_figure_size(layout='single', journal_type='large'))
    ax.set_aspect('equal')

    # Set margins
    if isinstance(margins, (int, float)):
        ax.margins(margins)
    else:
        ax.margins(x=margins[0], y=margins[1])
    
    # Collect and sort edges
    edges = []
    for cell in voronoi_cells:
        polygons = [cell] if isinstance(cell, Polygon) else cell.geoms
        for poly in polygons:
            coords = list(poly.exterior.coords)
            for i in range(len(coords) - 1):
                edge = LineString([coords[i], coords[i + 1]])
                edges.append((edge.length, edge))
    
    edges.sort(key=lambda x: x[0])
    
    # Select short edges
    short_edges = edges[:N] if N else [(l, e) for l, e in edges if l < threshold]
    
    # Plot Voronoi cells
    for cell in voronoi_cells:
        polygons = [cell] if isinstance(cell, Polygon) else cell.geoms
        for poly in polygons:
            x, y = poly.exterior.xy
            ax.plot(x, y,
                   color=cell_color,
                   linestyle=cell_style,
                   linewidth=cell_width)
    
    # Highlight short edges
    for _, edge in short_edges:
        x, y = edge.xy
        ax.plot(x, y,
                color=short_edge_color,
                linestyle=short_edge_style,
                linewidth=short_edge_width)
    
    # Plot points if provided
    if points is not None:
        ax.scatter(
            points[:, 0], points[:, 1],
            color=point_color,
            s=point_size,
            marker=point_marker,
            alpha=point_alpha,
            linewidth=0,
            zorder=2
        )
    
    # Handle axes visibility
    if not show_axes:
        ax.axis('off')
    
    plt.tight_layout()
    
    # Handle figure saving
    os.makedirs(output_dir, exist_ok=True)
    base_name, ext = os.path.splitext(figure_name)
    ext = ext if ext else ".png"
    
    plt.savefig(
        f'{output_dir}/{base_name}{ext}',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0
    )
    
    if show_figure:
        plt.show()
    
    plt.close()

# Plot the triangulated Geometries and Voronoi cells
def plot_triangulated_geometry(
    geometry: Union[Polygon, List[Polygon]],
    triangulation: Union[List[Polygon], Dict[int, List[Polygon]]],
    figure_name: str,
    highlight_short_edges: bool = False,
    N: int = None,
    threshold: float = None,
    show_figure: bool = True,
    cell_color: str = '#1B365D',
    cell_style: str = '-',
    cell_width: float = 0.4,
    triangle_color: str = 'k',
    triangle_style: str = '-',
    triangle_width: float = 0.2,
    short_edge_color: str = 'r',
    short_edge_style: str = '-',
    short_edge_width: float = 0.4,
    margins: Union[float, tuple] = 0.01,
    dpi: int = 1200,
    show_axes: bool = False,
    output_dir: str = 'Plots'
):
    """Plot triangulated geometry - handles both single polygon and multiple Voronoi cells.

    Parameters
    ----------
    geometry : Union[Polygon, List[Polygon]]
        Either a single shapely.geometry.Polygon or a list of Polygons (Voronoi cells)
    triangulation : Union[List[Polygon], Dict[int, List[Polygon]]]
        Either a list of triangles (for single polygon) or a dictionary mapping 
        cell indices to lists of triangles (for Voronoi cells)
    ... (rest of parameters same as original)

    Notes
    -----
    The function now handles two cases:
    1. Single polygon with holes and its triangulation
    2. Multiple Voronoi cells and their triangulations (original functionality)
    """
    # Input validation
    if highlight_short_edges:
        if (N is not None) == (threshold is not None):
            raise ValueError(
                "When highlight_short_edges is True, "
                "specify exactly one of 'N' or 'threshold'"
            )
        if N is not None and N < 0:
            raise ValueError("N must be non-negative")
        if threshold is not None and threshold < 0:
            raise ValueError("threshold must be non-negative")

    valid_line_styles = ['-', '--', '-.', ':']
    for style in [cell_style, triangle_style, short_edge_style]:
        if style not in valid_line_styles:
            raise ValueError(f"Line styles must be one of: {valid_line_styles}")

    # Normalize inputs to handle both cases
    if isinstance(geometry, Polygon):
        geometries = [geometry]
        if not isinstance(triangulation, list):
            raise ValueError("For single polygon input, triangulation must be a list of triangles")
        triangulations = {0: triangulation}
    else:
        geometries = geometry
        if not isinstance(triangulation, dict):
            raise ValueError("For multiple polygons input, triangulation must be a dictionary")
        triangulations = triangulation

    # Setup
    set_publication_style()
    fig, ax = plt.subplots(figsize=get_figure_size(layout='single', journal_type='large'))
    ax.set_aspect('equal')
    
    # Set margins
    if isinstance(margins, (int, float)):
        ax.margins(margins)
    else:
        ax.margins(x=margins[0], y=margins[1])
    
    # Process short edges if needed
    short_edge_coords = set()
    if highlight_short_edges:
        # Collect and measure all boundary edges
        edges = []
        for geom in geometries:
            # Handle exterior boundaries
            coords = list(geom.exterior.coords)
            for i in range(len(coords) - 1):
                edge = LineString([coords[i], coords[i + 1]])
                edges.append((edge.length, edge))
            
            # Handle interior boundaries (holes)
            has_interiors = bool(geom.interiors)
            if has_interiors:
                for interior in geom.interiors:
                    coords = list(interior.coords)
                    for i in range(len(coords) - 1):
                        edge = LineString([coords[i], coords[i + 1]])
                        edges.append((edge.length, edge))
        
        # Determine short edges based on criteria
        if N is not None:
            edges.sort(key=lambda x: x[0])
            short_edges = edges[:N]
        else:
            short_edges = [(l, e) for l, e in edges if l < threshold]
        
        # Create set of short edge coordinates for efficient lookup
        for _, edge in short_edges:
            coords = tuple(map(tuple, edge.coords))
            short_edge_coords.add(coords)
            short_edge_coords.add(coords[::-1])
    
    # Plot geometry boundaries
    for geom in geometries:
        # Plot exterior boundary
        coords = list(geom.exterior.coords)
        for i in range(len(coords) - 1):
            start, end = coords[i], coords[i + 1]
            edge_coords = ((start[0], start[1]), (end[0], end[1]))
            
            if highlight_short_edges and (edge_coords in short_edge_coords or 
                                        edge_coords[::-1] in short_edge_coords):
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color=short_edge_color,
                       linestyle=short_edge_style,
                       linewidth=short_edge_width,
                       zorder=3)
            else:
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color=cell_color,
                       linestyle=cell_style,
                       linewidth=cell_width,
                       zorder=1)
        
        # Plot interior boundaries (holes)
        has_interiors = bool(geom.interiors)
        if has_interiors:
            for interior in geom.interiors:
                coords = list(interior.coords)
                for i in range(len(coords) - 1):
                    start, end = coords[i], coords[i + 1]
                    edge_coords = ((start[0], start[1]), (end[0], end[1]))
                    
                    if highlight_short_edges and (edge_coords in short_edge_coords or 
                                                edge_coords[::-1] in short_edge_coords):
                        ax.plot([start[0], end[0]], [start[1], end[1]], 
                            color=short_edge_color,
                            linestyle=short_edge_style,
                            linewidth=short_edge_width,
                            zorder=3)
                    else:
                        ax.plot([start[0], end[0]], [start[1], end[1]], 
                            color=cell_color,
                            linestyle=cell_style,
                            linewidth=cell_width,
                            zorder=1)
    
    # Plot triangulations
    for cell_idx, triangles in triangulations.items():
        for triangle in triangles:
            x, y = triangle.exterior.xy
            ax.plot(x, y,
                   color=triangle_color,
                   linestyle=triangle_style,
                   linewidth=triangle_width,
                   zorder=2)
    
    # Handle axes visibility
    if not show_axes:
        ax.axis('off')

    plt.tight_layout()
    
    # Handle figure saving
    os.makedirs(output_dir, exist_ok=True)
    base_name, ext = os.path.splitext(figure_name)
    ext = ext if ext else ".png"
    
    plt.savefig(
        f'{output_dir}/{base_name}{ext}',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0
    )
    
    if show_figure:
        plt.show()
    
    plt.close()

# Plotting the error vs iteration comparison with and without decay mechanism
def plot_error_comparison(
    metrics1,
    metrics2,
    figure_name,
    show_figure=True,
    line1_color='k',
    line1_style='-',
    line1_width=1.0,
    line2_color='r',
    line2_style='--',
    line2_width=1.0,
    marker_size=4,
    grid_alpha=0.2,
    ref_line_color='gray',
    ref_line_style=':',
    ref_line_alpha=0.5,
    label1='Without Decay',
    label2='With Decay',
    dpi=1200,
    output_dir='Plots'
):
    """Plot error comparison between two optimization methods.

    Parameters
    ----------
    metrics1 : dict
        Dictionary containing error metrics for first method with keys:
        - 'error_values': list of error values per iteration
        - 'min_error': minimum error achieved
    metrics2 : dict
        Dictionary containing error metrics for second method with same structure
        as metrics1.
    figure_name : str
        Name of the output figure. If no extension provided, '.png' is used.
    show_figure : bool, optional
        Whether to display the figure, by default True.
    line1_color : str, optional
        Color for first method's line. Can use shorthand or full name.
        Common shorthands: 'k' (black), 'r' (red), 'b' (blue), 'g' (green).
        Default is 'k'.
    line1_style : str, optional
        Line style for first method. Options: ['-', '--', '-.', ':'].
        Default is '-' (solid).
    line1_width : float, optional
        Width of first method's line, by default 1.0.
    line2_color : str, optional
        Color for second method's line. Same options as line1_color.
        Default is 'r'.
    line2_style : str, optional
        Line style for second method. Same options as line1_style.
        Default is '--' (dashed).
    line2_width : float, optional
        Width of second method's line, by default 1.0.
    marker_size : float, optional
        Size of markers at target error points, by default 4.
    grid_alpha : float, optional
        Transparency of grid lines (0 to 1), by default 0.2.
    ref_line_color : str, optional
        Color for reference line at target error. Same options as line1_color.
        Default is 'gray'.
    ref_line_style : str, optional
        Line style for reference line. Same options as line1_style.
        Default is ':' (dotted).
    ref_line_alpha : float, optional
        Transparency of reference line (0 to 1), by default 0.5.
    label1 : str, optional
        Label for first method in legend, by default 'Without Decay'.
    label2 : str, optional
        Label for second method in legend, by default 'With Decay'.
    dpi : int, optional
        Resolution of saved figure in dots per inch, by default 1200.
    output_dir : str, optional
        Directory to save the figure, by default 'Plots'.

    Notes
    -----
    The function automatically creates the output directory if it doesn't exist.
    Uses the publication style settings from set_publication_style().
    Plot shows error on logarithmic scale vs. iteration number.
    Markers indicate where each method reaches the target error.

    Examples
    --------
    >>> metrics_no_decay = {
    ...     'error_values': [1.0, 0.5, 0.1, 0.01],
    ...     'min_error': 0.01
    ... }
    >>> metrics_with_decay = {
    ...     'error_values': [1.0, 0.3, 0.05, 0.005],
    ...     'min_error': 0.005
    ... }
    >>> plot_error_comparison(
    ...     metrics_no_decay,
    ...     metrics_with_decay,
    ...     'error_comparison',
    ...     line1_color='navy',
    ...     line2_color='darkred'
    ... )
    """
    # Input validation
    valid_line_styles = ['-', '--', '-.', ':']
    for style in [line1_style, line2_style, ref_line_style]:
        if style not in valid_line_styles:
            raise ValueError(f"Line styles must be one of: {valid_line_styles}")
    
    for alpha in [grid_alpha, ref_line_alpha]:
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha values must be between 0 and 1")
    
    required_keys = ['error_values', 'min_error']
    if not all(key in metrics1 for key in required_keys):
        raise ValueError("metrics1 missing required keys")
    if not all(key in metrics2 for key in required_keys):
        raise ValueError("metrics2 missing required keys")

    # Setup
    set_publication_style()
    
    # Find target error and iterations to reach it
    target_error = max(metrics1['min_error'], metrics2['min_error'])
    iter1 = next(i for i, error in enumerate(metrics1['error_values']) 
                if error <= target_error)
    iter2 = next(i for i, error in enumerate(metrics2['error_values']) 
                if error <= target_error)
    
    # Setup plot range
    max_iter = max(iter1, iter2) + 1
    iterations = range(max_iter)
    
    # Create plot
    fig, ax = plt.subplots(figsize=get_figure_size(layout='single', journal_type='large'))
    
    # Plot error curves
    ax.semilogy(iterations, metrics1['error_values'][:max_iter], 
                color=line1_color,
                linestyle=line1_style,
                linewidth=line1_width,
                label=f'{label1} (target error at iter {iter1})')
    
    ax.semilogy(iterations, metrics2['error_values'][:max_iter], 
                color=line2_color,
                linestyle=line2_style,
                linewidth=line2_width,
                label=f'{label2} (target error at iter {iter2})')
    
    # Add target markers
    ax.plot(iter1, metrics1['error_values'][iter1], 
            color=line1_color, marker='o', markersize=marker_size)
    ax.plot(iter2, metrics2['error_values'][iter2], 
            color=line2_color, marker='o', markersize=marker_size)
    
    # Add reference line
    ax.axhline(y=target_error, 
               color=ref_line_color,
               linestyle=ref_line_style,
               alpha=ref_line_alpha)
    
    # Customize plot
    ax.grid(True, which='both', linestyle='-', alpha=grid_alpha)
    ax.legend(frameon=False)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
    
    plt.tight_layout()
    
    # Handle figure saving
    os.makedirs(output_dir, exist_ok=True)
    base_name, ext = os.path.splitext(figure_name)
    ext = ext if ext else ".png"
    
    plt.savefig(
        f'{output_dir}/{base_name}{ext}',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0
    )
    
    if show_figure:
        plt.show()
    
    plt.close()


# Fuction to plot the Polygon boundaries highlighting the short edges
def plot_boundary_with_short_edges(
    polygon,
    figure_name,
    N=None,
    threshold=None,
    show_figure=True,
    boundary_color='#1B365D',
    boundary_style='-',
    boundary_width=0.5,
    interior_color='r',
    interior_style='-',
    interior_width=0.5,
    short_edge_color='orange',
    short_edge_style='-',
    short_edge_width=2.0,
    short_edge_alpha=0.8,
    margins=0.01,
    dpi=1200,
    show_axes=False,
    output_dir='Plots'
):
    """Plot polygon boundaries with highlighted short edges based on either N shortest
    edges or a threshold length.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        The polygon(s) whose boundary is to be plotted.
    figure_name : str
        Name of the output figure. If no extension provided, '.png' is used.
    N : int, optional
        Number of shortest edges to highlight. Must specify either N or threshold,
        but not both.
    threshold : float, optional
        Length threshold below which edges are highlighted. Must specify either N
        or threshold, but not both.
    show_figure : bool, optional
        Whether to display the figure, by default True.
    boundary_color : str, optional
        Color for exterior boundaries. Can use shorthand or full name.
        Common shorthands: 'k' (black), 'r' (red), 'b' (blue), 'g' (green).
        Default is 'k'.
    boundary_style : str, optional
        Line style for exterior boundaries. Options: ['-', '--', '-.', ':'].
        Default is '-' (solid).
    boundary_width : float, optional
        Width of exterior boundary lines, by default 0.5.
    interior_color : str, optional
        Color for interior boundaries. Same color options as boundary_color.
        Default is 'r'.
    interior_style : str, optional
        Line style for interior boundaries. Same options as boundary_style.
        Default is '-' (solid).
    interior_width : float, optional
        Width of interior boundary lines, by default 0.5.
    short_edge_color : str, optional
        Color for highlighted short edges. Same color options as boundary_color.
        Default is 'orange'.
    short_edge_style : str, optional
        Line style for short edges. Same options as boundary_style.
        Default is '-' (solid).
    short_edge_width : float, optional
        Width of short edge lines, by default 2.0.
    short_edge_alpha : float, optional
        Transparency of short edges (0 to 1), by default 0.8.
    margins : float or tuple, optional
        Margins around the plot. Can be:
        - Single float for same margins in x and y
        - Tuple (x_margin, y_margin) for different margins
        Default is 0.01 (1% margin in both x and y).
    dpi : int, optional
        Resolution of saved figure in dots per inch, by default 1200.
    show_axes : bool, optional
        Whether to show axes and ticks, by default False.
    output_dir : str, optional
        Directory to save the figure, by default 'Plots'.

    Raises
    ------
    ValueError
        If neither or both N and threshold are specified.
        If N is negative.
        If threshold is negative.

    Notes
    -----
    The function automatically creates the output directory if it doesn't exist.
    Uses the publication style settings from set_publication_style().

    Examples
    --------
    >>> # Highlight 5 shortest edges
    >>> plot_boundary_with_short_edges(
    ...     polygon,
    ...     'boundary_short',
    ...     N=5,
    ...     short_edge_color='red',
    ...     short_edge_width=3.0
    ... )
    >>> 
    >>> # Highlight edges shorter than 0.1 units
    >>> plot_boundary_with_short_edges(
    ...     polygon,
    ...     'boundary_short',
    ...     threshold=0.1,
    ...     boundary_color='navy',
    ...     short_edge_color='orange'
    ... )
    """
    # Input validation
    if (N is None) == (threshold is None):
        raise ValueError("Specify exactly one of 'N' or 'threshold'")
    
    if N is not None and N < 0:
        raise ValueError("N must be non-negative")
        
    if threshold is not None and threshold < 0:
        raise ValueError("threshold must be non-negative")

    valid_line_styles = ['-', '--', '-.', ':']
    for style in [boundary_style, interior_style, short_edge_style]:
        if style not in valid_line_styles:
            raise ValueError(f"Line styles must be one of: {valid_line_styles}")

    if not 0 <= short_edge_alpha <= 1:
        raise ValueError("Alpha value must be between 0 and 1")

    # Setup
    set_publication_style()
    fig, ax = plt.subplots(figsize=get_figure_size(layout='single', journal_type='large'))
    ax.set_aspect('equal')
    # Set margins
    if isinstance(margins, (int, float)):
        ax.margins(margins)
    else:
        ax.margins(x=margins[0], y=margins[1])

    # Function to collect edges from a polygon
    def collect_edges_from_polygon(polygon):
        edges = []
        # Exterior edges
        exterior_coords = list(polygon.exterior.coords)
        for i in range(len(exterior_coords) - 1):
            line = LineString([exterior_coords[i], exterior_coords[i + 1]])
            edges.append((line.length, line))
        # Interior edges
        for interior in polygon.interiors:
            interior_coords = list(interior.coords)
            for i in range(len(interior_coords) - 1):
                line = LineString([interior_coords[i], interior_coords[i + 1]])
                edges.append((line.length, line))
        return edges

    # Collect all edges
    edges = []
    if isinstance(polygon, Polygon):
        edges.extend(collect_edges_from_polygon(polygon))
    else:  # MultiPolygon
        for poly in polygon.geoms:
            edges.extend(collect_edges_from_polygon(poly))

    # Determine short edges
    edges.sort(key=lambda x: x[0])
    if N is not None:
        short_edges = [edge for _, edge in edges[:N]]
    else:
        short_edges = [edge for length, edge in edges if length < threshold]

    # Plot polygon boundaries
    polygons = [polygon] if isinstance(polygon, Polygon) else polygon.geoms
    for poly in polygons:
        # Plot exterior
        x, y = poly.exterior.xy
        ax.plot(x, y,
                color=boundary_color,
                linestyle=boundary_style,
                linewidth=boundary_width)
        
        # Plot interiors
        for interior in poly.interiors:
            x, y = interior.xy
            ax.plot(x, y,
                   color=interior_color,
                   linestyle=interior_style,
                   linewidth=interior_width)

    # Plot short edges
    for edge in short_edges:
        x, y = edge.xy
        ax.plot(x, y,
                color=short_edge_color,
                linestyle=short_edge_style,
                linewidth=short_edge_width,
                alpha=short_edge_alpha,
                zorder=2)

    # Handle axes visibility
    if not show_axes:
        ax.axis('off')

    plt.tight_layout()

    # Handle figure saving
    os.makedirs(output_dir, exist_ok=True)
    base_name, ext = os.path.splitext(figure_name)
    ext = ext if ext else ".png"
    
    plt.savefig(
        f'{output_dir}/{base_name}{ext}',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0
    )
    
    if show_figure:
        plt.show()
    
    plt.close()


# Function to plot the density of plate with hole
def plot_density_plate_with_hole(
   rect_point1,
   rect_point2,
   circle_center,
   circle_radius,
   figure_name,
   max_density=1.0,
   min_density=0.1,
   decay_rate=0.5,
   resolution=200,
   show_figure=True,
   colormap='viridis',
   rect_color='k',
   rect_style='-',
   rect_width=0.5,
   circle_color='r',
   circle_style='--',
   circle_width=0.5,
   view_elevation=30,
   view_azimuth=45,
   margins=0.01,
   dpi=1200,
   show_axes=True,
   output_dir='Plots'
):
   """Create 2D heatmap and 3D surface plot of density function for a plate with hole.

   Parameters
   ----------
   rect_point1 : tuple
       Lower-left corner coordinates (x, y) of the rectangular plate.
   rect_point2 : tuple
       Upper-right corner coordinates (x, y) of the rectangular plate.
   circle_center : tuple
       Center coordinates (x, y) of the circular hole.
   circle_radius : float
       Radius of the circular hole.
   figure_name : str
       Base name for the output figures. If no extension provided, '.png' is used.
       Two files will be created with suffixes: '_2DHeatmap' and '_3DSurface'.
   max_density : float, optional
       Maximum density value at circle boundary, by default 1.0.
   min_density : float, optional
       Minimum density value far from circle, by default 0.1.
   decay_rate : float, optional
       Rate of density decay with distance from circle, by default 0.5.
   resolution : int, optional
       Grid resolution for visualization, by default 200.
   show_figure : bool, optional
       Whether to display the figures, by default True.
   colormap : str, optional
       Matplotlib colormap name, by default 'viridis'.
   rect_color : str, optional
       Color for rectangle boundary. Can use shorthand or full name.
       Default is 'k'.
   rect_style : str, optional
       Line style for rectangle. Options: ['-', '--', '-.', ':'].
       Default is '-' (solid).
   rect_width : float, optional
       Width of rectangle boundary line, by default 0.5.
   circle_color : str, optional
       Color for circle boundary. Same options as rect_color.
       Default is 'r'.
   circle_style : str, optional
       Line style for circle. Same options as rect_style.
       Default is '--' (dashed).
   circle_width : float, optional
       Width of circle boundary line, by default 0.5.
   view_elevation : float, optional
       Elevation angle for 3D view, by default 30.
   view_azimuth : float, optional
       Azimuth angle for 3D view, by default 45.
   margins : float or tuple, optional
       Margins around the plot. Can be:
       - Single float for same margins in x and y
       - Tuple (x_margin, y_margin) for different margins
       Default is 0.01 (1% margin).
   dpi : int, optional
       Resolution of saved figures in dots per inch, by default 1200.
   show_axes : bool, optional
       Whether to show axes and ticks, by default True.
   output_dir : str, optional
       Directory to save the figures, by default 'Plots'.

   Notes
   -----
   The function automatically creates the output directory if it doesn't exist.
   Uses the publication style settings from set_publication_style().
   
   The density function uses an exponential decay from the circle boundary:
   density = max_density * exp(-decay_rate * distance) + min_density

   Examples
   --------
   >>> # Create density plots for a square plate with centered hole
   >>> plot_density_plate_with_hole(
   ...     rect_point1=(-1, -1),
   ...     rect_point2=(1, 1),
   ...     circle_center=(0, 0),
   ...     circle_radius=0.3,
   ...     figure_name='density_plot',
   ...     decay_rate=1.0
   ... )
   """
   # Input validation
   valid_line_styles = ['-', '--', '-.', ':']
   for style in [rect_style, circle_style]:
       if style not in valid_line_styles:
           raise ValueError(f"Line styles must be one of: {valid_line_styles}")

   if not 0 < min_density <= max_density:
       raise ValueError("min_density must be positive and not greater than max_density")
   
   if decay_rate <= 0:
       raise ValueError("decay_rate must be positive")
   
   if resolution < 10:
       raise ValueError("resolution must be at least 10")

   def create_density_function(circle_center, circle_radius):
       """Create density function based on distance from circle boundary."""
       def distance_to_circle(x, y):
           dist_to_center = np.sqrt((x - circle_center[0])**2 + 
                                  (y - circle_center[1])**2)
           return abs(dist_to_center - circle_radius)
       
       def density_function(x, y):
           dist = distance_to_circle(x, y)
           density = max_density * np.exp(-decay_rate * dist) + min_density
           return np.clip(density, min_density, max_density)
       
       return density_function

   # Setup
   set_publication_style()
   density = create_density_function(circle_center, circle_radius)

   # Create grid
   x = np.linspace(rect_point1[0], rect_point2[0], resolution)
   y = np.linspace(rect_point1[1], rect_point2[1], resolution)
   X, Y = np.meshgrid(x, y)
   Z = np.vectorize(density)(X, Y)

   # Prepare file names
   os.makedirs(output_dir, exist_ok=True)
   base_name, ext = os.path.splitext(figure_name)
   ext = ext if ext else ".png"

   # Plot 2D Heatmap
   fig, ax = plt.subplots(figsize=get_figure_size(layout='single', 
                                                 journal_type='large'))

   # Set margins for 2D plot
   if isinstance(margins, (int, float)):
       ax.margins(margins)
   else:
       ax.margins(x=margins[0], y=margins[1])

   heatmap = ax.pcolormesh(X, Y, Z, shading='auto', cmap=colormap)
   cbar = plt.colorbar(heatmap, ax=ax)
   cbar.set_label('Density', fontsize=10)
   cbar.ax.tick_params(labelsize=10)

   # Add geometry
   circle = plt.Circle(circle_center, circle_radius, fill=False,
                      color=circle_color, linestyle=circle_style,
                      linewidth=circle_width)
   rect = plt.Rectangle(rect_point1, 
                       rect_point2[0] - rect_point1[0],
                       rect_point2[1] - rect_point1[1],
                       fill=False, color=rect_color,
                       linestyle=rect_style, linewidth=rect_width)
   ax.add_artist(circle)
   ax.add_artist(rect)
   ax.set_aspect('equal')

   if not show_axes:
       ax.axis('off')
   else:
       ax.set_xlabel('$x$')
       ax.set_ylabel('$y$')

   plt.tight_layout()
   plt.savefig(f'{output_dir}/{base_name}_2DHeatmap{ext}',
               dpi=dpi, bbox_inches='tight', pad_inches=0)
   if show_figure:
       plt.show()
   plt.close()

   # Plot 3D Surface
   fig = plt.figure(figsize=get_figure_size(layout='single', 
                                          journal_type='large'))
   ax3d = fig.add_subplot(111, projection='3d')

   # Set margins for 3D plot
   if isinstance(margins, (int, float)):
       ax3d.margins(margins)
   else:
       ax3d.margins(x=margins[0], y=margins[1])

   surface = ax3d.plot_surface(X, Y, Z, cmap=colormap,
                              linewidth=0, antialiased=True)

   cbar = plt.colorbar(surface, ax=ax3d, pad=0.15)
   cbar.set_label('Density', fontsize=10)
   cbar.ax.tick_params(labelsize=10)

   ax3d.view_init(elev=view_elevation, azim=view_azimuth)
   
   if not show_axes:
       ax3d.set_axis_off()
   else:
       ax3d.set_xlabel('$x$')
       ax3d.set_ylabel('$y$')
       ax3d.set_zlabel('Density')

   plt.tight_layout()
   plt.savefig(f'{output_dir}/{base_name}_3DSurface{ext}',
               dpi=dpi, bbox_inches='tight', pad_inches=0)
   if show_figure:
       plt.show()
   plt.close()


# Function to plot hexagon density
def plot_density_hexagon(
    hexagon_geometry,
    density_function,
    figure_name,
    resolution=200,
    show_figure=True,
    margins=0.01,
    colormap='viridis',
    hex_color='r',
    hex_style='--',
    hex_width=0.5,
    view_elevation=30,
    view_azimuth=45,
    dpi=1200,
    show_axes=True,
    output_dir='Plots'
):
    """Create 2D heatmap and 3D surface plot of density function over a hexagonal region.

    Parameters
    ----------
    hexagon_geometry : dict
        Dictionary containing the hexagonal geometry with key:
        - 'polygon': shapely.geometry.Polygon object representing the hexagon
    density_function : callable
        Function that takes (x, y) arrays and returns density values.
        Must be vectorized to handle numpy arrays.
    figure_name : str
        Base name for the output figures. If no extension provided, '.png' is used.
        Two files will be created with suffixes: '_2DHeatmap' and '_3DSurface'.
    resolution : int, optional
        Grid resolution for visualization, by default 200.
    show_figure : bool, optional
        Whether to display the figures, by default True.
    margins : float or tuple, optional
        Margins around the plot. Can be:
        - Single float for same margins in x and y
        - Tuple (x_margin, y_margin) for different margins
        Default is 0.01 (1% margin).
    colormap : str, optional
        Matplotlib colormap name, by default 'viridis'.
    hex_color : str, optional
        Color for hexagon boundary. Can use shorthand or full name.
        Common shorthands: 'k' (black), 'r' (red), 'b' (blue), 'g' (green).
        Default is 'r'.
    hex_style : str, optional
        Line style for hexagon. Options: ['-', '--', '-.', ':'].
        Default is '--' (dashed).
    hex_width : float, optional
        Width of hexagon boundary line, by default 0.5.
    view_elevation : float, optional
        Elevation angle for 3D view, by default 30.
    view_azimuth : float, optional
        Azimuth angle for 3D view, by default 45.
    dpi : int, optional
        Resolution of saved figures in dots per inch, by default 1200.
    show_axes : bool, optional
        Whether to show axes and ticks, by default True.
    output_dir : str, optional
        Directory to save the figures, by default 'Plots'.

    Notes
    -----
    The function automatically creates the output directory if it doesn't exist.
    Uses the publication style settings from set_publication_style().
    Points outside the hexagon are masked with NaN values.

    Examples
    --------
    >>> # Create hexagon with constant density
    >>> hex_geom = {'polygon': create_hexagon(center=(0, 0), radius=1)}
    >>> constant_density = lambda x, y: np.ones_like(x)
    >>> plot_density_hexagon(
    ...     hex_geom,
    ...     constant_density,
    ...     'hex_density',
    ...     hex_color='blue',
    ...     colormap='plasma'
    ... )
    """
    # Input validation
    valid_line_styles = ['-', '--', '-.', ':']
    if hex_style not in valid_line_styles:
        raise ValueError(f"Line style must be one of: {valid_line_styles}")

    if resolution < 10:
        raise ValueError("resolution must be at least 10")

    if not callable(density_function):
        raise ValueError("density_function must be callable")

    # Setup
    set_publication_style()
    hex_polygon = hexagon_geometry['polygon']

    # Get domain bounds
    minx, miny, maxx, maxy = hex_polygon.bounds
    x = np.linspace(minx, maxx, resolution)
    y = np.linspace(miny, maxy, resolution)
    X, Y = np.meshgrid(x, y)

    # Create mask for points inside hexagon
    points = [Point(x, y) for x, y in zip(X.flatten(), Y.flatten())]
    mask = np.array([hex_polygon.contains(p) for p in points]).reshape(X.shape)
    Z = np.where(mask, density_function(X, Y), np.nan)

    # Prepare file names
    os.makedirs(output_dir, exist_ok=True)
    base_name, ext = os.path.splitext(figure_name)
    ext = ext if ext else ".png"

    # Plot 2D Heatmap
    fig, ax = plt.subplots(figsize=get_figure_size(layout='single', 
                                                  journal_type='large'))
    # Set margins for 2D plot
    if isinstance(margins, (int, float)):
        ax.margins(margins)
    else:
        ax.margins(x=margins[0], y=margins[1])

    heatmap = ax.pcolormesh(X, Y, Z, shading='auto', cmap=colormap)
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Density', fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    # Draw hexagon boundary
    x, y = hex_polygon.exterior.xy
    ax.plot(x, y, color=hex_color, linestyle=hex_style, 
            linewidth=hex_width)

    ax.set_aspect('equal')
    if not show_axes:
        ax.axis('off')
    else:
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{base_name}_2DHeatmap{ext}',
                dpi=dpi, bbox_inches='tight', pad_inches=0)
    if show_figure:
        plt.show()
    plt.close()

    # Plot 3D Surface
    fig = plt.figure(figsize=get_figure_size(layout='single', 
                                           journal_type='large'))
    ax3d = fig.add_subplot(111, projection='3d')

    # Set margins for 3D plot
    if isinstance(margins, (int, float)):
        ax3d.margins(margins)
    else:
        ax3d.margins(x=margins[0], y=margins[1])

    surface = ax3d.plot_surface(X, Y, Z, cmap=colormap,
                               linewidth=0, antialiased=True)

    cbar = plt.colorbar(surface, ax=ax3d, pad=0.15)
    cbar.set_label('Density', fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    ax3d.view_init(elev=view_elevation, azim=view_azimuth)
    
    if not show_axes:
        ax3d.set_axis_off()
    else:
        ax3d.set_xlabel('$x$')
        ax3d.set_ylabel('$y$')
        ax3d.set_zlabel('Density')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{base_name}_3DSurface{ext}',
                dpi=dpi, bbox_inches='tight', pad_inches=0)
    if show_figure:
        plt.show()
    plt.close()


# Plotting the modified Vroronoi cells with the introduced gaps
def plot_voronoi_cells_with_gaps(
    original_cells,
    adjusted_cells,
    figure_name,
    show_figure=True,
    cell_color='#1B365D',
    cell_style='-',
    cell_width=0.5,
    gap_color='r',
    gap_alpha=1.0,
    margins=0.01,
    dpi=1200,
    show_axes=False,
    output_dir='Plots'
):
    """Plot Voronoi cells with highlighted gaps between adjacent cells.

    Parameters
    ----------
    original_cells : list
        List of shapely.geometry.Polygon objects representing original Voronoi cells
        before adjustment.
    adjusted_cells : list
        List of shapely.geometry.Polygon objects representing adjusted Voronoi cells
        with gaps.
    figure_name : str
        Name of the output figure. If no extension provided, '.png' is used.
    show_figure : bool, optional
        Whether to display the figure, by default True.
    cell_color : str, optional
        Color for cell boundaries. Can use shorthand or full name.
        Common shorthands: 'k' (black), 'r' (red), 'b' (blue), 'g' (green).
        Default is 'k'.
    cell_style : str, optional
        Line style for cell boundaries. Options: ['-', '--', '-.', ':'].
        Default is '-' (solid).
    cell_width : float, optional
        Width of cell boundary lines, by default 0.5.
    gap_color : str, optional
        Color to fill the gaps between cells, by default 'lightgray'.
    gap_alpha : float, optional
        Transparency of gap filling (0 to 1), by default 0.5.
    margins : float or tuple, optional
        Margins around the plot. Can be:
        - Single float for same margins in x and y
        - Tuple (x_margin, y_margin) for different margins
        Default is 0.01 (1% margin in both x and y).
    dpi : int, optional
        Resolution of saved figure in dots per inch, by default 1200.
    show_axes : bool, optional
        Whether to show axes and ticks, by default False.
    output_dir : str, optional
        Directory to save the figure, by default 'Plots'.

    Notes
    -----
    The function automatically creates the output directory if it doesn't exist.
    Uses the publication style settings from set_publication_style().
    Gaps are visualized by plotting the difference between original and adjusted cells.

    Examples
    --------
    >>> # Plot cells with default gap styling
    >>> plot_voronoi_cells_with_gaps(
    ...     original_cells,
    ...     adjusted_cells,
    ...     'cells_with_gaps'
    ... )
    >>> 
    >>> # Custom styling for cells and gaps
    >>> plot_voronoi_cells_with_gaps(
    ...     original_cells,
    ...     adjusted_cells,
    ...     'cells_with_gaps',
    ...     cell_color='navy',
    ...     gap_color='lightblue',
    ...     gap_alpha=0.3
    ... )
    """
    # Input validation
    valid_line_styles = ['-', '--', '-.', ':']
    if cell_style not in valid_line_styles:
        raise ValueError(f"Line style must be one of: {valid_line_styles}")
    
    if not 0 <= gap_alpha <= 1:
        raise ValueError("gap_alpha must be between 0 and 1")

    # Setup
    set_publication_style()
    fig, ax = plt.subplots(figsize=get_figure_size(layout='single', 
                                                  journal_type='large'))
    ax.set_aspect('equal')

    # Set margins
    if isinstance(margins, (int, float)):
        ax.margins(margins)
    else:
        ax.margins(x=margins[0], y=margins[1])

    # First plot the gaps
    # Get the union of original and adjusted cells
    from shapely.ops import unary_union
    original_region = unary_union(original_cells)
    adjusted_region = unary_union(adjusted_cells)
    
    # The gaps are the difference between original and adjusted regions
    gaps = original_region.difference(adjusted_region)
    
    # Plot gaps first (they will be behind the cell boundaries)
    if isinstance(gaps, Polygon):
        gaps = MultiPolygon([gaps])
    
    for gap in gaps.geoms:
        # Fill the gap area
        x, y = gap.exterior.xy
        ax.fill(x, y, facecolor=gap_color, alpha=gap_alpha, edgecolor='none')
        
        # Handle any holes in the gaps
        for interior in gap.interiors:
            x, y = interior.xy
            ax.fill(x, y, facecolor='white', edgecolor='none')

    # Then plot the adjusted cells on top
    for cell in adjusted_cells:
        # Plot cell boundary
        x, y = cell.exterior.xy
        ax.plot(x, y,
                color=cell_color,
                linestyle=cell_style,
                linewidth=cell_width)
        
        # Plot interior boundaries if any
        for interior in cell.interiors:
            x, y = interior.xy
            ax.plot(x, y,
                   color=cell_color,
                   linestyle=cell_style,
                   linewidth=cell_width)

    # Handle axes visibility
    if not show_axes:
        ax.axis('off')

    plt.tight_layout()

    # Handle figure saving
    os.makedirs(output_dir, exist_ok=True)
    base_name, ext = os.path.splitext(figure_name)
    ext = ext if ext else ".png"
    
    plt.savefig(
        f'{output_dir}/{base_name}{ext}',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0
    )
    
    if show_figure:
        plt.show()
    
    plt.close()


# Create a desired grid layout from the provided figures
def create_figure_grid(
    figure_paths,
    output_name,
    layout='1x2',
    figsize=None,
    show_figure=True,
    label_fontsize=10,
    label_distance=0.05,
    dpi=1200,
    output_dir='Plots'
):
    """Create a publication-quality subplot grid from saved figures.

    Parameters
    ----------
    figure_paths : list
        List of paths to individual figure images to be included in the grid.
    output_name : str
        Name of the output file. If no extension provided, '.png' is used.
    layout : str, optional
        Layout of the grid. Must be one of:
        - '1x2': 1 row, 2 columns
        - '1x3': 1 row, 3 columns
        - '2x2': 2 rows, 2 columns
        - '2x3': 2 rows, 3 columns
        - '3x3': 3 rows, 3 columns
        Default is '1x2'.
    figsize : tuple, optional
        Custom figure size as (width, height) in inches.
        If None, uses preset sizes based on layout.
    show_figure : bool, optional
        Whether to display the figure, by default True.
    label_fontsize : float, optional
        Font size for subplot labels, by default 10.
    label_distance : float, optional
        Distance of labels below subplots (in axis units), by default 0.05.
    dpi : int, optional
        Resolution of saved figure in dots per inch, by default 1200.
    output_dir : str, optional
        Directory to save the figure, by default 'Plots'.

    Raises
    ------
    ValueError
        If layout is not one of the supported options.
        If figsize is provided but not a tuple of length 2.
        If number of figure_paths exceeds subplot count for layout.

    Notes
    -----
    The function automatically creates the output directory if it doesn't exist.
    Uses the publication style settings from set_publication_style().
    Subplots are labeled alphabetically ((a), (b), etc.) below each figure.
    Empty subplots are removed if fewer figures than grid spaces.

    Examples
    --------
    >>> # Create a 1x2 grid from two figures
    >>> figure_paths = ['figure1.png', 'figure2.png']
    >>> create_figure_grid(
    ...     figure_paths,
    ...     'combined_figure',
    ...     layout='1x2',
    ...     label_fontsize=12
    ... )
    >>> 
    >>> # Create a 2x2 grid with custom size
    >>> figure_paths = ['fig1.png', 'fig2.png', 'fig3.png', 'fig4.png']
    >>> create_figure_grid(
    ...     figure_paths,
    ...     'quad_figure',
    ...     layout='2x2',
    ...     figsize=(10, 10)
    ... )
    """
    # Input validation
    layout_configs = {
        '1x2': (1, 2, {'figsize': get_figure_size(layout='1x2', 
                                                 journal_type='large')}),
        '1x3': (1, 3, {'figsize': get_figure_size(layout='1x3', 
                                                 journal_type='large')}),
        '2x2': (2, 2, {'figsize': get_figure_size(layout='2x2', 
                                                 journal_type='large')}),
        '2x3': (2, 3, {'figsize': get_figure_size(layout='2x3', 
                                                 journal_type='large')}),
        '3x3': (3, 3, {'figsize': get_figure_size(layout='3x3', 
                                                 journal_type='large')})
    }

    if layout not in layout_configs:
        raise ValueError(f"Layout must be one of: {list(layout_configs.keys())}")
    
    rows, cols, config = layout_configs[layout]
    max_subplots = rows * cols
    
    if len(figure_paths) > max_subplots:
        raise ValueError(
            f"Number of figures ({len(figure_paths)}) exceeds "
            f"subplot count ({max_subplots}) for layout {layout}"
        )

    if figsize is not None:
        if not isinstance(figsize, tuple) or len(figsize) != 2:
            raise ValueError("figsize must be a tuple of (width, height)")
        config['figsize'] = figsize

    # Setup
    set_publication_style()
    fig, axs = plt.subplots(rows, cols, **config)
    axs = axs.flatten()

    # Plot figures
    for idx, (ax, path) in enumerate(zip(axs, figure_paths)):
        # Load and display image
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis('off')
        
        # Add subplot label
        ax.text(0.5, -label_distance, f'({chr(97 + idx)})',
                transform=ax.transAxes,
                fontsize=label_fontsize,
                verticalalignment='top',
                horizontalalignment='center')

    # Remove empty subplots
    for ax in axs[len(figure_paths):]:
        ax.remove()

    plt.tight_layout()

    # Handle figure saving
    os.makedirs(output_dir, exist_ok=True)
    base_name, ext = os.path.splitext(output_name)
    ext = ext if ext else ".png"
    
    plt.savefig(
        f'{output_dir}/{base_name}{ext}',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0
    )
    
    if show_figure:
        plt.show()
    
    plt.close()
