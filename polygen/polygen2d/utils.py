import os
import re
import time
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, LineString

from .geometry import Geometry
from .IO import IO
from .constrainedPointGen import generate_poisson_points
from .errorBasedLloyd import lloyd_with_density as lloyd
from .voroGen import VoronoiGenerator
from .optimize import MeshOptimizer
from .finiteThicknessCohesiveZone import CohesiveZoneAdjuster
from .triangularMeshing import triangulate_geometry
from .plotting import plot_boundary_with_points, plot_voronoi_cells, plot_voronoi_cells_with_short_edges, plot_triangulated_geometry

# Function to read the input file
def parse_input_file(input_file_path):
    parameters = {
        "boundary": None,
        "N_points": 100,
        "points_seed": 42,
        "margin": 0.01,
        "N_iter": 100000,
        "use_decay": True,
        "edgeLength_threshold": None,
        "cohesiveThickness": None,
        "triangularMesh_minLength": None,
        "visualize": False,
        "saveData": True
    }

    with open(input_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            # Skip empty lines or lines that don't contain an '=' or are commented out
            if '=' not in line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")  # Remove any surrounding quotes

            if key in parameters:
                if value.lower() == 'true':
                    parameters[key] = True
                elif value.lower() == 'false':
                    parameters[key] = False
                elif value.isdigit():
                    parameters[key] = int(value)
                else:
                    try:
                        parameters[key] = float(value)
                    except ValueError:
                        parameters[key] = value

    # Check if boundary is a method of Geometry
    boundary_value = parameters["boundary"]
    if boundary_value is not None:
        try:
            # Evaluate the boundary value if it's a Geometry method
            parameters["boundary"] = eval(boundary_value, {"Geometry": Geometry, "Point": Point})
        except (NameError, SyntaxError):
            # If it's not a Geometry method, treat it as a file path
            parameters["boundary"] = os.path.abspath(boundary_value)

    return parameters

# Function to generate variables' name dynamically based on the .obj file name
def generate_variable_names(input_obj):
    """
    Generate variable names dynamically based on the given geometry object or .obj file.
    
    Parameters:
    - input_obj: Path to the .obj file or a predefined geometric object from the Geometry class.
    
    Returns:
    - A dictionary with dynamically generated variable names based on the input object's name or file name.
    """
    if isinstance(input_obj, str) and input_obj.endswith('.obj'):
        base_name = os.path.splitext(os.path.basename(input_obj))[0]
    elif isinstance(input_obj, dict) and 'name' in input_obj:
        base_name = input_obj['name']
    else:
        raise ValueError("Input must be a path to a .obj file or a predefined geometric object with a 'name' attribute.")

    variable_names = {
        "base_name": base_name,
        "polygon_region": f"polygon_region_{base_name}",
        "seed_points": f"seed_points_{base_name}",
        "relaxed_points": f"relaxed_points_{base_name}",
        "voronoi_cells": f"voronoi_cells_{base_name}",
        "collapsed_voronoi_cells": f"collapsed_voronoi_cells_{base_name}",
        "adjusted_voronoi_cells": f"adjusted_voronoi_cells_{base_name}",
        "collapsed_polygonRegion": f"collapsed_polygonRegion_{base_name}",
        "triangular_mesh": f"triangular_mesh_{base_name}",
        "figure_boundary_with_lloyds_seed_points": f"{base_name}_boundary_with_Lloyds_seed_points.png",
        "figure_voronoi_with_short_edges": f"{base_name}_Voronoi_with_short_edges.png",
        "figure_original_voronoi_cells": f"{base_name}_original_Voronoi_cells.png",
        "figure_collapsed_voronoi_cells": f"{base_name}_collapsed_Voronoi_cells.png",
        "figure_adjusted_voronoi_cells": f"{base_name}_adjusted_Voronoi_cells.png",
        "figure_triangulated_geometry": f"{base_name}_triangulated_geometry.png",
        "polygon_data_file": f"polygon_dataFile_{base_name}.py",
        "polygon_data": f"polygon_boundariesData_{base_name}",
        "voronoi_data_file": f"Voronoi_dataFile_{base_name}.py",
        "voronoi_data": f"Voronoi_data_{base_name}"
    }

    return variable_names

# Function to create the 2D Voronoi mesh from the given parameters
def computeVoronoi2d(boundary, N_points, points_seed=42, margin=0.01, N_iter=100000, use_decay=True, edgeLength_threshold=None, cohesiveThickness=None, triangularMesh_minLength=None,  visualize=False, saveData=True):
    """
    Compute and visualize Voronoi diagrams within an arbitrarily shaped 2D region.
    
    This function integrates various stages of Voronoi diagram generation and provides 
    visualization and data-saving capabilities.

    Parameters
    ----------
    boundary : str or dict
        The path to the OBJ file defining the boundary or predefined geometry object
    N_points : int
        Number of initial seed points for the Voronoi diagram
    points_seed : int, optional
        Random seed for generating initial Poisson disk sampling points (default: 42)
    margin : float, optional
        Margin to add around the boundary for generating seed points (default: 0.01)
    N_iter : int, optional
        Number of iterations for Lloyd's algorithm to relax the points (default: 100000)
    use_decay : bool, optional
        Whether to use decay in Lloyd's algorithm (default: True)
    edgeLength_threshold : float, optional
        Length threshold for collapsing short edges in the Voronoi diagram (default: None)
    cohesiveThickness : float, optional
        Thickness of the cohesive zone for adjusting the Voronoi cells (default: None)
    triangularMesh_minLength : float, optional
        Minimum edge length for triangular meshing (default: None)
    visualize : bool, optional
        Whether to show visualization plots (default: False)
    saveData : bool, optional
        Whether to save the resulting data structures to files (default: True)

    Use Cases
    ---------
    1. Basic Voronoi Diagram Generation:
       Generate and visualize a Voronoi diagram within a given boundary.
       Example: computeVoroni('boundary.obj', 100, visualize=True)
    
    2. Optimized Voronoi Diagram with Edge Collapsing:
       Generate a Voronoi diagram and collapse edges shorter than a threshold.
       Example: computeVoroni('boundary.obj', 100, edgeLength_threshold=0.1)
    
    3. Data Saving:
       Save the resulting Voronoi cell and boundary data to files.
       Example: computeVoroni('boundary.obj', 100, saveData=True)

    Workflow
    --------
    1. Load the polygonal boundary from the specified OBJ file
    2. Generate initial Poisson disk sampling points within the boundary
    3. Relax the points using Lloyd's algorithm
    4. Generate Voronoi cells based on the relaxed points
    5. Optionally collapse short edges and optimize the mesh
    6. Visualize results and save plots if requested
    7. Save data to files if requested

    Example
    -------
    >>> computeVoroni(
    ...     boundary='boundary.obj',
    ...     N_points=100,
    ...     points_seed=42,
    ...     margin=0.01,
    ...     N_iter=100000,
    ...     use_decay=True,
    ...     edgeLength_threshold=0.1,
    ...     cohesiveThickness=0.1,
    ...     triangularMesh_minLength=0.02,
    ...     visualize=True,
    ...     saveData=True
    ... )
    """
    start_time = time.time()  # Start timing

    names = generate_variable_names(boundary)

    # Initialize Voronoi generators and optimizers
    voronoi_generator = VoronoiGenerator(buffer_factor=1.0)
    optimizer = MeshOptimizer(verbose=True)

    # Initialize a dictionary to hold the dynamically named variables
    variables = {}
    
    # Use the generated variable names
    variables[names["polygon_region"]] = IO.load_polygon_from_file(boundary)
    variables[names["seed_points"]] = generate_poisson_points(domain=variables[names["polygon_region"]], N=N_points, seed=points_seed, margin=margin)
    print('')
    print("Applying Lloyd's algorithm to relax the seed points...")
    print('')
    variables[names["relaxed_points"]], _ = lloyd(polygon=variables[names["polygon_region"]], seed_points=variables[names["seed_points"]], max_iterations=N_iter, use_decay=use_decay)
    variables[names["voronoi_cells"]] = voronoi_generator.generate_cells(domain=variables[names["polygon_region"]], points=variables[names["relaxed_points"]])
    
    print('')
    optimizer.analyze_voronoi_edges(variables[names["voronoi_cells"]], n=10)
    get_voronoi_stats(variables[names["voronoi_cells"]], N=10, file_name='Report_OriginalVoronoi.txt')
    print('')
    # print('==========================================================')
    # print('')
    # print_short_boundaryEdges(variables[names["polygon_region"]], N=10)
    print('==========================================================')
    print('==========================================================')
    print('')
    
    if edgeLength_threshold is not None:
        variables[names["collapsed_voronoi_cells"]] = optimizer.optimize_voronoi_cells(variables[names["voronoi_cells"]], threshold=edgeLength_threshold)
        variables[names["collapsed_polygonRegion"]] = optimizer.optimize_boundary(variables[names["polygon_region"]], threshold=edgeLength_threshold)
        get_voronoi_stats(variables[names["collapsed_voronoi_cells"]], N=10, file_name='Report_OptimizedVoronoi.txt')

    voronoiCompute_time = time.time()  # End timing
    elapsed_time = voronoiCompute_time - start_time

    # Print the execution time in an excellent format
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Voronoi Computation Time: {int(hours):02} Hours: {int(minutes):02} Minutes: {seconds:05.2f} Seconds")
    print('')
    print('==========================================================')
    print('')

    # Plotting functions
    plot_boundary_with_points(figure_name=names["figure_boundary_with_lloyds_seed_points"], polygon=variables[names["polygon_region"]], points=variables[names["relaxed_points"]], show_figure=visualize)
    plot_voronoi_cells(figure_name=names["figure_original_voronoi_cells"], voronoi_cells=variables[names["voronoi_cells"]], show_figure=visualize)
    
    if edgeLength_threshold is not None:
        plot_voronoi_cells_with_short_edges(figure_name=names["figure_voronoi_with_short_edges"], voronoi_cells=variables[names["voronoi_cells"]], threshold=edgeLength_threshold, show_figure=visualize)
        plot_voronoi_cells(figure_name=names["figure_collapsed_voronoi_cells"], voronoi_cells=variables[names["collapsed_voronoi_cells"]], show_figure=visualize)

    if cohesiveThickness is not None:
        cohesive_adjuster = CohesiveZoneAdjuster(tolerance=0.005, max_iterations=10, verbose=True)
        cells = variables[names["collapsed_voronoi_cells"]] if edgeLength_threshold is not None else variables[names["voronoi_cells"]]
        variables[names["adjusted_voronoi_cells"]] = cohesive_adjuster.adjust_fixed_thickness(cells=cells, thickness=cohesiveThickness)
        plot_voronoi_cells(figure_name=names["figure_adjusted_voronoi_cells"], voronoi_cells=variables[names["adjusted_voronoi_cells"]], show_figure=visualize)
    
    # Triangular Meshing (Optional)
    if triangularMesh_minLength is not None:
        geometry = variables[names["adjusted_voronoi_cells"]] if cohesiveThickness is not None else (variables[names["collapsed_voronoi_cells"]] if edgeLength_threshold is not None else variables[names["voronoi_cells"]])
        variables[names["triangular_mesh"]] = triangulate_geometry(geometry=geometry, min_edge_length=triangularMesh_minLength, save_mesh=saveData, filename=f"{names['base_name']}_TriangularMesh.inp", file_format='abaqus')
        plot_triangulated_geometry(geometry=geometry, triangulation=variables[names["triangular_mesh"]], figure_name=names["figure_triangulated_geometry"], show_figure=visualize)

    if saveData:
        if cohesiveThickness is not None:
            target_cells = variables[names["adjusted_voronoi_cells"]]
            if edgeLength_threshold is not None:
                original_cells = variables[names["collapsed_voronoi_cells"]]
            else:
                original_cells = variables[names["voronoi_cells"]]
        else:
            if edgeLength_threshold is not None:
                target_cells = variables[names["collapsed_voronoi_cells"]]
            else:
                target_cells = variables[names["voronoi_cells"]]
            original_cells = None

        if edgeLength_threshold is not None:
            variables[names["polygon_region"]] = variables[names["collapsed_polygonRegion"]]
        
        # Saving functions
        print('')
        IO.save_polygon_data(polygon=variables[names["polygon_region"]], filename=names["polygon_data_file"], data_name=names["polygon_data"])
        print('')
        IO.save_voronoi_data(target_cells=target_cells, filename=names["voronoi_data_file"], data_name=names["voronoi_data"], original_cells=original_cells)
        print('')
        IO.save_voronoi_to_obj(voronoi_cells=target_cells, filename=f"{names['base_name']}_Voronoi.obj")
        

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time

    # Print the execution time in an excellent format
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total Execution Time: {int(hours):02} Hours: {int(minutes):02} Minutes: {seconds:05.2f} Seconds")
    print('')
    print('==========================================================')


# Function to compute 2D Voronoi based on the given input file
def computeVoronoi2d_fromInputFile(input_file_path):
    """
    Compute the 2D Voronoi mesh from parameters specified in an input file.

    This function reads the parameters from an input file and then calls the 
    `computeVoronoi2d` function to generate the Voronoi diagram based on those parameters.

    Parameters:
    - input_file_path (str): The path to the input file containing the parameters.

    The input file should have the following format:
    ```
    boundary = /path/to/boundary.obj
    N_points = 100
    points_seed = 42
    margin = 0.01
    N_iter = 100000
    use_decay = True
    edgeLength_threshold = 0.1
    cohesiveThickness = 0.1
    triangularMesh_minLength = 0.02
    visualize = False
    saveData = True
    ```

    Example usage:
    ```
    # Assuming the input file is located at './examples/input.in'
    computeVoronoi2d_fromInputFile('./examples/input.in')
    ```

    The function extracts the following parameters from the input file:
    - boundary: The path to the OBJ file defining the boundary.
    - N_points: Number of initial seed points for the Voronoi diagram.
    - points_seed: Random seed for generating initial Poisson disk sampling points.
    - margin: Margin to add around the boundary for generating seed points.
    - N_iter: Number of iterations for Lloyd's algorithm to relax the points.
    - use_decay: Boolean indicating whether to use decay in Lloyd's algorithm.
    - edgeLength_threshold: Length threshold for collapsing short edges in the Voronoi diagram.
    - cohesiveThickness: Thickness of the cohesive zone for adjusting the Voronoi cells.
    - triangularMesh_minLength: Minimum edge length for triangular meshing.
    - saveData: Boolean indicating whether to save the resulting data structures to .py files.
    """
    parameters = parse_input_file(input_file_path)

    # Extract parameters from the dictionary
    boundary = parameters["boundary"]
    N_points = parameters["N_points"]
    points_seed = parameters["points_seed"]
    margin = parameters["margin"]
    N_iter = parameters["N_iter"]
    use_decay = parameters["use_decay"]
    edgeLength_threshold = parameters["edgeLength_threshold"]
    cohesiveThickness = parameters["cohesiveThickness"]
    triangularMesh_minLength = parameters["triangularMesh_minLength"]
    visualize = parameters["visualize"]
    saveData = parameters["saveData"]

    if boundary is None:
        print('Please at least provide an accurate path of the OBJ file as the boundary in the input file')
        return

    # Call the computeVoronoi2d function with the extracted parameters
    computeVoronoi2d(
        boundary=boundary,
        N_points=N_points,
        points_seed=points_seed,
        margin=margin,
        N_iter=N_iter,
        use_decay=use_decay,
        edgeLength_threshold=edgeLength_threshold,
        cohesiveThickness=cohesiveThickness,
        triangularMesh_minLength=triangularMesh_minLength,
        visualize=visualize,
        saveData=saveData
    )


# Function to print the docstring of any object in the project. This function will help the user to see a detailed description of the object in a more structured way.
def print_docstring(obj):
    """
    Print the docstring of a given object with enhanced formatting.

    Parameters:
    - obj: The object whose docstring is to be printed.
    """
    docstring = obj.__doc__
    if not docstring:
        print("No docstring available for this object.")
        return
    
    # Split the docstring into lines
    lines = docstring.strip().split('\n')
    
    # Extract the summary, parameters, returns, example usage, and other sections
    summary = []
    params = []
    returns = []
    example_usage = []
    others = []
    current_section = summary

    # param_pattern = re.compile(r'\s*- \w+:')

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("Parameters:"):
            current_section = params
            continue
        if stripped_line.startswith("Returns:"):
            current_section = returns
            continue
        if stripped_line.startswith("Example usage:"):
            current_section = example_usage
            continue
        current_section.append(line)
    
    # Define a function to format a section
    def format_section(section, title):
        if section:
            return '\n'.join([
                "-" * 50,
                title,
                "-" * 50,
                '\n'.join(section),
                ""
            ])
        return ""

    # Print the formatted docstring
    formatted_docstring = "\n".join([
        "=" * 50,
        "DOCSTRING",
        "=" * 50,
        '\n'.join(summary),
        format_section(params, "PARAMETERS"),
        format_section(returns, "RETURNS"),
        format_section(example_usage, "EXAMPLE USAGE"),
        format_section(others, "OTHER SECTIONS")
    ])
    
    print(formatted_docstring)

# Function to analyze the statistics of the Voronoi Cells
def analyze_voronoi_cells(voronoi_cells):
    """
    Analyze the Voronoi cells to calculate useful statistics such as average grain size,
    percentage of different grain sizes, and the lengths of the shortest edges.

    Parameters:
    - voronoi_cells: List of Shapely Polygon or MultiPolygon objects representing the Voronoi cells.

    Returns:
    - A dictionary containing various statistics about the Voronoi cells.
    """
    edge_lengths = []
    cell_areas = []

    for cell in voronoi_cells:
        if isinstance(cell, Polygon):
            exterior_coords = list(cell.exterior.coords)
            cell_areas.append(cell.area)
            for i in range(len(exterior_coords) - 1):
                line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                edge_lengths.append(line.length)
        elif isinstance(cell, MultiPolygon):
            for poly in cell.geoms:
                exterior_coords = list(poly.exterior.coords)
                cell_areas.append(poly.area)
                for i in range(len(exterior_coords) - 1):
                    line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                    edge_lengths.append(line.length)

    edge_lengths = np.array(edge_lengths)
    cell_areas = np.array(cell_areas)

    stats = {
        "average_edge_length": np.mean(edge_lengths),
        "min_edge_length": np.min(edge_lengths),
        "max_edge_length": np.max(edge_lengths),
        "average_cell_area": np.mean(cell_areas),
        "min_cell_area": np.min(cell_areas),
        "max_cell_area": np.max(cell_areas),
        "total_cells": len(cell_areas),
        "total_edges": len(edge_lengths)
    }

    grain_size_bins = np.histogram(cell_areas, bins='auto')
    stats["grain_size_distribution"] = grain_size_bins

    return stats

# Function to create the report of the statistics
def get_voronoi_stats(voronoi_cells, N=None, threshold=None, file_name='Report_OriginalVoronoi.txt'):
    """
    Print useful statistics about the Voronoi cells, including the lengths of the shortest edges.
    Save the print statements in the Report.txt file inside the 'Report' directory.

    Parameters:
    - voronoi_cells: List of Shapely Polygon or MultiPolygon objects representing the Voronoi cells.
    - N: Optional, the number of shortest edges to print. If None, print all edges.
    - threshold: Optional, the length threshold to print edges shorter than this value. If None, print edges based on N.

    Prints the lengths of the edges in the format:
    Index: <index>, Length: <length>
    
    Example usage:
        - print_voronoi_statistics(voronoi_cells, N=10)
        - print_voronoi_statistics(voronoi_cells, threshold=0.1)
    """
    if (N is None and threshold is None) or (N is not None and threshold is not None):
        raise ValueError("Specify exactly one of 'N' or 'threshold'.")

    stats = analyze_voronoi_cells(voronoi_cells)

    # Ensure the 'Report' directory exists
    if not os.path.exists('Reports'):
        os.makedirs('Reports')

    # Write to file inside 'Report' directory
    file_path = os.path.join('Reports', file_name)
    
    with open(file_path, 'w') as file:
        # Print general statistics
        file.write("Voronoi Cell Statistics:\n")
        file.write(f"Average Edge Length: {stats['average_edge_length']:.5f}\n")
        file.write(f"Minimum Edge Length: {stats['min_edge_length']:.5f}\n")
        file.write(f"Maximum Edge Length: {stats['max_edge_length']:.5f}\n")
        file.write(f"Average Cell Area: {stats['average_cell_area']:.5f}\n")
        file.write(f"Minimum Cell Area: {stats['min_cell_area']:.5f}\n")
        file.write(f"Maximum Cell Area: {stats['max_cell_area']:.5f}\n")
        file.write(f"Total Number of Cells: {stats['total_cells']}\n")
        file.write(f"Total Number of Edges: {stats['total_edges']}\n")

        # Print grain size distribution
        file.write("\nGrain Size Distribution:\n")
        for i in range(len(stats["grain_size_distribution"][0])):
            file.write(f"Grain size range {stats['grain_size_distribution'][1][i]:.5f} to {stats['grain_size_distribution'][1][i + 1]:.5f}: {stats['grain_size_distribution'][0][i]} cells\n")

        # Collect all edges with their lengths for the shortest edges analysis
        edges = []
        for cell in voronoi_cells:
            if isinstance(cell, Polygon):
                exterior_coords = list(cell.exterior.coords)
                for i in range(len(exterior_coords) - 1):
                    line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                    edges.append((line.length, line))
            elif isinstance(cell, MultiPolygon):
                for poly in cell.geoms:
                    exterior_coords = list(poly.exterior.coords)
                    for i in range(len(exterior_coords) - 1):
                        line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                        edges.append((line.length, line))

        # Sort edges by length
        edges.sort(key=lambda x: x[0])

        # Determine the edges to print based on N or threshold
        if N is not None:
            edges_to_print = edges[:N]
        else:
            edges_to_print = [edge for edge in edges if edge[0] < threshold]

        if not edges_to_print:
            # If there are no edges to print, print a message
            file.write(f'\nThere are no edges shorter than the threshold of {threshold}.\n')
            return

        # Print the shortest edges
        file.write('\n10 Shortest Edge Lengths:\n')
        for idx, (length, _) in enumerate(edges_to_print):
            file.write(f"Index: {idx}, Length: {length:.5f}\n")
