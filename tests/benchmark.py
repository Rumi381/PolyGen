import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
from typing import List, Union, Tuple

from polygen.polygen2d import Geometry, IO
from polygen.polygen2d import generate_poisson_points, generate_sequence_points
from polygen.polygen2d import lloyd_with_density as lloyd
from polygen.polygen2d import VoronoiGenerator, MeshOptimizer, CohesiveZoneAdjuster
from shapely.geometry import Polygon
from shapely.affinity import scale

def scale_domain_to_reference(domain, reference_domain=None, target_area=1000.0):
    """
    Scale a domain to match a reference domain's area or to a target area.
    
    Parameters
    ----------
    domain : Polygon
        Domain to be scaled
    reference_domain : Polygon, optional
        Reference domain for scaling
    target_area : float, optional
        Target area for scaling if no reference domain is provided
    
    Returns
    -------
    Polygon
        Scaled domain
    """
    # Calculate the target area
    if reference_domain is not None:
        target = reference_domain.area
    else:
        target = target_area
    
    # Calculate scale factor based on area
    scale_factor = (target / domain.area) ** 0.5
    
    # Apply scaling transformation
    domain_centroid = domain.centroid
    scaled_domain = scale(domain, 
                          xfact=scale_factor, 
                          yfact=scale_factor, 
                          origin=domain_centroid)
    
    return scaled_domain

def normalize_domains(domains_list, domain_names, use_reference=False, target_area=1000.0):
    """
    Normalize all domains either to a reference domain or target area.
    
    Parameters
    ----------
    domains_list : list
        List of domain polygons
    domain_names : list
        List of domain names
    use_reference : bool
        Whether to use the largest domain as reference 
    target_area : float
        Target area for all domains if use_reference is False
        
    Returns
    -------
    list
        List of normalized domain polygons
    """
    normalized_domains = []
    
    if use_reference and len(domains_list) > 1:
        # Find the largest domain to use as reference
        domain_areas = [domain.area for domain in domains_list]
        reference_idx = domain_areas.index(max(domain_areas))
        reference_domain = domains_list[reference_idx]
        reference_name = domain_names[reference_idx]
        
        print(f"\nScaling domains to match reference domain '{reference_name}' with area {reference_domain.area:.2f}")
        
        # Scale all domains to match the reference
        for i, domain in enumerate(domains_list):
            if i == reference_idx:
                normalized_domains.append(domain)  # Keep reference domain as is
                print(f"  - Domain '{domain_names[i]}' is the reference (area: {domain.area:.2f})")
            else:
                original_area = domain.area
                scaled_domain = scale_domain_to_reference(domain, reference_domain=reference_domain)
                normalized_domains.append(scaled_domain)
                print(f"  - Scaled domain '{domain_names[i]}' from area {original_area:.2f} to {scaled_domain.area:.2f}")
    else:
        # Scale all domains to target area
        print(f"\nScaling all domains to target area of {target_area:.2f} square units")
        for i, domain in enumerate(domains_list):
            original_area = domain.area
            scaled_domain = scale_domain_to_reference(domain, target_area=target_area)
            normalized_domains.append(scaled_domain)
            print(f"  - Scaled domain '{domain_names[i]}' from area {original_area:.2f} to {scaled_domain.area:.2f}")
    
    return normalized_domains

def measure_memory_usage():
    """Measure current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def benchmark_pipeline(domain, domain_name, n_points, seed=7845, output_dir="benchmark_results"):
    """
    Benchmark each stage of the PolyGen pipeline for a given domain and point count.
    
    Parameters
    ----------
    domain : Polygon
        The domain to benchmark on
    domain_name : str
        Name of the domain for reporting
    n_points : int
        Number of points to generate
    seed : int
        Random seed for reproducibility
    output_dir : str
        Directory to save benchmark results
    
    Returns
    -------
    dict
        Dictionary containing timing measurements for each stage
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "n_points": n_points,
        "domain_name": domain_name,
        "domain_area": domain.area,
        "domain_complexity": len(list(domain.exterior.coords)),
        "concavity": calculate_concavity(domain),
        "pds_time": 0,
        "pds_memory": 0,
        "lloyd_time": 0,
        "lloyd_memory": 0,
        "lloyd_iterations": 0,
        "voronoi_time": 0,
        "voronoi_memory": 0,
        "cohesive_time": 0, 
        "cohesive_memory": 0,
        "edge_collapse_time": 0,
        "edge_collapse_memory": 0,
        "total_time": 0,
        "peak_memory": 0
    }
    
    # Initialize components
    voronoi_generator = VoronoiGenerator(buffer_factor=1.0)
    optimizer = MeshOptimizer(verbose=False)
    cohesive_adjuster = CohesiveZoneAdjuster(tolerance=0.005, max_iterations=10, verbose=False)
    
    # Global time tracking
    start_time_total = time.time()
    initial_memory = measure_memory_usage()
    peak_memory = initial_memory
    
    # 1. Measure Poisson disk sampling
    print(f"Generating {n_points} points with PDS...")
    start_time = time.time()
    # poissonPoints = generate_poisson_points(
    #     domain=domain, 
    #     N=n_points, 
    #     seed=seed, 
    #     margin=0.05, 
    #     tolerance=1e-6,
    #     max_iterations=100
    # )
    
    sequencePoints = generate_sequence_points(
        domain=domain, 
        N=n_points, 
        seed=seed, 
        margin=0.05, 
        use_sobol=False
    )
    results["pds_time"] = time.time() - start_time
    current_memory = measure_memory_usage()
    results["pds_memory"] = current_memory - initial_memory
    peak_memory = max(peak_memory, current_memory)
    
    # 2. Measure Lloyd relaxation
    print("Running Lloyd relaxation...")
    start_time = time.time()
    relaxedPoints, metrics = lloyd(
        polygon=domain, 
        seed_points=sequencePoints, 
        density_function=None, 
        max_iterations=10000000, 
        tol=5e-3, 
        use_decay=True, 
        grad_increase_tol=10000
    )
    results["lloyd_time"] = time.time() - start_time
    current_memory = measure_memory_usage()
    results["lloyd_memory"] = current_memory - initial_memory
    peak_memory = max(peak_memory, current_memory)
    results["lloyd_iterations"] = metrics.get("iterations", 0)
    
    # 3. Measure Voronoi cell generation
    print("Generating Voronoi cells...")
    start_time = time.time()
    voronoi_cells = voronoi_generator.generate_cells(domain=domain, points=relaxedPoints)
    results["voronoi_time"] = time.time() - start_time
    current_memory = measure_memory_usage()
    results["voronoi_memory"] = current_memory - initial_memory
    peak_memory = max(peak_memory, current_memory)
    
    # 4. Measure cohesive zone adjustment
    print("Applying cohesive zone adjustment...")
    start_time = time.time()
    adjusted_cells = cohesive_adjuster.adjust_target_ratio(
        cells=voronoi_cells, 
        target_ratio=0.9
    )
    results["cohesive_time"] = time.time() - start_time
    current_memory = measure_memory_usage()
    results["cohesive_memory"] = current_memory - initial_memory
    peak_memory = max(peak_memory, current_memory)
    
    # 5. Measure edge collapse optimization
    print("Performing edge collapse optimization...")
    start_time = time.time()
    optimized_cells = optimizer.optimize_voronoi_cells(
        voronoi_cells=voronoi_cells, 
        threshold=0.05 * (domain.area / n_points) ** 0.5  # Adaptive threshold
    )
    results["edge_collapse_time"] = time.time() - start_time
    current_memory = measure_memory_usage()
    results["edge_collapse_memory"] = current_memory - initial_memory
    peak_memory = max(peak_memory, current_memory)
    
    # Record final metrics
    results["total_time"] = time.time() - start_time_total
    results["peak_memory"] = peak_memory - initial_memory
    
    # Save individual result
    result_df = pd.DataFrame([results])
    result_df.to_csv(f"{output_dir}/benchmark_{n_points}_{results['domain_name']}.csv", index=False)
    
    return results

def calculate_concavity(polygon):
    """
    Calculate a concavity metric for the polygon.
    
    Returns a ratio between the actual area and the area of the convex hull.
    Values closer to 1 indicate more convex shapes.
    """
    convex_hull = polygon.convex_hull
    return polygon.area / convex_hull.area

def run_comprehensive_benchmark(domains, point_counts, seeds=None, output_dir="benchmark_results", scale_domains=True, use_reference=False, target_area=1000.0):
    """
    Run comprehensive benchmarks across multiple domains and point counts.
    
    Parameters
    ----------
    domains : list
        List of domains to benchmark. Each item can be either:
        - Path to a .obj file
        - A geometry object from Geometry module
        - A tuple of (geometry_func, params_dict) to be evaluated
    point_counts : list
        List of point counts to benchmark
    seeds : list, optional
        List of random seeds for reproducibility, default is [7845]
    output_dir : str
        Directory to save benchmark results
    scale_domains : bool
        Whether to scale domains to a common reference
    use_reference : bool
        If True, use largest domain as reference; if False, use target_area
    target_area : float
        Target area for all domains if use_reference is False
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing all benchmark results
    """
    if seeds is None:
        seeds = [7845]
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    processed_domains = []
    domain_names = []
    
    # Process domain inputs
    for domain_item in domains:
        if isinstance(domain_item, str):
            # It's a file path
            try:
                domain = IO.load_polygon_from_file(domain_item)
                # Set the domain name based on the file name
                domain_name = os.path.basename(domain_item).split('.')[0]
                processed_domains.append(domain)
                domain_names.append(domain_name)
                print(f"Successfully loaded domain: {domain_name} (area: {domain.area:.2f})")
            except Exception as e:
                print(f"Error loading domain from file {domain_item}: {str(e)}")
        elif isinstance(domain_item, tuple) and callable(domain_item[0]):
            # It's a function to call with parameters
            geometry_func, params = domain_item
            try:
                geometry = geometry_func(**params)
                domain = IO.load_polygon_from_file(geometry)
                domain_name = params.get('name', f"{geometry_func.__name__}")
                processed_domains.append(domain)
                domain_names.append(domain_name)
                print(f"Successfully created domain: {domain_name} (area: {domain.area:.2f})")
            except Exception as e:
                print(f"Error creating domain from function {geometry_func.__name__}: {str(e)}")
        elif isinstance(domain_item, Polygon) or (hasattr(domain_item, '__class__') and 'Polygon' in domain_item.__class__.__name__):
            # It's already a domain polygon
            domain_name = getattr(domain_item, 'name', f"domain_{hash(str(domain_item))}"[:8])
            processed_domains.append(domain_item)
            domain_names.append(domain_name)
            print(f"Added domain object: {domain_name} (area: {domain_item.area:.2f})")
        else:
            print(f"Skipping unsupported domain type: {type(domain_item)}")
    
    # Scale domains if requested
    if scale_domains and processed_domains:
        processed_domains = normalize_domains(
            processed_domains, 
            domain_names, 
            use_reference=use_reference,
            target_area=target_area
        )
    
    # Run benchmarks on all processed domains
    for i, (domain, domain_name) in enumerate(zip(processed_domains, domain_names)):
        print(f"\nBenchmarking domain {i+1}/{len(processed_domains)}: {domain_name} (area: {domain.area:.2f})")
        
        for n_points in point_counts:
            for seed in seeds:
                print(f"\nRunning benchmark with {n_points} points, seed {seed}")
                try:
                    result = benchmark_pipeline(
                        domain=domain, 
                        domain_name=domain_name,
                        n_points=n_points, 
                        seed=seed,
                        output_dir=output_dir
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"Error benchmarking {domain_name} with {n_points} points: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    # Combine all results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{output_dir}/all_benchmarks.csv", index=False)
        
        # Generate summary tables
        generate_summary_tables(results_df, output_dir)
        
        # Generate plots
        generate_benchmark_plots(results_df, output_dir)
        
        return results_df
    else:
        print("No valid results were generated.")
        return pd.DataFrame()

def generate_summary_tables(results_df, output_dir):
    """
    Generate summary tables from benchmark results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing all benchmark results
    output_dir : str
        Directory to save summary tables
    """
    # Table A: Pipeline stage costs for highest point count
    max_points = results_df["n_points"].max()
    table_a = results_df[results_df["n_points"] == max_points].agg({
        "pds_time": "mean",
        "lloyd_time": "mean",
        "lloyd_iterations": "mean",
        "cohesive_time": "mean",
        "edge_collapse_time": "mean",
        "total_time": "mean",
        "peak_memory": "mean"
    }).reset_index()
    
    table_a.columns = ["Stage", "Value"]
    table_a["Stage"] = ["PDS Time (s)", "Lloyd Time (s)", "Lloyd Iterations", 
                      "Cohesive Zone Time (s)", "Edge Collapse Time (s)", 
                      "Total Time (s)", "Peak Memory (MB)"]
    
    table_a.to_csv(f"{output_dir}/table_a_pipeline_costs.csv", index=False)
    
    # Print Table A to console
    print("\nTable A: Pipeline stage costs")
    print("==============================")
    for _, row in table_a.iterrows():
        print(f"{row['Stage']}: {row['Value']:.2f}")
    
    # Table B: Scaling with grain count
    scaling_summary = results_df.groupby("n_points").agg({
        "total_time": ["mean", "std"],
        "peak_memory": ["mean", "std"],
        "lloyd_iterations": ["mean", "std"]
    }).reset_index()
    
    scaling_summary.columns = [' '.join(col).strip() for col in scaling_summary.columns.values]
    scaling_summary.to_csv(f"{output_dir}/table_b_scaling.csv", index=False)
    
    # Print Table B to console
    print("\nTable B: Scaling with grain count")
    print("================================")
    print(f"{'n_points':<10}{'Time (s)':<15}{'Memory (MB)':<15}{'Lloyd Iter.':<10}")
    for _, row in scaling_summary.iterrows():
        print(f"{row['n_points']:<10}{row['total_time mean']:.2f} ± {row['total_time std']:.2f}   {row['peak_memory mean']:.1f} ± {row['peak_memory std']:.1f}      {row['lloyd_iterations mean']:.1f}")
    
    # Table C: Impact of domain complexity
    complexity_summary = results_df.groupby(["domain_name", "concavity"]).agg({
        "total_time": ["mean", "std"],
        "voronoi_time": ["mean", "std"],
        "lloyd_time": ["mean", "std"]
    }).reset_index()
    
    complexity_summary.columns = [' '.join(col).strip() for col in complexity_summary.columns.values]
    complexity_summary.to_csv(f"{output_dir}/table_c_complexity_impact.csv", index=False)
    
    # Print domain complexity impact
    print("\nTable C: Domain complexity impact")
    print("================================")
    print(f"{'Domain':<20}{'Concavity':<10}{'Total (s)':<15}{'Voronoi (s)':<15}{'Lloyd (s)':<15}")
    for _, row in complexity_summary.iterrows():
        print(f"{row['domain_name']:<20}{row['concavity']:.3f}       {row['total_time mean']:.2f} ± {row['total_time std']:.2f}   {row['voronoi_time mean']:.2f} ± {row['voronoi_time std']:.2f}   {row['lloyd_time mean']:.2f} ± {row['lloyd_time std']:.2f}")

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
    # ratio = aspect_ratios.get(layout, aspect_ratios['single'])
    # width, height = ratio

    # # Scale width and height to fit the journal's column width
    # if layout == 'single':
    #     width = single_column
    # else:
    #     width = double_column
    # height = (width / ratio[0]) * ratio[1]

    # # Ensure height doesn't exceed maximum allowed
    # height = min(height, max_height)
    # return (width, height)
    ratio_w, ratio_h = aspect_ratios.get(layout, aspect_ratios['single'])

    # absolute width forced by journal spec
    abs_width = single_column if layout == 'single' else double_column

    # scale height to maintain aspect ratio
    abs_height = abs_width * (ratio_h / ratio_w)
    abs_height = min(abs_height, max_height)

    return (abs_width, abs_height)
 
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

def create_improved_domain_complexity_plot(results_df, output_dir):
    """
    Create an improved domain complexity plot that shows computational time vs grain count
    for different domain geometries, with concavity values indicated in the legend.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing all benchmark results
    output_dir : str
        Directory to save the plot
    """

    # Apply publication styling
    set_publication_style()

    # Create figure
    plt.figure(figsize=get_figure_size(layout='single', journal_type='large'))
    
    # Ensure we're grouping properly by converting domain name to string
    results_df["domain_name_str"] = results_df["domain_name"].astype(str)
    
    # Define domain colors
    domain_colors = {
        "CalciteBoundary": "#1f77b4",     # blue
        "WrenchGeometry": "#ff7f0e",      # orange
        "PlateWithHole": "#2ca02c"        # green
    }
    
    # Get unique domain names and grain counts
    domains = results_df["domain_name_str"].unique()
    grain_counts = sorted(results_df["n_points"].unique())
    
    # Plot data points for each domain
    for domain_name in domains:
        domain_data = results_df[results_df["domain_name_str"] == domain_name]
        color = domain_colors.get(domain_name, "#1f77b4")  # Default to blue if not found
        
        # Get domain concavity (should be the same for all grain counts)
        concavity = domain_data["concavity"].iloc[0]
        concavity_formatted = f"{concavity:.2f}"  # Format to 2 decimal places
        
        # Group by grain count and calculate mean total time
        time_data = domain_data.groupby("n_points")["total_time"].mean().reset_index()
        
        # Plot time vs grain count for this domain
        plt.plot(
            time_data["n_points"], 
            time_data["total_time"],
            'o-',
            color=color,
            linewidth=1.5,
            markersize=5,
            markeredgecolor='k',
            markeredgewidth=0.5,
            label=f"{domain_name} (concavity: {concavity_formatted})",
            zorder=3
        )
    
    # Set x-axis to log scale
    plt.xscale('log')
    
    # Add labels and grid
    plt.xlabel('Number of grains ($n$)')
    plt.ylabel('Total computation time (s)')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
    
    # Add legend in top-left corner
    plt.legend(fontsize=8, loc='upper left', framealpha=0.9, 
               edgecolor='gray', facecolor='white')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/domain_complexity_impact.eps", dpi=2400, bbox_inches='tight')
    plt.savefig(f"{output_dir}/domain_complexity_impact.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_benchmark_plots(results_df, output_dir):
    """
    Generate publication-quality plots visualizing benchmark results as separate figures.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing all benchmark results
    output_dir : str
        Directory to save the plots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply publication styling
    set_publication_style()
    
    # Define a publication-friendly color palette
    colors = {
        'pds': '#1f77b4',      # blue
        'lloyd': '#ff7f0e',    # orange
        'voronoi': '#2ca02c',  # green
        'edge_collapse': '#d62728',  # red
        'cohesive': '#9467bd'  # purple
    }
    
    # ======================================================================
    # Figure 1: Scaling with grain count (log-log plot)
    # ======================================================================
    # Create figure
    plt.figure(figsize=get_figure_size(layout='single', journal_type='large'))
    
    scaling_data = results_df.groupby("n_points")["total_time"].mean().reset_index()
    plt.loglog(scaling_data["n_points"], scaling_data["total_time"], 'o-', 
              linewidth=1.5, color=colors['lloyd'], markersize=4)
    
    # Add reference line for O(n log n)
    x_range = np.array(scaling_data["n_points"])
    y_ref = x_range * np.log(x_range) / (x_range[0] * np.log(x_range[0])) * scaling_data["total_time"].iloc[0]
    plt.loglog(x_range, y_ref, 'k--', alpha=0.7, linewidth=1, label='$O(n log n)$')
    
    plt.xlabel('Number of grains ($n$)')
    plt.ylabel('Total time (s)')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
    plt.legend(fontsize=8, frameon=False, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/computational_scaling.eps", dpi=2400, bbox_inches='tight')
    plt.savefig(f"{output_dir}/computational_scaling.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # Figure 2: Pipeline stage breakdown as stacked area
    # ======================================================================
    # Create figure
    plt.figure(figsize=get_figure_size(layout='single', journal_type='large'))
    
    # Get all point counts in ascending order
    point_counts = sorted(results_df["n_points"].unique())
    
    # Calculate mean times for each pipeline stage for each point count
    stage_times = {}
    stages = ['pds_time', 'lloyd_time', 'voronoi_time', 'edge_collapse_time', 'cohesive_time']
    stage_labels = ['Initial seeding (Halton)', 'CVT (Lloyd)', 'Clipped Voronoi generation', 
                   'Optimization (short edge collapse)', 'Cohesive zone generation']
    
    for stage in stages:
        stage_times[stage] = []
        for n in point_counts:
            subset = results_df[results_df["n_points"] == n]
            stage_times[stage].append(subset[stage].mean())
    
    # Create stacked area plot
    previous = np.zeros(len(point_counts))
    for i, stage in enumerate(stages):
        plt.fill_between(
            point_counts, 
            previous, 
            previous + np.array(stage_times[stage]),
            label=stage_labels[i],
            color=list(colors.values())[i],
            alpha=0.8
        )
        previous += np.array(stage_times[stage])
    
    plt.xlabel('Number of grains ($n$)')
    plt.ylabel('Time (s)')
    plt.xscale('log')
    # plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.3, axis='y')
    
    # Create a separate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], fontsize=8, loc='upper left', 
              frameon=False, bbox_to_anchor=(0.05, 0.95))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pipeline_stage_distribution.eps", dpi=2400, bbox_inches='tight')
    plt.savefig(f"{output_dir}/pipeline_stage_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # Figure 3: Domain complexity impact
    # ======================================================================
    create_improved_domain_complexity_plot(results_df, output_dir)
    
    # ======================================================================
    # Figure 4: Lloyd iterations vs grain count
    # ======================================================================
    # Create figure
    plt.figure(figsize=get_figure_size(layout='single', journal_type='large'))

    lloyd_data = results_df.groupby("n_points")["lloyd_iterations"].mean().reset_index()
    
    plt.plot(lloyd_data["n_points"], lloyd_data["lloyd_iterations"], 'o-', 
           linewidth=1.5, color=colors['lloyd'], markersize=4)
    
    plt.xlabel('Number of grains ($n$)')
    plt.ylabel('Average Lloyd iterations')
    plt.xscale('log')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lloyd_convergence.eps", dpi=2400, bbox_inches='tight')
    plt.savefig(f"{output_dir}/lloyd_convergence.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # Additional plot: Memory usage scaling
    # ======================================================================
    # Create figure
    plt.figure(figsize=get_figure_size(layout='single', journal_type='large'))

    memory_data = results_df.groupby("n_points")["peak_memory"].mean().reset_index()
    
    plt.loglog(memory_data["n_points"], memory_data["peak_memory"], 'o-', 
              linewidth=1.5, color=colors['pds'], markersize=4)
    
    # Add reference line for O(n)
    x_range = np.array(memory_data["n_points"])
    y_ref = x_range / x_range[0] * memory_data["peak_memory"].iloc[0]
    plt.loglog(x_range, y_ref, 'k--', alpha=0.7, linewidth=1, label='$O(n)$')
    
    plt.xlabel('Number of grains ($n$)')
    plt.ylabel('Peak memory usage (MB)')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
    plt.legend(fontsize=8, frameon=False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/memory_scaling.eps", dpi=2400, bbox_inches='tight')
    plt.savefig(f"{output_dir}/memory_scaling.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Define domains to test (mixed format)
    domains = [
        # File paths
        "./examples/CalciteBoundary.obj",  # Complex, concave
        "./examples/WrenchGeometry.obj",  # Complex, concave
        
        # Geometry function with parameters
        (Geometry.rectangle_with_circular_hole, {
            'rect_point1': (0, 0),
            'rect_point2': (10, 6),
            'circle_center': (5, 3),
            'circle_radius': 1,
            'name': 'PlateWithHole'
        }),
    ]
    
    # Define grain counts to test logarithmic scaling
    point_counts = [100, 500, 1000, 5000, 10000]  #, 1000, 5000, 10000
    
    # Run a single seed for faster testing
    seeds = [7845]
    
    # Run the comprehensive benchmark with domain scaling
    results = run_comprehensive_benchmark(
        domains=domains,
        point_counts=point_counts,
        seeds=seeds,
        output_dir="polygen_benchmark_results",
        scale_domains=True,  # Enable domain scaling
        use_reference=False, # Use target area instead of reference domain
        target_area=1.5e6   # Target area of 1000 square units
    )
    
    print("\nBenchmark complete!")
    print(f"Results saved to 'polygen_benchmark_results' directory")
    
    # Print summary for quick reference
    if not results.empty:
        print("\nSummary of results:")
        for n in sorted(results["n_points"].unique()):
            subset = results[results["n_points"] == n]
            print(f"{n} grains: {subset['total_time'].mean():.2f}s ± {subset['total_time'].std():.2f}s")
        
        # Calculate empirical scaling factor
        if len(point_counts) >= 2:
            n1, n2 = min(point_counts), max(point_counts)
            t1 = results[results["n_points"] == n1]["total_time"].mean()
            t2 = results[results["n_points"] == n2]["total_time"].mean()
            
            log_ratio = np.log(t2/t1) / np.log(n2/n1)
            print(f"\nEmpirical scaling factor: O(n^{log_ratio:.3f})")
            
            expected = np.log(n2 * np.log(n2) / (n1 * np.log(n1))) / np.log(n2/n1)
            print(f"Expected for O(n log n): {expected:.3f}")