import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
from typing import List, Union, Tuple, Dict, Any

from polygen.polygen2d import Geometry, IO
from polygen.polygen2d import generate_poisson_points
from polygen.polygen2d import lloyd_with_density as lloyd
from polygen.polygen2d import VoronoiGenerator, CohesiveZoneAdjuster
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

class TestCohesiveZoneAdjuster(CohesiveZoneAdjuster):
    """Extended CohesiveZoneAdjuster that captures detailed metrics."""
    
    def adjust_target_ratio_with_metrics(
        self,
        cells,
        target_ratio,
        initial_thickness=0.1,
        preserve_boundary=True
    ):
        """Adjustment method that returns detailed metrics about the convergence."""
        self.history.clear()
        current_thickness = initial_thickness
        metrics = {
            "iterations": 0,
            "final_thickness": 0,
            "achieved_ratio": 0,
            "convergence_status": "Not started",
            "history": []
        }
        
        for iteration in range(self.max_iterations):
            # Adjust cells with current thickness
            adjusted_cells = self.adjust_fixed_thickness(
                cells, current_thickness, preserve_boundary
            )
            
            # Calculate achieved ratio
            achieved_ratio = self._calculate_area_ratio(cells, adjusted_cells)
            
            # Update history
            self.history.append((current_thickness, achieved_ratio))
            metrics["history"].append({
                "iteration": iteration + 1,
                "thickness": current_thickness,
                "achieved_ratio": achieved_ratio,
                "error": abs(achieved_ratio - target_ratio)
            })
            
            # Check convergence
            if abs(achieved_ratio - target_ratio) <= self.tolerance:
                metrics["iterations"] = iteration + 1
                metrics["final_thickness"] = current_thickness
                metrics["achieved_ratio"] = achieved_ratio
                metrics["convergence_status"] = "Converged to tolerance"
                return adjusted_cells, metrics
                
            # Update thickness through interpolation
            current_thickness = self._interpolate_thickness(target_ratio)
        
        # Maximum iterations reached
        metrics["iterations"] = self.max_iterations
        metrics["final_thickness"] = current_thickness
        metrics["achieved_ratio"] = achieved_ratio
        metrics["convergence_status"] = "Maximum iterations reached"
        
        return adjusted_cells, metrics

def run_sensitivity_analysis(
    domains,
    target_ratios=[0.3, 0.5, 0.7, 0.9],
    initial_thicknesses=[0.05, 0.1, 0.2, 0.3],
    tolerances=[5e-3, 1e-3, 5e-4],
    cell_counts=[100, 500, 1000, 5000, 10000],
    num_seeds=3,
    output_dir="sensitivity_results",
    scale_domains=True,
    use_reference=False,
    target_area=1000.0
):
    """
    Run a comprehensive sensitivity analysis for the cohesive zone generation algorithm.
    
    Parameters
    ----------
    domains : list
        List of domains to analyze. Each item can be either:
        - Path to a .obj file
        - A geometry object from Geometry module
        - A tuple of (geometry_func, params_dict) to be evaluated
    target_ratios : list
        Target area ratios to test
    initial_thicknesses : list
        Initial thickness values to test
    tolerances : list
        Convergence tolerance values to test
    cell_counts : list
        Number of cells to generate for each test
    num_seeds : int
        Number of random seeds to use for each configuration
    output_dir : str
        Directory to save results
    scale_domains : bool
        Whether to scale domains to a common reference
    use_reference : bool
        If True, use largest domain as reference; if False, use target_area
    target_area : float
        Target area for all domains if use_reference is False
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing all results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results list
    results = []
    
    # Process domain inputs
    processed_domains = []
    domain_names = []
    
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
    
    # Initialize components
    voronoi_generator = VoronoiGenerator(buffer_factor=1.0)
    
    # Track convergence trajectory data separately for detailed analysis
    trajectory_data = []
    
    # Run the analysis for each domain
    for i, (domain, domain_name) in enumerate(zip(processed_domains, domain_names)):
        print(f"\nProcessing domain {i+1}/{len(processed_domains)}: {domain_name} (area: {domain.area:.2f})")
        
        for n_cells in cell_counts:
            print(f"  Cell count: {n_cells}")
            
            for seed in range(1, num_seeds + 1):
                # Generate points and Voronoi cells
                try:
                    points = generate_poisson_points(
                        domain=domain, 
                        N=n_cells, 
                        seed=seed, 
                        margin=0.05, 
                        tolerance=1e-6,
                        max_iterations=10000
                    )
                    
                    # Relax points with Lloyd's algorithm
                    relaxed_points, _ = lloyd(
                        polygon=domain, 
                        seed_points=points, 
                        density_function=None, 
                        max_iterations=10000000,
                        tol=1e-5, 
                        use_decay=True, 
                        grad_increase_tol=1.4
                    )
                    
                    # Generate Voronoi cells
                    voronoi_cells = voronoi_generator.generate_cells(domain=domain, points=relaxed_points)
                    
                    # Run tests for each configuration
                    for target_ratio in target_ratios:
                        for initial_thickness in initial_thicknesses:
                            for tolerance in tolerances:
                                # Configure adjuster with logging
                                test_adjuster = TestCohesiveZoneAdjuster(
                                    tolerance=tolerance,
                                    max_iterations=10,
                                    verbose=False
                                )
                                
                                # Measure time
                                start_time = time.time()
                                
                                try:
                                    # Run adjustment
                                    _, metrics = test_adjuster.adjust_target_ratio_with_metrics(
                                        cells=voronoi_cells,
                                        target_ratio=target_ratio,
                                        initial_thickness=initial_thickness
                                    )
                                    
                                    runtime = time.time() - start_time
                                    
                                    # Extract the error history for convergence trajectory analysis
                                    iteration_errors = []
                                    for entry in metrics["history"]:
                                        iteration_errors.append(entry["error"])
                                    
                                    # Store trajectory data for later analysis
                                    trajectory_info = {
                                        "domain": domain_name,
                                        "n_cells": n_cells,
                                        "seed": seed,
                                        "target_ratio": target_ratio,
                                        "initial_thickness": initial_thickness,
                                        "tolerance": tolerance,
                                        "errors": iteration_errors
                                    }
                                    trajectory_data.append(trajectory_info)
                                    
                                    # Store all results
                                    result_data = {
                                        "domain": domain_name,
                                        "domain_area": domain.area,
                                        "n_cells": n_cells,
                                        "seed": seed,
                                        "target_ratio": target_ratio,
                                        "initial_thickness": initial_thickness,
                                        "tolerance": tolerance,
                                        "iterations": metrics["iterations"],
                                        "final_thickness": metrics["final_thickness"],
                                        "achieved_ratio": metrics["achieved_ratio"],
                                        "error": abs(metrics["achieved_ratio"] - target_ratio),
                                        "convergence_status": metrics["convergence_status"],
                                        "runtime": runtime
                                    }
                                    
                                    results.append(result_data)
                                    
                                    # Print progress
                                    if metrics["convergence_status"] == "Converged to tolerance":
                                        print(f"    r*={target_ratio}, t_init={initial_thickness}, tol={tolerance}: " 
                                              f"Converged in {metrics['iterations']} iterations")
                                    else:
                                        print(f"    r*={target_ratio}, t_init={initial_thickness}, tol={tolerance}: " 
                                              f"Failed to converge ({metrics['iterations']} iterations)")
                                        
                                except Exception as e:
                                    print(f"    Error in cohesive adjustment: {str(e)}")
                                    
                except Exception as e:
                    print(f"  Error processing cell count {n_cells}, seed {seed}: {str(e)}")
    
    # Convert to DataFrame
    if not results:
        print("No valid results were generated.")
        return pd.DataFrame()
        
    results_df = pd.DataFrame(results)
    
    # Save full results
    results_df.to_csv(f"{output_dir}/sensitivity_analysis_full.csv", index=False)
    
    # Save trajectory data
    trajectory_df = pd.DataFrame({
        "domain": [t["domain"] for t in trajectory_data],
        "n_cells": [t["n_cells"] for t in trajectory_data],
        "seed": [t["seed"] for t in trajectory_data],
        "target_ratio": [t["target_ratio"] for t in trajectory_data],
        "initial_thickness": [t["initial_thickness"] for t in trajectory_data],
        "tolerance": [t["tolerance"] for t in trajectory_data],
        "errors": [t["errors"] for t in trajectory_data]
    })
    trajectory_df.to_csv(f"{output_dir}/convergence_trajectories_data.csv", index=False)
    
    # Generate summary statistics
    summary = results_df.groupby(["target_ratio", "initial_thickness", "tolerance"]).agg({
        "iterations": ["mean", "std", "max"],
        "error": ["mean", "max"],
        "runtime": ["mean", "std"]
    })
    
    # Format summary for readability
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(f"{output_dir}/sensitivity_analysis_summary.csv", index=False)
    
    # Print summary
    print("\nSummary by configuration:")
    print("========================")
    print(f"{'Target Ratio':<12}{'Init Thick':<10}{'Tolerance':<10}{'Iterations':<15}{'Error':<10}{'Runtime (s)':<12}")
    for _, row in summary.iterrows():
        iter_info = f"{row['iterations_mean']:.1f} ± {row['iterations_std']:.1f}"
        print(f"{row['target_ratio']:<12}{row['initial_thickness']:<10}{row['tolerance']:<10}{iter_info:<15}{row['error_mean']:.4f}{row['runtime_mean']:.2f} ± {row['runtime_std']:.2f}")
    
    # Generate summary by target ratio
    ratio_summary = results_df.groupby(["target_ratio"]).agg({
        "iterations": ["mean", "std", "max"],
        "error": ["mean", "max"]
    })
    ratio_summary.columns = [f"{col[0]}_{col[1]}" for col in ratio_summary.columns]
    ratio_summary = ratio_summary.reset_index()
    
    print("\nSummary by target ratio:")
    print("=====================")
    for _, row in ratio_summary.iterrows():
        print(f"Target ratio {row['target_ratio']}: {row['iterations_mean']:.1f} ± {row['iterations_std']:.1f} iterations, max {row['iterations_max']}")
    
    # Generate plots
    generate_sensitivity_plots(results_df, trajectory_data, output_dir)
    
    return results_df, trajectory_data

def generate_sensitivity_plots(results_df, trajectory_data, output_dir):
    """Generate plots visualizing sensitivity analysis results."""
    # Set plot style
    plt.style.use('ggplot')
    
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Iterations vs Target Ratio with different initial thicknesses
    ax1 = axs[0, 0]
    for t_init in sorted(results_df["initial_thickness"].unique()):
        data = results_df[results_df["initial_thickness"] == t_init]
        means = data.groupby("target_ratio")["iterations"].mean()
        stds = data.groupby("target_ratio")["iterations"].std()
        ax1.errorbar(means.index, means.values, yerr=stds.values, marker='o', 
                  label=f"t_init = {t_init}", capsize=4)
    
    ax1.set_xlabel("Target Ratio")
    ax1.set_ylabel("Average Iterations")
    ax1.set_title("Convergence Speed vs Target Ratio")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Error vs Iterations - Scatter plot with logarithmic y-axis
    ax2 = axs[0, 1]
    # Group by iterations and calculate mean error
    error_by_iter = results_df.groupby("iterations")["error"].mean()
    ax2.semilogy(error_by_iter.index, error_by_iter.values, 'ko-', linewidth=2, label="Mean Error")
    
    # Add scatter points for individual results
    ax2.scatter(results_df["iterations"], results_df["error"], alpha=0.3, s=20, color='blue')
    
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Final Error |r - r*| (log scale)")
    ax2.set_title("Error vs. Iterations")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Iterations distribution histogram
    ax3 = axs[1, 0]
    bins = range(1, min(results_df["iterations"].max() + 2, 11))
    ax3.hist(results_df["iterations"], bins=bins, rwidth=0.8, alpha=0.7)
    ax3.set_xlabel("Iterations to Converge")
    ax3.set_ylabel("Frequency")
    ax3.set_title(f"Iteration Count Distribution (n={len(results_df)})")
    ax3.grid(True, alpha=0.3)
    
    # 4. Iterations vs Initial Thickness - Box plot
    ax4 = axs[1, 1]
    thickness_data = []
    tick_labels = []
    
    for thickness in sorted(results_df["initial_thickness"].unique()):
        subset = results_df[results_df["initial_thickness"] == thickness]["iterations"]
        thickness_data.append(subset.values)
        tick_labels.append(str(thickness))
    
    # Use tick_labels parameter to avoid deprecation warning
    # Try both versions to be compatible with different matplotlib versions
    try:
        # For matplotlib 3.9+
        ax4.boxplot(thickness_data, tick_labels=tick_labels)
    except TypeError:
        # For older matplotlib versions
        ax4.boxplot(thickness_data, labels=tick_labels)
    
    ax4.set_xlabel("Initial Thickness")
    ax4.set_ylabel("Iterations to Converge")
    ax4.set_title("Effect of Initial Thickness on Convergence")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_overview.png", dpi=300)
    plt.close()
    
    # Convergence trajectories plot using actual data
    plt.figure(figsize=(12, 8))
    
    # Group trajectories by target ratio and plot representative examples
    for target_ratio in sorted(set(t["target_ratio"] for t in trajectory_data)):
        # Find examples for this target ratio with initial_thickness=0.1
        target_examples = [t for t in trajectory_data 
                           if t["target_ratio"] == target_ratio 
                           and t["initial_thickness"] == 0.1]
        
        if target_examples:
            # Choose the first example
            example = target_examples[0]
            errors = example["errors"]
            
            if errors:
                iterations = list(range(1, len(errors) + 1))
                plt.semilogy(iterations, errors, 'o-', linewidth=2, 
                             label=f"r* = {target_ratio}")
    
    plt.xlabel("Iteration")
    plt.ylabel("Error |r - r*| (log scale)")
    plt.title("Convergence Trajectories for Different Target Ratios")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{output_dir}/convergence_trajectories.png", dpi=300)
    plt.close()
    
    # Create heatmap of iterations by target ratio and initial thickness
    plt.figure(figsize=(10, 8))
    heatmap_data = results_df.groupby(["target_ratio", "initial_thickness"])["iterations"].mean().unstack()
    
    if not heatmap_data.empty:
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Average Iterations')
        plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
        plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
        plt.xlabel("Initial Thickness")
        plt.ylabel("Target Ratio")
        plt.title("Average Iterations by Configuration")
        
        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                plt.text(j, i, f"{value:.1f}", ha="center", va="center", 
                        color="white" if value > heatmap_data.values.mean() else "black")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/iterations_heatmap.png", dpi=300)
    plt.close()
    
    # Calculate convergence rate (error reduction per iteration)
    # This demonstrates the quadratic convergence property
    plt.figure(figsize=(10, 8))
    
    # For each trajectory with at least 3 iterations
    rates = []
    iterations = []
    
    for trajectory in trajectory_data:
        errors = trajectory["errors"]
        if len(errors) >= 3:
            # Calculate error reduction rate at each step
            for i in range(1, len(errors) - 1):
                if errors[i] > 0:  # Avoid division by zero
                    # log(e_{i+1}/e_i)/log(e_i/e_{i-1}) ≈ 2 for quadratic convergence
                    rate = np.log(errors[i+1]/errors[i]) / np.log(errors[i]/errors[i-1])
                    rates.append(rate)
                    iterations.append(i)
    
    if rates:
        plt.scatter(iterations, rates, alpha=0.5)
        plt.axhline(y=2.0, color='r', linestyle='-', label="Quadratic convergence (rate = 2)")
        plt.axhline(y=1.0, color='g', linestyle='--', label="Linear convergence (rate = 1)")
        
        # Calculate average rate
        avg_rate = np.mean(rates)
        plt.axhline(y=avg_rate, color='b', linestyle=':', 
                    label=f"Average measured rate = {avg_rate:.2f}")
        
        plt.xlabel("Iteration")
        plt.ylabel("Convergence Rate")
        plt.title("Convergence Rate Analysis\n(Rate ≈ 2 indicates quadratic convergence)")
        plt.ylim(-0.5, 3.5)  # Limit to meaningful range
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/convergence_rate_analysis.png", dpi=300)
        plt.close()
    
    # Additional plot: Average iterations vs tolerance
    plt.figure(figsize=(10, 6))
    tolerance_data = results_df.groupby("tolerance")["iterations"].agg(["mean", "std"]).reset_index()
    plt.errorbar(tolerance_data["tolerance"], tolerance_data["mean"], 
                yerr=tolerance_data["std"], marker='o', capsize=5)
    plt.xscale('log')
    plt.xlabel("Tolerance")
    plt.ylabel("Average Iterations")
    plt.title("Effect of Tolerance on Convergence Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tolerance_effect.png", dpi=300)
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
    
    # Test settings - use the same cell counts as in benchmarking
    test_settings = {
        "target_ratios": [0.3, 0.5, 0.7, 0.9],
        "initial_thicknesses": [0.05, 0.1, 0.2, 0.3],
        "tolerances": [5e-3, 1e-3],
        "cell_counts": [100, 500, 1000, 5000, 10000],#ved 10000 for faster testing
        "num_seeds": 2,  # Reduced for faster testing
        "output_dir": "cohesive_sensitivity_results",
        "scale_domains": True,
        "use_reference": True,
        "target_area": 1000.0
    }
    
    # Run the analysis
    results, trajectory_data = run_sensitivity_analysis(
        domains=domains,
        **test_settings
    )
    
    print("\nSensitivity analysis complete!")
    
    if not results.empty:
        print(f"Total test cases: {len(results)}")
        print(f"Average iterations: {results['iterations'].mean():.2f}")
        print(f"Maximum iterations: {results['iterations'].max()}")
        print(f"Average error: {results['error'].mean():.6f}")
        
        # Check if any tests failed to converge
        failed = results[results["convergence_status"] != "Converged to tolerance"]
        if len(failed) > 0:
            print(f"Warning: {len(failed)} tests failed to converge within iteration limit.")
            print(f"Failed configurations:")
            for _, row in failed.iterrows():
                print(f"  Domain: {row['domain']}, Target ratio: {row['target_ratio']}, " 
                     f"Initial thickness: {row['initial_thickness']}, " 
                     f"Tolerance: {row['tolerance']}")
        else:
            print("All tests converged successfully!")