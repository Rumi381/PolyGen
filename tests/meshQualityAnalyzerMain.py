from shapely.affinity import scale
from polygen.polygen2d import IO
from polygen.polygen2d import generate_sequence_points
from polygen.polygen2d import lloyd_with_density as lloyd
from polygen.polygen2d import VoronoiGenerator, MeshOptimizer
from lloydPolymesher import polymesher_lloyd
from voronoiPolymesher import PolyMesherVoronoiGenerator
import time
import os
import numpy as np

# Import our quality analyzer
from meshQualityAnalyzer import MeshQualityAnalyzer, calculate_lloyd_metrics

def scale_domain_to_reference(domain, reference_domain=None, target_area=1000.0):
    """Scale a domain to match a reference domain's area or to a target area."""
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

def main():
    # Set up output directory
    output_dir = "quality_comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize quality analyzer
    analyzer = MeshQualityAnalyzer(output_dir=output_dir)
    
    # Load domain
    file_path = './examples/CalciteBoundary.obj'
    domain = IO.load_polygon_from_file(file_path)
    
    # Scale domain for consistent analysis
    scaled_domain = scale_domain_to_reference(domain, target_area=1.5e6)
    
    # Generate seed points (same for all methods)
    N_points = 10000
    points_seed = 7845
    sequencePoints = generate_sequence_points(domain=scaled_domain, N=N_points, 
                                            seed=points_seed, margin=0.05)
    
    print(f"Analyzing mesh quality for domain with {N_points} points...")
    
    # ==================== Method 1: PolyGen + CVT ====================
    print("\n1. Running PolyGen + CVT...")
    polygen_cvt_start = time.time()
    
    # Run Lloyd with decay
    relaxedSequencePointsWithDecay, metrics_relaxedSequencePointsWithDecay = lloyd(
        polygon=scaled_domain, 
        seed_points=sequencePoints, 
        density_function=None, 
        max_iterations=10000000, 
        tol=5e-3, 
        use_decay=True, 
        grad_increase_tol=10000
    )
    
    # Generate Voronoi cells
    voronoi_generator = VoronoiGenerator(buffer_factor=1.0)
    calcite_voronoi = voronoi_generator.generate_cells(
        domain=scaled_domain,
        points=relaxedSequencePointsWithDecay
    )
    
    # Optimize cells (CVT phase)
    optimizer = MeshOptimizer(verbose=True)
    optimized_calciteVoronoi = optimizer.optimize_voronoi_cells(
        voronoi_cells=calcite_voronoi, 
        threshold=0.02
    )
    
    polygen_cvt_time = time.time() - polygen_cvt_start
    
    # Extract metrics
    lloyd_metrics = calculate_lloyd_metrics(metrics_relaxedSequencePointsWithDecay)
    lloyd_metrics['time_per_element'] = polygen_cvt_time * 1000 / N_points  # ms per element
    
    # Analyze mesh quality
    analyzer.analyze_mesh(
        cells=optimized_calciteVoronoi, 
        domain=scaled_domain, 
        method_name="PolyGen + CVT",
        algorithm_metrics=lloyd_metrics
    )
    
    # ==================== Method 2: PolyGen (CLIP only) ====================
    print("\n2. Running PolyGen (CLIP only)...")
    polygen_clip_start = time.time()
    
    # Generate Voronoi cells directly (no optimization)
    calcite_voronoi_clip = voronoi_generator.generate_cells(
        domain=scaled_domain,
        points=sequencePoints  # Use original points without relaxation
    )
    
    polygen_clip_time = time.time() - polygen_clip_start
    
    # Create metrics for CLIP-only (using a theoretically accurate value of 1.0)
    clip_metrics = {
        'normalized_energy': 1.0,  # No energy reduction has occurred
        'time_per_element': polygen_clip_time * 1000 / N_points  # ms per element
    }
    
    # Analyze mesh quality
    analyzer.analyze_mesh(
        cells=calcite_voronoi_clip, 
        domain=scaled_domain, 
        method_name="PolyGen (CLIP only)",
        algorithm_metrics=clip_metrics
    )
    
    # ==================== Method 3: PolyMesher ====================
    print("\n3. Running PolyMesher...")
    polymesher_start = time.time()
    
    # Run PolyMesher Lloyd algorithm
    relaxedPolymesher, polymesher_metrics = polymesher_lloyd(
        polygon=scaled_domain, 
        seed_points=sequencePoints, 
        density_function=None, 
        max_iterations=10000000, 
        tol=5e-3
    )
    
    # Generate PolyMesher cells
    polymesher_generator = PolyMesherVoronoiGenerator(reflection_factor=1.5)
    polymesher_voronoi = polymesher_generator.generate_cells(
        domain=scaled_domain, 
        points=relaxedPolymesher
    )

    # Optimize cells (CVT phase)
    optimizer = MeshOptimizer(verbose=True)
    optimized_polymesher_voronoi = optimizer.optimize_voronoi_cells(
        voronoi_cells=polymesher_voronoi, 
        threshold=0.02
    )
    
    polymesher_time = time.time() - polymesher_start
    
    # Extract metrics
    polymesher_lloyd_metrics = calculate_lloyd_metrics(polymesher_metrics)
    polymesher_lloyd_metrics['time_per_element'] = polymesher_time * 1000 / N_points  # ms per element
    
    # Analyze mesh quality
    analyzer.analyze_mesh(
        cells=optimized_polymesher_voronoi, 
        domain=scaled_domain, 
        method_name="PolyMesher",
        algorithm_metrics=polymesher_lloyd_metrics
    )
    
    # ==================== Generate Results ====================
    print("\nGenerating comparison results...")
    
    # Generate LaTeX table
    latex_table = analyzer.generate_comparison_table(
        filename=os.path.join(output_dir, "quality_table.tex")
    )
    print("\nLaTeX Table generated:")
    print(latex_table)
    
    # Generate CDF plots
    print("\nGenerating CDF plots...")
    analyzer.plot_cdf(
        metrics=["radius_ratio", "min_angle"],
        filename=os.path.join(output_dir, "quality_cdf.png")
    )
    
    # Generate correlation plot
    print("\nGenerating edge-angle correlation plot...")
    analyzer.plot_edge_angle_correlation(
        filename=os.path.join(output_dir, "edge_angle_correlation.png")
    )
    
    print(f"\nAll results saved to {output_dir}")

if __name__ == "__main__":
    main()