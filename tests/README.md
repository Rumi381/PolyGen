# PolyGen Tests

This directory contains various test scripts for benchmarking, performance analysis, and quality assessment of the PolyGen Python package. These tests help evaluate the performance, correctness, and robustness of the polygon mesh generation algorithms.

## Table of Contents

- [Available Test Scripts](#available-test-scripts)
  - [1. benchmark.py - Performance Benchmarking Tool](#1-benchmarkpy---performance-benchmarking-tool)
  - [2. sensCohesive.py - Cohesive Zone Sensitivity Analysis](#2-senscohesivepy---cohesive-zone-sensitivity-analysis)
  - [3. Mesh Quality Analysis Framework](#3-mesh-quality-analysis-framework)
- [Adding New Tests](#adding-new-tests)
- [Requirements](#requirements)
- [License](#license)

## Available Test Scripts

### 1. benchmark.py - Performance Benchmarking Tool

This script provides comprehensive benchmarking capabilities for the PolyGen polygon generation pipeline, measuring computational performance across different domains, grain counts, and pipeline stages.

#### Features

- Measures execution time and memory usage for each stage of the polygon generation pipeline
- Supports multiple domain geometries (from files or generated programmatically)
- Scales domains to comparable sizes to ensure fair comparisons
- Generates publication-quality plots and summary tables
- Calculates complexity metrics for domains (e.g., concavity)

#### How to Run

```bash
python benchmark.py
```

By default, the script will:
1. Load three test domains: "CalciteBoundary.obj", "WrenchGeometry.obj", and a programmatically generated rectangle with a circular hole
2. Run benchmarks with grain counts of [100, 500, 1000, 5000, 10000]
3. Save results and visualizations to the "polygen_benchmark_results" directory

#### Customizing Benchmarks

Edit the script's main section to customize:

```python
# Define domains to test
domains = [
    "./path/to/your_domain.obj",
    (Geometry.your_geometry_function, {
        'param1': value1,
        'param2': value2,
        'name': 'CustomGeometry'
    }),
]

# Define grain counts to test
point_counts = [100, 500, 1000]

# Run with multiple seeds for statistical significance
seeds = [7845, 1234, 5678]

# Run the benchmark
results = run_comprehensive_benchmark(
    domains=domains,
    point_counts=point_counts,
    seeds=seeds,
    output_dir="your_output_directory",
    scale_domains=True,
    target_area=1.5e6
)
```

#### Output

The script generates:

1. **CSV Files**:
   - Individual benchmark results for each domain/point count combination
   - Combined results in "all_benchmarks.csv"
   - Summary tables for pipeline costs, scaling behavior, and domain complexity impact

2. **Visualizations**:
   - Computational scaling (log-log plot)
   - Pipeline stage distribution (stacked area chart)
   - Domain complexity impact
   - Lloyd relaxation convergence behavior
   - Memory usage scaling

#### Interpreting Results

- **Computational Scaling**: Shows how total computation time scales with grain count, compared to theoretical O(n log n) scaling
- **Pipeline Stage Distribution**: Visualizes the relative computational cost of each pipeline stage
- **Domain Complexity Impact**: Demonstrates how domain geometry affects computational performance
- **Lloyd Convergence**: Shows how Lloyd relaxation iteration count changes with grain count
- **Memory Scaling**: Illustrates memory usage scaling compared to theoretical O(n) scaling

### 2. sensCohesive.py - Cohesive Zone Sensitivity Analysis

This script performs a comprehensive sensitivity analysis for the cohesive zone generation algorithm, examining how different parameters affect convergence behavior, accuracy, and computational efficiency.

#### Features

- Tests multiple configuration parameters:
  - Target area ratios (controls cohesive zone thickness)
  - Initial thickness values (for algorithm initialization)
  - Convergence tolerance thresholds
  - Cell/grain counts
- Tracks detailed convergence metrics and trajectories
- Extends the CohesiveZoneAdjuster class to capture convergence history
- Analyzes convergence rates to verify theoretical properties
- Supports the same domain loading and scaling features as benchmark.py

#### How to Run

```bash
python sensCohesive.py
```

By default, the script will:
1. Load the same three test domains used in benchmark.py
2. Test combinations of:
   - Target ratios: [0.3, 0.5, 0.7, 0.9]
   - Initial thicknesses: [0.05, 0.1, 0.2, 0.3]
   - Tolerances: [5e-3, 1e-3]
   - Cell counts: [100, 500, 1000, 5000, 10000]
3. Save results and visualizations to the "cohesive_sensitivity_results" directory

#### Customizing the Analysis

Edit the script's main section to customize:

```python
# Test settings
test_settings = {
    "target_ratios": [0.3, 0.5, 0.7, 0.9],
    "initial_thicknesses": [0.05, 0.1, 0.2, 0.3],
    "tolerances": [5e-3, 1e-3, 5e-4],
    "cell_counts": [100, 500, 1000],
    "num_seeds": 3,
    "output_dir": "your_results_directory",
    "scale_domains": True,
    "use_reference": False,
    "target_area": 1000.0
}

# Run the analysis
results, trajectory_data = run_sensitivity_analysis(
    domains=domains,
    **test_settings
)
```

#### Output

The script generates:

1. **CSV Files**:
   - Full sensitivity analysis results (sensitivity_analysis_full.csv)
   - Summary statistics by configuration (sensitivity_analysis_summary.csv)
   - Convergence trajectory data (convergence_trajectories_data.csv)

2. **Visualizations**:
   - Sensitivity overview with multiple plots (iterations vs target ratio, error vs iterations, etc.)
   - Convergence trajectories for different target ratios
   - Iterations heatmap by configuration
   - Convergence rate analysis
   - Effect of tolerance on convergence speed

#### Interpreting Results

- **Convergence Speed**: Understand how different parameters affect the number of iterations required to converge
- **Accuracy**: Assess the final error achieved for different configurations
- **Optimal Parameters**: Identify the best parameter combinations for different use cases
- **Convergence Rate**: Verify the theoretical quadratic convergence property of the algorithm
- **Robustness**: Determine which parameter combinations fail to converge or converge slowly

### 3. Mesh Quality Analysis Framework

A set of scripts that compare PolyGen with the open-source alternative PolyMesher, providing detailed quality metrics for polygonal meshes.

#### Files

- `lloydPolymesher.py`: Python implementation of PolyMesher's Lloyd algorithm
- `voronoiPolymesher.py`: Implementation of PolyMesher's Voronoi generation approach
- `meshQualityAnalyzer.py`: Comprehensive framework for analyzing mesh quality metrics
- `meshQualityAnalyzerMain.py`: Main script that runs the comparative analysis
- `analyzeCalciumBndy.ipynb`: Get the boundary comparison figures by comparing PolyGen and PolyMesher for the calcium Plaque boundary

#### Features

- Compares PolyGen with and without CVT optimization against PolyMesher
- Calculates publication-quality mesh metrics:
  - Shape quality (radius ratio, minimum angle, etc.)
  - Size quality (area coefficient of variation)
  - Boundary quality (Hausdorff distance)
  - Algorithmic performance (convergence energy, computation time)
- Generates standardized visualizations:
  - CDF plots for key quality metrics
  - Edge-angle correlation analysis
  - Publication-ready LaTeX tables

#### How to Run

```bash
python meshQualityAnalyzerMain.py
```

By default, the script will:
1. Load the CalciteBoundary.obj domain
2. Generate a 10,000-element mesh using each method
3. Run the quality analysis on all methods
4. Save results to the "quality_comparison_results" directory

#### Customizing the Analysis

To customize the analysis, modify the main function in `meshQualityAnalyzerMain.py`:

```python
# Set the number of points to use
N_points = 5000  # Change to your desired value

# Load a different domain
file_path = './path/to/your_domain.obj'
domain = IO.load_polygon_from_file(file_path)

# Add more methods to compare
# ...

# Set different output directory
output_dir = "your_custom_directory"
```

#### Output

The script generates:

1. **LaTeX Tables**:
   - Publication-ready quality comparison table (quality_table.tex)

2. **Visualizations**:
   - CDF plots for key metrics (quality_cdf.png)
   - Edge-angle correlation (edge_angle_correlation.png)

#### Interpreting Results

- **Shape Quality**: Higher radius ratio and minimum angle values indicate better-shaped elements
- **Size Quality**: Lower area coefficient of variation indicates more uniform element sizes
- **Boundary Quality**: Lower Hausdorff distance indicates better boundary representation
- **Energy**: Lower normalized energy (F/F₀) indicates better convergence of the Lloyd algorithm
- **Performance**: Time per element shows computational efficiency

## Adding New Tests

When adding new test scripts to this directory, please follow these guidelines:

1. Use a descriptive filename that indicates the test's purpose
2. Include documentation as Python docstrings
3. Provide command-line arguments for customization
4. Save results in a standardized format
5. Update this README.md with information about the new test

## Requirements

The test scripts require the following dependencies:

- PolyGen (core package)
- NumPy, Pandas, Matplotlib, Seaborn (data analysis and visualization)
- tqdm (progress tracking)
- psutil (memory usage monitoring)
- shapely (geometry operations)
- scipy (computational libraries)
- rtree (spatial indexing for PolyMesher implementation)
- tabulate (for formatting tables)

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn tqdm psutil shapely scipy rtree tabulate
```

## License

These test scripts are distributed under the same license as the PolyGen package.
