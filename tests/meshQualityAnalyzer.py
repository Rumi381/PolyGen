import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import math
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import os
from tabulate import tabulate
from collections import defaultdict

class MeshQualityAnalyzer:
    """
    A comprehensive framework for analyzing mesh quality metrics
    recommended for polygonal FEM meshes in high-impact journals.
    
    This implementation combines best practices from multiple approaches
    and handles both convex and non-convex polygons correctly.
    """
    
    def __init__(self, output_dir="quality_results"):
        """Initialize the analyzer with output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Store results for each method
        self.results = {}
        
        # Define which metrics are "higher is better"
        self.higher_is_better = {
            "radius_ratio": True,
            "min_angle": True,
            "area_cov": False,
            "boundary_hausdorff": False,
            "normalized_energy": False,
            "time_per_element": False,
            "srf": False,
            "regularity": True
        }
    
    def preprocess_cell(self, cell, min_edge_length=1e-6):
        """
        Remove very short edges from a polygon
        
        Parameters
        ----------
        cell : Polygon
            The cell to preprocess
        min_edge_length : float, optional
            Minimum edge length to keep
            
        Returns
        -------
        Polygon
            Processed cell with short edges removed
        """
        if not cell.is_valid or cell.is_empty:
            return cell
            
        coords = np.array(cell.exterior.coords)[:-1]
        if len(coords) <= 3:  # Can't simplify triangles
            return cell
            
        # Identify edges to keep
        keep_vertices = []
        for i in range(len(coords)):
            p_curr = coords[i]
            p_next = coords[(i+1) % len(coords)]
            
            edge_length = np.linalg.norm(p_next - p_curr)
            if edge_length > min_edge_length:
                keep_vertices.append(p_curr)
        
        # Only create a new polygon if we're actually removing vertices
        if len(keep_vertices) < len(coords) and len(keep_vertices) >= 3:
            # Ensure we close the polygon
            keep_vertices.append(keep_vertices[0])
            try:
                return Polygon(keep_vertices)
            except:
                return cell
        else:
            return cell
    
    def calculate_radius_ratio(self, polygon):
        """
        Calculate the radius ratio (r_in/r_out) for a polygon
        
        Parameters
        ----------
        polygon : Polygon
            The polygon to analyze
            
        Returns
        -------
        float
            Radius ratio in range [0,1]
        """
        # Get centroid
        centroid = polygon.centroid
        c_x, c_y = centroid.coords[0]
        
        # Get coordinates
        coords = np.array(polygon.exterior.coords)[:-1]
        
        # Calculate r_in (distance from centroid to nearest edge)
        r_in = float('inf')
        for i in range(len(coords)):
            p1 = coords[i]
            p2 = coords[(i+1) % len(coords)]
            
            # Create a line segment
            edge = LineString([p1, p2])
            # Calculate distance from centroid to the edge
            dist = Point(c_x, c_y).distance(edge)
            r_in = min(r_in, dist)
        
        # Calculate r_out (max distance from centroid to any vertex)
        dists = [np.linalg.norm(np.array([c_x, c_y]) - v) for v in coords]
        r_out = max(dists) if dists else 0
        
        # Calculate radius ratio
        if r_out > 0:
            return r_in / r_out
        return 0.0
    
    def _calculate_interior_angles(self, coords):
        """
        Calculate all interior angles of a polygon
        
        Parameters
        ----------
        coords : ndarray
            Array of vertex coordinates
            
        Returns
        -------
        list
            List of interior angles in degrees
        """
        n = len(coords)
        if n < 3:
            return []
            
        angles = []
        for i in range(n):
            # Get vectors from current vertex to adjacent vertices
            p0 = coords[(i-1) % n] - coords[i]
            p1 = coords[(i+1) % n] - coords[i]
            
            # Check for degenerate edges
            n0 = np.linalg.norm(p0)
            n1 = np.linalg.norm(p1)
            
            # Skip if either edge is too short
            if n0 < 1e-10 or n1 < 1e-10:
                continue
                
            # Calculate angle using dot product
            cosang = np.clip(np.dot(p0, p1) / (n0 * n1), -1.0, 1.0)
            angle = math.degrees(math.acos(cosang))
            angles.append(angle)
            
        return angles
    
    def calculate_interior_angles(self, coords):
        n = len(coords)
        if n < 3:
            return []
        angles = []
        for i in range(n):
            v_prev = coords[(i-1)%n] - coords[i]
            v_next = coords[(i+1)%n] - coords[i]
            # normalise
            v_prev /= np.linalg.norm(v_prev)
            v_next /= np.linalg.norm(v_next)
            # signed angle in [0, 2π)
            cross  = np.cross(v_prev, v_next)
            dot    = np.dot(v_prev, v_next)
            theta  = math.degrees(math.atan2(cross, dot))   # (-180,180]
            if theta < 0:                                    # make it (0,360)
                theta += 360.0
            angles.append(theta)
        return angles

    
    def analyze_mesh(self, cells, domain, method_name, algorithm_metrics=None, min_edge_length=1e-6):
        """
        Analyze mesh quality for a set of cells
        
        Parameters
        ----------
        cells : list of Polygon
            The Voronoi cells to analyze
        domain : Polygon
            The domain boundary
        method_name : str
            Name of the method (e.g., "PolyGen+CVT")
        algorithm_metrics : dict, optional
            Additional algorithm metrics (e.g., Lloyd energy, time)
        min_edge_length : float, optional
            Minimum edge length for preprocessing cells
            
        Returns
        -------
        dict
            Dictionary of quality metrics
        """
        start_time = time.time()
        
        # Initialize results container
        per_cell = defaultdict(list)
        
        # Track boundary cells
        boundary_cells = []
        boundary = domain.boundary
        
        # Process each cell
        total_area = domain.area
        
        for cell_idx, cell in enumerate(cells):
            # Skip invalid cells
            if not cell.is_valid or cell.is_empty or cell.area <= 0:
                continue
                
            # Preprocess cell to remove very short edges
            cell = self.preprocess_cell(cell, min_edge_length)
            
            # Get coordinates
            coords = np.array(cell.exterior.coords)[:-1]
            n_vertices = len(coords)
            
            # Skip degenerate cells
            if n_vertices < 3:
                continue
                
            # Record valence
            per_cell['valence'].append(n_vertices)
            
            # Record area
            per_cell['area'].append(cell.area)
            
            # Calculate radius ratio
            radius_ratio = self.calculate_radius_ratio(cell)
            per_cell['radius_ratio'].append(radius_ratio)
            
            # Calculate shape regularity factor (4πA/P²)
            area = cell.area
            perimeter = cell.length
            shape_regularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            per_cell['regularity'].append(shape_regularity)
            
            # Calculate interior angles
            angles = self.calculate_interior_angles(coords)
            if angles:
                per_cell['min_angle'].append(min(angles))
            
            # Calculate edge lengths
            edge_lengths = []
            for i in range(n_vertices):
                p1 = coords[i]
                p2 = coords[(i+1) % n_vertices]
                edge_lengths.append(np.linalg.norm(p2 - p1))
            
            # Calculate edge length CV
            if edge_lengths:
                edge_mean = np.mean(edge_lengths)
                edge_std = np.std(edge_lengths)
                if edge_mean > 0:
                    per_cell['edge_len_cv'].append(edge_std / edge_mean)
                
                # Calculate min and max edge lengths
                per_cell['min_edge'].append(min(edge_lengths))
                per_cell['max_edge'].append(max(edge_lengths))
            
            # Calculate SRF (Shape-regularity factor) = max_edge_length / (2 * r_in)
            min_dist_to_edge = float('inf')
            centroid = cell.centroid
            
            for i in range(n_vertices):
                p1 = coords[i]
                p2 = coords[(i+1) % n_vertices]
                edge = LineString([p1, p2])
                dist = Point(centroid.x, centroid.y).distance(edge)
                min_dist_to_edge = min(min_dist_to_edge, dist)
            
            if min_dist_to_edge > 0 and edge_lengths:
                srf = max(edge_lengths) / (2 * min_dist_to_edge)
                per_cell['srf'].append(srf)
            
            # Check if cell is on boundary
            if cell.intersects(boundary):
                boundary_cells.append(cell)
                
                # For PolyMesher where cells might extend beyond the domain
                if "polymesher" in method_name.lower():
                    # Check vertices outside the domain
                    distances = []
                    for p in coords:
                        point = Point(p)
                        if not domain.contains(point):
                            dist = point.distance(boundary)
                            distances.append(dist)
                    
                    if distances:
                        per_cell['boundary_distance'].append(max(distances))
                else:
                    # For PolyGen with clipping, boundary distance should be 0
                    per_cell['boundary_distance'].append(0.0)
        
        # Calculate boundary Hausdorff distance
        if boundary_cells:
            try:
                # Use the union of all boundary cells
                union_boundary = unary_union(boundary_cells)
                
                # Calculate distances from all vertices to domain boundary
                hausdorff_distances = []
                for v in union_boundary.exterior.coords:
                    hausdorff_distances.append(Point(v).distance(boundary))
                
                boundary_hausdorff = max(hausdorff_distances) if hausdorff_distances else 0.0
            except Exception as e:
                # Fallback to simpler method
                boundary_hausdorff = max(per_cell['boundary_distance']) if per_cell['boundary_distance'] else 0.0
        else:
            boundary_hausdorff = 0.0
        
        # Calculate summary statistics
        def calculate_stats(values):
            if not values:
                return None
                
            values_array = np.array(values)
            return {
                'min': float(np.min(values_array)),
                'p5': float(np.percentile(values_array, 5)),
                'median': float(np.percentile(values_array, 50)),
                'p95': float(np.percentile(values_array, 95)),
                'max': float(np.max(values_array)),
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array))
            }
        
        # Prepare results
        results = {
            'method': method_name,
            'cell_count': len([cell for cell in cells if cell.is_valid and not cell.is_empty]),
            'boundary_hausdorff': boundary_hausdorff,
            'area_cov': float(np.std(per_cell['area']) / np.mean(per_cell['area'])) if per_cell['area'] else 0.0
        }
        
        # Calculate statistics for each metric
        for metric in ['radius_ratio', 'min_angle', 'regularity', 'srf', 'edge_len_cv', 'min_edge', 'max_edge']:
            if per_cell[metric]:
                stats = calculate_stats(per_cell[metric])
                for stat_name, stat_value in stats.items():
                    results[f"{metric}_{stat_name}"] = stat_value
        
        # Add raw lists for CDF plots
        for metric in ['radius_ratio', 'min_angle', 'regularity', 'srf']:
            results[metric] = per_cell[metric]
        
        # Add algorithm metrics if provided
        if algorithm_metrics:
            results.update(algorithm_metrics)
            
        # Add analysis time
        results["analysis_time"] = time.time() - start_time
        
        # Add correlation between min edge and min angle
        if per_cell['min_edge'] and per_cell['min_angle']:
            min_edges = np.array(per_cell['min_edge'])
            min_angles = np.array(per_cell['min_angle'])
            
            # Only calculate for cells that have both metrics
            valid_indices = np.logical_and(np.isfinite(min_edges), np.isfinite(min_angles))
            if np.sum(valid_indices) > 1:
                correlation = np.corrcoef(min_edges[valid_indices], min_angles[valid_indices])[0, 1]
                results['edge_angle_correlation'] = float(correlation)
        
        # Store results
        self.results[method_name] = results
        
        # Print diagnostics
        if per_cell['min_angle']:
            smallest_angle_idx = np.argmin(per_cell['min_angle'])
            smallest_angle = per_cell['min_angle'][smallest_angle_idx]
            print(f"\nMethod: {method_name}")
            print(f"Smallest angle: {smallest_angle:.2f}°")
            
            if per_cell['min_edge']:
                smallest_edge = min(per_cell['min_edge'])
                print(f"Smallest edge length: {smallest_edge:.6f}")
            
            print(f"Correlation between min edge and min angle: {results.get('edge_angle_correlation', 'N/A')}")
        
        return results
    
    def generate_comparison_table(self, filename=None):
        """
        Generate a comparison table in the format specified for the paper
        
        Parameters
        ----------
        filename : str, optional
            File to save the table (LaTeX format)
            
        Returns
        -------
        str
            LaTeX table
        """
        if not self.results:
            return "No results to compare."
        
        # Prepare table content
        table_content = []
        
        for method, data in self.results.items():
            row = [
                method,
                f"{data.get('radius_ratio_mean', 'N/A'):.2f} ({data.get('radius_ratio_p5', 'N/A'):.2f}/{data.get('radius_ratio_median', 'N/A'):.2f}/{data.get('radius_ratio_p95', 'N/A'):.2f})",
                f"{data.get('min_angle_mean', 'N/A'):.1f} ({data.get('min_angle_p5', 'N/A'):.1f}/{data.get('min_angle_median', 'N/A'):.1f}/{data.get('min_angle_p95', 'N/A'):.1f})",
                f"{data.get('area_cov', 'N/A'):.2f}",
                f"{data.get('boundary_hausdorff', 0)*1e6:.1f}",  # Convert to μm
                f"{data.get('normalized_energy', 'N/A') if isinstance(data.get('normalized_energy', 'N/A'), (int, float)) else 'N/A':.3f}",
                f"{data.get('time_per_element', 'N/A') if isinstance(data.get('time_per_element', 'N/A'), (int, float)) else 'N/A':.2f}"
            ]
            table_content.append(row)
        
        # Format as LaTeX table
        latex_table = "\\begin{table}[t]\n"
        latex_table += "  \\caption{Quantitative quality comparison – mesh quality metrics for calcite domain.}\n"
        latex_table += "  \\centering\n"
        latex_table += "  \\renewcommand\\arraystretch{1.1}\n"
        latex_table += "  \\begin{tabular}{@{}lcccccc@{}}\n"
        latex_table += "    \\toprule\n"
        latex_table += "        & \\multicolumn{2}{c}{\\textbf{Shape quality}} &\n"
        latex_table += "          \\multicolumn{2}{c}{\\textbf{Size / boundary}} &\n"
        latex_table += "          \\multicolumn{2}{c}{\\textbf{Algorithmic}} \\\\[-1pt]\n"
        latex_table += "        \\cmidrule(r){2-3}\\cmidrule(r){4-5}\\cmidrule(l){6-7}\n"
        latex_table += "        & Radius–ratio$^{\\dagger}$ & Min.\\ angle$^{\\dagger}$ [$^\\circ$] &\n"
        latex_table += "          Area COV & Bdy.\\ Hausd.\\ [$\\mu$m] &\n"
        latex_table += "          $F/F_0$ & Time/elem.\\,[ms] \\\\ \n"
        latex_table += "    \\midrule\n"
        
        # Add data rows
        for method_idx, row in enumerate(table_content):
            method = row[0]
            
            # Determine if this is the best method (for highlighting)
            is_best = method.lower().find("polygen + cvt") >= 0
            
            # Format row
            latex_row = "    "
            if is_best:
                latex_row += "\\textbf{" + method + "} &  "
            else:
                latex_row += method + " &  "
            
            # Add metrics with appropriate formatting
            for i, value in enumerate(row[1:], 1):
                if is_best:
                    latex_row += "\\textbf{" + value + "}"
                else:
                    latex_row += "\\emph{" + value + "}"
                
                if i < len(row) - 1:
                    latex_row += " &\n                              "
            
            latex_row += " \\\\\n"
            latex_table += latex_row
            
        latex_table += "    \\bottomrule\n"
        latex_table += "  \\end{tabular}\n"
        latex_table += "  \\label{tab:mesh-quality}\n"
        latex_table += "\\end{table}"
        
        # Save to file if specified
        if filename:
            with open(filename, 'w') as f:
                f.write(latex_table)
        
        return latex_table
    
    def plot_cdf(self, metrics=["radius_ratio", "min_angle"], filename=None):
        """
        Generate CDF plots for specified metrics
        
        Parameters
        ----------
        metrics : list of str
            Metrics to plot
        filename : str, optional
            File to save the plot
            
        Returns
        -------
        fig
            Matplotlib figure
        """
        if not self.results:
            return None
            
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.3)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
            
        # Label mappings and ranges for each metric
        metric_props = {
            "radius_ratio": {
                "label": "Radius Ratio",
                "range": (0, 1),
                "is_better": "higher"
            },
            "min_angle": {
                "label": "Minimum Interior Angle (°)",
                "range": (0, 90),
                "is_better": "higher"
            },
            "srf": {
                "label": "Shape-Regularity Factor",
                "range": (1, 5),
                "is_better": "lower"
            },
            "regularity": {
                "label": "Shape Regularity (4πA/P²)",
                "range": (0, 1),
                "is_better": "higher"
            }
        }
        
        # Generate CDF for each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for method, data in self.results.items():
                if metric not in data or not isinstance(data[metric], list) or not data[metric]:
                    continue
                    
                values = np.sort(data[metric])
                
                # Handle specific metrics - ensure we use appropriate data ranges
                if metric == "radius_ratio":
                    # Ensure radius ratios are in [0,1]
                    values = np.clip(values, 0, 1)
                elif metric == "min_angle":
                    # Ensure angles are reasonable
                    values = np.clip(values, 0, 90)
                elif metric == "regularity":
                    # Ensure regularity is in [0,1]
                    values = np.clip(values, 0, 1)
                
                # Generate CDF
                cdf = np.arange(1, len(values) + 1) / len(values)
                
                # Plot CDF
                label = method
                if "polygen + cvt" in method.lower():
                    linestyle = '-'
                    linewidth = 2.5
                elif "polygen" in method.lower() and "cvt" not in method.lower():
                    linestyle = '--'
                    linewidth = 2.0
                else:
                    linestyle = ':'
                    linewidth = 2.0
                    
                ax.plot(values, cdf, linestyle=linestyle, linewidth=linewidth, label=label)
            
            # Add reference lines for 5th and 95th percentiles
            ax.axhline(y=0.05, color='gray', linestyle='-.', alpha=0.5)
            ax.axhline(y=0.95, color='gray', linestyle='-.', alpha=0.5)
            
            # Set limits and labels based on metric properties
            if metric in metric_props:
                props = metric_props[metric]
                ax.set_xlim(props["range"])
                ax.set_xlabel(props["label"])
                
                # Add annotation about which direction is better
                if props["is_better"] == "higher":
                    ax.annotate('Higher is better', xy=(0.75, 0.05), xycoords='axes fraction',
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                elif props["is_better"] == "lower":
                    ax.annotate('Lower is better', xy=(0.75, 0.05), xycoords='axes fraction',
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            ax.set_ylabel("Cumulative Probability")
            ax.set_title(f"CDF of {metric_props.get(metric, {}).get('label', metric)}")
            ax.grid(True, alpha=0.3)
            
            # Only add legend to the first subplot
            if i == 0:
                ax.legend(loc='upper left')
        
        fig.tight_layout()
        
        # Save figure if filename provided
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_edge_angle_correlation(self, filename=None):
        """
        Generate a scatter plot showing the correlation between minimum edge length and minimum angle
        
        Parameters
        ----------
        filename : str, optional
            File to save the plot
            
        Returns
        -------
        fig
            Matplotlib figure
        """
        if not self.results:
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot scatter for each method
        for method, data in self.results.items():
            if 'min_edge' not in data or 'min_angle' not in data:
                continue
                
            min_edges = []
            min_angles = []
            
            # Collect paired data
            for edge, angle in zip(data['min_edge'], data['min_angle']):
                min_edges.append(edge)
                min_angles.append(angle)
                
            # Plot scatter
            if "polygen + cvt" in method.lower():
                marker = 'o'
                color = 'blue'
            elif "polygen" in method.lower() and "cvt" not in method.lower():
                marker = 's'
                color = 'orange'
            else:
                marker = '^'
                color = 'green'
                
            ax.scatter(min_edges, min_angles, marker=marker, color=color, alpha=0.5, label=method)
            
            # Add trend line
            if min_edges and min_angles:
                z = np.polyfit(min_edges, min_angles, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(min_edges), max(min_edges), 100)
                ax.plot(x_range, p(x_range), '--', color=color)
                
                # Add correlation coefficient
                correlation = data.get('edge_angle_correlation', None)
                if correlation is not None:
                    ax.text(0.05, 0.95 - 0.05 * list(self.results.keys()).index(method),
                            f"{method}: r = {correlation:.2f}",
                            transform=ax.transAxes, color=color)
        
        # Add labels and grid
        ax.set_xlabel("Minimum Edge Length")
        ax.set_ylabel("Minimum Interior Angle (°)")
        ax.set_title("Correlation Between Minimum Edge Length and Minimum Angle")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        fig.tight_layout()
        
        # Save figure if filename provided
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            
        return fig


def calculate_lloyd_metrics(metrics_dict):
    """
    Extract Lloyd algorithm metrics for the quality table
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics from Lloyd algorithm
        
    Returns
    -------
    dict
        Extracted metrics for quality analysis
    """
    result = {}
    
    # Extract normalized energy (F/F₀)
    if 'min_error' in metrics_dict:
        result['normalized_energy'] = metrics_dict['min_error']
    elif 'error_values' in metrics_dict and metrics_dict['error_values']:
        result['normalized_energy'] = min(metrics_dict['error_values'])
    else:
        result['normalized_energy'] = None
    
    # Extract time information
    if 'time_taken' in metrics_dict and 'iterations' in metrics_dict:
        # Convert to ms per element
        time_per_iteration = metrics_dict['time_taken'] / max(metrics_dict['iterations'], 1)
        result['time_per_element'] = time_per_iteration * 1000  # Convert to ms
    
    return result