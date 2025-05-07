import numpy as np
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial import Voronoi
import time
import warnings
from shapely import prepared
from collections import deque
from rtree import index

class SpatialIndex:
    """
    Spatial indexing wrapper for efficient point-in-polygon tests.
    Only used for non-uniform density cases (not in original PolyMesher).
    """
    def __init__(self, polygon, resolution=0.01):
        self.polygon = polygon
        self.prepared_polygon = prepared.prep(polygon)
        self.index = self._build_index(polygon, resolution)
        
    def _build_index(self, polygon, resolution):
        """Build a spatial index for the polygon boundary"""
        idx = index.Index()
        
        # Index the exterior boundary
        self._add_boundary_to_index(idx, polygon.exterior.coords, 0, resolution)
        
        # Index any interior boundaries (holes)
        for i, interior in enumerate(polygon.interiors):
            self._add_boundary_to_index(idx, interior.coords, i+1, resolution)
            
        return idx
    
    def _add_boundary_to_index(self, idx, coords, boundary_id, resolution):
        """Add a boundary segment to the spatial index"""
        coords = list(coords)
        for i in range(len(coords)-1):
            p1 = coords[i]
            p2 = coords[i+1]
            
            # Create a bounding box for this segment
            bbox = (
                min(p1[0], p2[0]), min(p1[1], p2[1]),
                max(p1[0], p2[0]), max(p1[1], p2[1])
            )
            
            # Add segment to index
            segment_id = (boundary_id, i)
            idx.insert(id=hash(segment_id), coordinates=bbox)
    
    def filter_points(self, points, buffer_distance=1e-10):
        """
        Efficiently filter points that might be inside the polygon.
        """
        n_points = len(points)
        result = np.zeros(n_points, dtype=bool)
        
        # Expand the bounds slightly to catch edge cases
        bounds = self.polygon.bounds
        expanded_bounds = (
            bounds[0] - buffer_distance,
            bounds[1] - buffer_distance,
            bounds[2] + buffer_distance,
            bounds[3] + buffer_distance
        )
        
        # First quick filter: points outside expanded bounds are definitely outside
        in_bounds_mask = (
            (points[:, 0] >= expanded_bounds[0]) &
            (points[:, 1] >= expanded_bounds[1]) &
            (points[:, 0] <= expanded_bounds[2]) &
            (points[:, 1] <= expanded_bounds[3])
        )
        
        # Only test points within bounds
        candidate_indices = np.where(in_bounds_mask)[0]
        
        # If no candidates, return all False
        if len(candidate_indices) == 0:
            return result
        
        # Perform exact containment test only on candidates
        candidate_points = points[candidate_indices]
        candidate_results = np.array([
            self.prepared_polygon.contains(Point(p)) for p in candidate_points
        ])
        
        # Update result array
        result[candidate_indices] = candidate_results
        
        return result

class PointHistoryBuffer:
    """
    Memory-efficient circular buffer for storing point configurations.
    Not in original PolyMesher but useful for tracking optimal configurations.
    """
    def __init__(self, max_size, point_shape, dtype=np.float64):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.point_shape = point_shape
        self.dtype = dtype
        
        # Track optimal configuration
        self.optimal_config = None
        self.optimal_norm = float('inf')
        self.optimal_iteration = -1
    
    def append(self, points, iteration, grad_norm):
        """Add a new point configuration to the buffer."""
        self.buffer.append(points.copy())
        
        # Update optimal configuration if necessary
        if grad_norm < self.optimal_norm:
            self.optimal_norm = grad_norm
            self.optimal_config = points.copy()
            self.optimal_iteration = iteration
    
    def get_optimal_config(self):
        """Retrieve the configuration with the lowest gradient norm."""
        return self.optimal_config.copy() if self.optimal_config is not None else None
        
    def __len__(self):
        return len(self.buffer)

def domain_dist_function(points, polygon):
    """
    Compute distance components to each segment of the polygon boundary.
    This emulates the Domain('Dist', P) call in PolyMesher.
    
    Parameters
    ----------
    points : ndarray
        Array of point coordinates
    polygon : shapely.geometry.Polygon
        The domain boundary
    
    Returns
    -------
    ndarray
        Matrix of distance values with shape (n_points, n_segments+1)
        The last column contains the maximum of the distance components
    """
    n_points = len(points)
    
    # Get boundary segments
    boundary_segments = []
    boundary_segments.append(list(polygon.exterior.coords)[:-1])  # Exclude last point (same as first)
    
    for interior in polygon.interiors:
        boundary_segments.append(list(interior.coords)[:-1])
    
    # Calculate total number of segments
    total_segments = sum(len(segment) for segment in boundary_segments)
    
    # Initialize distance matrix
    d = np.zeros((n_points, total_segments + 1))
    
    # Compute distances to each segment
    seg_idx = 0
    for i, segment in enumerate(boundary_segments):
        is_interior = i > 0  # First segment is exterior, others are interior
        
        for j in range(len(segment)):
            # Get segment endpoints
            p1 = segment[j]
            p2 = segment[(j+1) % len(segment)]
            
            # Compute distances using vectorized operations
            # Vector from p1 to p2
            v = np.array(p2) - np.array(p1)
            # Length of segment squared
            l2 = np.sum(v**2)
            
            if l2 == 0:  # p1 and p2 are the same point
                dist = np.sqrt(np.sum((points - np.array(p1))**2, axis=1))
            else:
                # Vector from p1 to points
                v1 = points - np.array(p1)
                
                # Projection of v1 onto v (dot product)
                t = np.sum(v1 * v, axis=1) / l2
                
                # Clamp t to [0, 1] for line segment
                t = np.clip(t, 0, 1)
                
                # Closest point on segment
                projection = np.array(p1) + t.reshape(-1, 1) * v
                
                # Distance to closest point
                dist = np.sqrt(np.sum((points - projection)**2, axis=1))
                
                # Calculate sign (positive if point is to the right of directed segment)
                cross = v[0] * (points[:, 1] - p1[1]) - v[1] * (points[:, 0] - p1[0])
                dist = dist * np.sign(cross)
            
            # Adjust sign based on whether this is interior or exterior boundary
            if is_interior:
                dist = -dist  # Interior boundaries have opposite sign convention
                
            d[:, seg_idx] = dist
            seg_idx += 1
    
    # Last column is the maximum value across all components
    # This follows PolyMesher's approach for the distance function
    d[:, -1] = np.max(d[:, :-1], axis=1)
    
    return d

def generate_reflections(points, polygon, alpha, eps=1e-8, eta=0.9):
    """
    Generate reflections of points near the boundary of the domain.
    This closely follows PolyMesher's reflection strategy.
    
    Parameters
    ----------
    points : ndarray
        Array of point coordinates
    polygon : shapely.geometry.Polygon
        The domain boundary
    alpha : float
        Parameter controlling how close to boundary points need to be for reflection
    eps : float, optional
        Small value for numerical differentiation (default=1e-8)
    eta : float, optional
        Threshold for reflection acceptance (default=0.9)
    
    Returns
    -------
    ndarray
        Array of reflection points
    """
    n_points = len(points)
    
    # Get distance components using the domain distance function
    d = domain_dist_function(points, polygon)
    n_bdry_segs = d.shape[1] - 1
    
    # Compute normal vectors via numerical differentiation
    # This closely follows PolyMesher's approach
    points_x_shift = points.copy()
    points_x_shift[:, 0] += eps
    d_x_shift = domain_dist_function(points_x_shift, polygon)
    n1 = (d_x_shift - d) / eps
    
    points_y_shift = points.copy()
    points_y_shift[:, 1] += eps
    d_y_shift = domain_dist_function(points_y_shift, polygon)
    n2 = (d_y_shift - d) / eps
    
    # Identify points near boundary segments (within alpha distance)
    near_boundary = np.abs(d[:, :-1]) < alpha
    
    # Create a list to store reflection points and corresponding distances
    reflections = []
    original_distances = []
    
    # Process each point-segment pair that's near a boundary
    for i in range(n_points):
        for j in range(n_bdry_segs):
            if near_boundary[i, j]:
                # Calculate reflection for this point across this boundary segment
                # Using PolyMesher's reflection formula: P1(I)-2*n1(I).*d(I)
                reflection_x = points[i, 0] - 2 * n1[i, j] * d[i, j]
                reflection_y = points[i, 1] - 2 * n2[i, j] * d[i, j]
                reflections.append((reflection_x, reflection_y))
                original_distances.append(np.abs(d[i, j]))
    
    if not reflections:
        return np.zeros((0, 2))
    
    # Convert to numpy array
    reflections = np.array(reflections)
    original_distances = np.array(original_distances)
    
    # Check validity: reflected points must be outside domain and satisfy distance criterion
    d_reflections = domain_dist_function(reflections, polygon)
    
    # Apply validity criteria from PolyMesher
    # Points must be outside domain (d_R_P(:,end) > 0) and 
    # satisfy abs(d_R_P(:,end)) >= eta*abs(d(I))
    valid = (d_reflections[:, -1] > 0) & (np.abs(d_reflections[:, -1]) >= eta * original_distances)
    
    if not np.any(valid):
        return np.zeros((0, 2))
    
    # Return valid reflections
    valid_reflections = reflections[valid]
    
    # Remove duplicates (unique in MATLAB)
    unique_reflections = np.unique(valid_reflections, axis=0)
    
    return unique_reflections

def compute_polygon_centroid(vertices_x, vertices_y):
    """
    Compute the centroid of a polygon using the formula from PolyMesher.
    This matches the calculation in PolyMshr_CntrdPly.
    
    Parameters
    ----------
    vertices_x, vertices_y : ndarray
        Arrays of vertex coordinates
    
    Returns
    -------
    tuple
        (x_centroid, y_centroid, area) of the polygon
    """
    # Number of vertices
    n = len(vertices_x)
    
    # Shifted vertices (equivalent to vxS, vyS in PolyMesher)
    vx_shifted = np.roll(vertices_x, -1)
    vy_shifted = np.roll(vertices_y, -1)
    
    # Calculate intermediate terms
    temp = vertices_x * vy_shifted - vertices_y * vx_shifted
    
    # Calculate area
    area = 0.5 * np.sum(temp)
    
    # Check for zero area
    if abs(area) < 1e-10:
        # Return mean of vertices for near-zero area
        return np.mean(vertices_x), np.mean(vertices_y), abs(area)
    
    # Calculate centroid components using PolyMesher's formula
    x_centroid = np.sum((vertices_x + vx_shifted) * temp) / (6 * area)
    y_centroid = np.sum((vertices_y + vy_shifted) * temp) / (6 * area)
    
    return x_centroid, y_centroid, abs(area)

def create_density_grid(bounds, samples, density_fn):
    """
    Create a discretized grid and compute density values for numerical integration.
    Only used for non-uniform density cases (not in original PolyMesher).
    """
    x_min, y_min, x_max, y_max = bounds
    x = np.linspace(x_min, x_max, samples)
    y = np.linspace(y_min, y_max, samples)
    X, Y = np.meshgrid(x, y)
    
    # Vectorized density computation
    density_vals = density_fn(X, Y)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    return X, Y, density_vals, dx, dy

def compute_weighted_centroid(X, Y, density_vals, mask, dx, dy):
    """
    Compute the density-weighted centroid of a region using numerical integration.
    Only used for non-uniform density cases (not in original PolyMesher).
    """
    masked_density = density_vals * mask
    total_mass = np.sum(masked_density) * dx * dy
    
    if total_mass < 1e-10:
        # Return geometric centroid based on the masked area
        x_centroid = np.sum(X * mask) / np.sum(mask)
        y_centroid = np.sum(Y * mask) / np.sum(mask)
        return x_centroid, y_centroid, total_mass
        
    x_centroid = np.sum(X * masked_density) * dx * dy / total_mass
    y_centroid = np.sum(Y * masked_density) * dx * dy / total_mass
    
    return x_centroid, y_centroid, total_mass

def polymesher_lloyd(polygon, seed_points, density_function=None, max_iterations=1000, 
                    tol=5e-3, verbose=False, history_buffer_size=10, ignore_unbounded=False):
    """
    Python implementation of PolyMesher's Lloyd algorithm.
    This closely follows the approach in the original MATLAB version.
    
    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The domain in which to compute the centroidal Voronoi tessellation.
    seed_points : array_like
        Initial seed points inside the polygon.
    density_function : callable, optional
        A function f(x, y) -> float that defines pointwise density. If None,
        a uniform density of 1 is assumed. This extends PolyMesher's capabilities.
    max_iterations : int, optional
        Maximum number of Lloyd iterations (default=1000).
    tol : float, optional
        Convergence tolerance for the error measure (default=5e-3).
    verbose : bool, optional
        Whether to print detailed progress information (default=False).
    history_buffer_size : int, optional
        Maximum size of the point configuration buffer (default=10).
    ignore_unbounded : bool, optional
        Whether to ignore unbounded Voronoi cells (default=False).
        
    Returns
    -------
    tuple
        A tuple (points, metrics) where:
        - points: The final point configuration as a numpy array
        - metrics: A dictionary containing convergence information and error history
    """
    # Initial setup
    is_uniform_density = density_function is None
    density_fn = (lambda x, y: np.ones_like(x)) if is_uniform_density else density_function
    
    seed_points = np.array(seed_points)
    n_points = len(seed_points)
    total_area = polygon.area
    
    # Constants from original PolyMesher
    c = 1.5  # Constant for Alpha calculation
    
    # Initialize tracking variables
    error = 1.0
    it = 0
    
    # Initialize point buffer for tracking configurations
    point_buffer = PointHistoryBuffer(
        max_size=history_buffer_size,
        point_shape=seed_points.shape
    )
    
    # Initialize metrics dictionary
    metrics = {
        'error_values': [],
        'convergence_status': None,
        'iterations': 0,
        'time_taken': 0,
        'min_error': float('inf'),
        'min_error_iteration': 0
    }
    
    # Store initial configuration
    current_points = seed_points.copy()
    point_buffer.append(current_points, 0, float('inf'))
    
    start_time = time.time()
    
    # Main Lloyd iteration loop
    if verbose:
        print(f"Running Lloyd's algorithm for up to {max_iterations} iterations")
    
    try:
        while it < max_iterations and error > tol:
            # Calculate Alpha based on PolyMesher approach
            alpha = c * np.sqrt(total_area / n_points)
            
            # Store previous points (P = Pc from PolyMesher)
            points = current_points.copy()
            
            # Generate reflections (R_P in PolyMesher)
            reflections = generate_reflections(points, polygon, alpha)
            
            # Construct Voronoi diagram with reflections
            if len(reflections) > 0:
                vor = Voronoi(np.vstack([points, reflections]))
            else:
                vor = Voronoi(points)
            
            # Initialize arrays for new centroids and areas
            new_centroids = np.zeros_like(points)
            areas = np.zeros(n_points)
            
            # Process Voronoi cells and compute centroids
            for point_idx in range(n_points):
                region_idx = vor.point_region[point_idx]
                vertices_idx = vor.regions[region_idx]
                
                # Skip if region contains -1 (unbounded region)
                if -1 in vertices_idx or len(vertices_idx) == 0:
                    new_centroids[point_idx] = points[point_idx]
                    continue
                
                # Get vertices of the Voronoi cell
                vertices = vor.vertices[vertices_idx]
                vx = vertices[:, 0]
                vy = vertices[:, 1]
                
                # Compute centroid using PolyMesher's method for uniform density
                if is_uniform_density:
                    try:
                        x_centroid, y_centroid, area = compute_polygon_centroid(vx, vy)
                        new_centroids[point_idx] = [x_centroid, y_centroid]
                        areas[point_idx] = area
                    except Exception as e:
                        # Fallback to original point if error in centroid calculation
                        warnings.warn(f"Error in centroid calculation: {str(e)}. Using original point.")
                        new_centroids[point_idx] = points[point_idx]
                else:
                    # Non-uniform density case (extension to PolyMesher)
                    # Create polygon for the cell
                    try:
                        cell = Polygon(vertices)
                        if not cell.is_valid:
                            cell = cell.buffer(0)
                        
                        # Skip empty cells
                        if cell.is_empty:
                            new_centroids[point_idx] = points[point_idx]
                            continue
                        
                        # Create spatial index for the cell
                        cell_spatial_index = SpatialIndex(cell)
                        
                        # Calculate density-weighted centroid
                        samples = min(50, max(20, int(np.sqrt(cell.area / total_area) * 100)))
                        X, Y, density_vals, dx, dy = create_density_grid(
                            cell.bounds, samples, density_fn
                        )
                        
                        grid_points = np.column_stack((X.flatten(), Y.flatten()))
                        mask = cell_spatial_index.filter_points(grid_points).astype(float)
                        mask = mask.reshape(X.shape)
                        
                        x_centroid, y_centroid, area = compute_weighted_centroid(
                            X, Y, density_vals, mask, dx, dy
                        )
                        
                        new_centroids[point_idx] = [x_centroid, y_centroid]
                        areas[point_idx] = area
                    except Exception as e:
                        # Fallback to geometric centroid on error
                        warnings.warn(f"Error in density calculation: {str(e)}. Using geometric centroid.")
                        try:
                            x_centroid, y_centroid, area = compute_polygon_centroid(vx, vy)
                            new_centroids[point_idx] = [x_centroid, y_centroid]
                            areas[point_idx] = area
                        except:
                            # Ultimate fallback to original point
                            new_centroids[point_idx] = points[point_idx]
            
            # Update current total area (for Alpha calculation)
            total_area = np.sum(np.abs(areas))
            
            # Handle case where total_area is too small
            if total_area < 1e-10:
                warnings.warn("Total area is too small. Using original area.")
                total_area = polygon.area
            
            # Compute error using PolyMesher's formula
            # Err = sqrt(sum((A.^2).*sum((Pc-P).*(Pc-P),2)))*NElem/Area^1.5
            diffs = new_centroids - points
            squared_diffs = np.sum(diffs * diffs, axis=1)
            error = np.sqrt(np.sum((areas**2) * squared_diffs)) * n_points / (total_area**1.5)
            
            # Update metrics
            metrics['error_values'].append(error)
            
            if error < metrics['min_error']:
                metrics['min_error'] = error
                metrics['min_error_iteration'] = it
            
            # Update buffer
            point_buffer.append(new_centroids, it + 1, error)
            
            # Print progress if verbose
            if verbose and (it % 10 == 0 or it == max_iterations - 1):
                print(f"Iteration {it+1}, Error: {error:.5e}, Using {len(reflections)} reflections")
            else:
                # Print progress like in original PolyMesher
                print(f"It: {it+1:3d}   Error: {error:.3e}")
            
            # Update points for next iteration
            current_points = new_centroids
            it += 1
        
        # Set convergence status
        if error <= tol:
            metrics['convergence_status'] = 'Converged to tolerance'
        else:
            metrics['convergence_status'] = 'Maximum iterations reached'
        
        # Finalize metrics
        metrics['iterations'] = it
        metrics['time_taken'] = time.time() - start_time
        metrics['final_error'] = error
        
        # Get the best point configuration
        final_points = point_buffer.get_optimal_config()
        if final_points is None:
            final_points = current_points.copy()
        
        # Print minimal info even if not verbose
        print(f"Total iterations: {it}")
        print(f"Total time: {metrics['time_taken']:.5f} seconds")
        print(f"Final error: {error:.5e}")
        
        return final_points, metrics
        
    except Exception as e:
        # Handle unexpected errors
        metrics['convergence_status'] = f'Error: {str(e)}'
        metrics['time_taken'] = time.time() - start_time
        raise

def generate_voronoi_mesh(points, polygon, generate_reflections_fn=None, alpha=None):
    """
    Generate a Voronoi mesh from points within a domain.
    This is a utility function that can be used after Lloyd's algorithm.
    
    Parameters
    ----------
    points : ndarray
        Array of point coordinates
    polygon : shapely.geometry.Polygon
        The domain boundary
    generate_reflections_fn : callable, optional
        Function to generate reflections (default=None)
    alpha : float, optional
        Parameter for reflection generation (default=None)
        
    Returns
    -------
    tuple
        (nodes, elements) where:
        - nodes: Array of node coordinates
        - elements: List of cell vertex indices
    """
    # Generate reflections if function provided
    if generate_reflections_fn is not None and alpha is not None:
        reflections = generate_reflections_fn(points, polygon, alpha)
        if len(reflections) > 0:
            vor = Voronoi(np.vstack([points, reflections]))
        else:
            vor = Voronoi(points)
    else:
        vor = Voronoi(points)
    
    # Extract nodes and elements
    nodes = vor.vertices
    elements = []
    
    for i in range(len(points)):
        region_idx = vor.point_region[i]
        vertices_idx = vor.regions[region_idx]
        
        # Skip unbounded regions
        if -1 in vertices_idx or len(vertices_idx) == 0:
            continue
        
        # Add region vertices to elements
        elements.append(vertices_idx)
    
    return nodes, elements