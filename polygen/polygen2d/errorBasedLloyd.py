import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial import Voronoi
import time
import warnings
from shapely import prepared
from collections import deque

class PointHistoryBuffer:
    """
    Memory-efficient circular buffer for storing point configurations during Lloyd's algorithm.
    
    This class implements a circular buffer to store the history of point configurations
    and track the optimal configuration based on gradient norm values. It uses a deque
    data structure to maintain memory efficiency when storing multiple configurations.
    
    Parameters
    ----------
    max_size : int
        Maximum number of point configurations to store in the buffer
    point_shape : tuple
        Shape of each point configuration array (e.g., (n_points, 2) for 2D points)
    dtype : numpy.dtype, optional
        Data type for storing point coordinates, default is np.float64
    
    Attributes
    ----------
    buffer : collections.deque
        Circular buffer containing point configurations
    optimal_config : ndarray
        Point configuration with the lowest gradient norm
    optimal_norm : float
        Lowest gradient norm value encountered
    optimal_iteration : int
        Iteration number where the optimal configuration was found
    
    Notes
    -----
    The buffer automatically removes oldest configurations when max_size is reached.
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
        """
        Add a new point configuration to the buffer.
        
        Parameters
        ----------
        points : ndarray
            Array of point coordinates to store
        iteration : int
            Current iteration number
        grad_norm : float
            Gradient norm for the current configuration
        """
        self.buffer.append(points.copy())
        
        # Update optimal configuration if necessary
        if grad_norm < self.optimal_norm:
            self.optimal_norm = grad_norm
            self.optimal_config = points.copy()
            self.optimal_iteration = iteration
    
    def get_config(self, index):
        """
        Retrieve a specific point configuration from the buffer.
        
        Parameters
        ----------
        index : int
            Index of the configuration to retrieve
        
        Returns
        -------
        ndarray or None
            Point configuration at the specified index, or None if index is invalid
        """
        if 0 <= index < len(self.buffer):
            return self.buffer[index]
        return None
    
    def get_optimal_config(self):
        """
        Retrieve the configuration with the lowest gradient norm.
        
        Returns
        -------
        ndarray or None
            Optimal point configuration, or None if no configurations stored
        """
        return self.optimal_config.copy() if self.optimal_config is not None else None
    
    def clear(self):
        """
        Clear all configurations from the buffer.
        """
        self.buffer.clear()
        
    def __len__(self):
        return len(self.buffer)

def compute_weighted_centroid(X, Y, density_vals, mask, dx, dy):
    """
    Compute the density-weighted centroid of a region using numerical integration.
    
    This function calculates the centroid of a region weighted by a density function
    using discrete numerical integration over a grid.
    
    Parameters
    ----------
    X, Y : ndarray
        Meshgrid arrays of x and y coordinates
    density_vals : ndarray
        Array of density values at each grid point
    mask : ndarray
        Boolean mask indicating points inside the region
    dx, dy : float
        Grid spacing in x and y directions
    
    Returns
    -------
    tuple
        (x_centroid, y_centroid) coordinates of the weighted centroid
    """
    masked_density = density_vals * mask
    total_mass = np.sum(masked_density) * dx * dy
    
    if total_mass < 1e-10:
        # Return geometric centroid based on the masked area
        x_centroid = np.sum(X * mask) / np.sum(mask)
        y_centroid = np.sum(Y * mask) / np.sum(mask)
        return x_centroid, y_centroid
        
    x_centroid = np.sum(X * masked_density) * dx * dy / total_mass
    y_centroid = np.sum(Y * masked_density) * dx * dy / total_mass
    
    return x_centroid, y_centroid

def create_density_grid(bounds, samples, density_fn):
    """
    Create a discretized grid and compute density values for numerical integration.
    
    Parameters
    ----------
    bounds : tuple
        (x_min, y_min, x_max, y_max) defining the bounding box
    samples : int
        Number of sample points in each dimension
    density_fn : callable
        Function that takes (x, y) arrays and returns density values
    
    Returns
    -------
    X : ndarray
        2D array of x-coordinates
    Y : ndarray
        2D array of y-coordinates
    density_vals : ndarray
        2D array of density values at grid points
    dx : float
        Grid spacing in x-direction
    dy : float
        Grid spacing in y-direction
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

def compute_adaptive_samples(polygon_region, reference_area, min_samples=20, max_samples=100):
    """
    Determine appropriate grid resolution based on region size.
    
    Adaptively computes the number of grid samples needed for accurate
    numerical integration based on the relative size of the region.
    
    Parameters
    ----------
    polygon_region : shapely.geometry.Polygon
        Region to compute samples for
    reference_area : float
        Area used to determine relative size for sampling
        For individual cells, this should be the expected cell area
    min_samples, max_samples : int
        Bounds for number of samples
    
    Returns
    -------
    int
        Number of samples to use for the grid
    
    Notes
    -----
    The number of samples scales linearly with the relative area of the region.
    """
    relative_area = polygon_region.area / reference_area
    samples = int(min_samples + (max_samples - min_samples) * relative_area)
    return np.clip(samples, min_samples, max_samples)

def calculate_density_weighted_centroid(polygon_region, density_fn, is_uniform_density, total_area):
    """
    Calculate the density-weighted centroid of a polygon region.
    
    This function computes the centroid of a polygon region weighted by a density
    function using adaptive numerical integration.
    
    Parameters
    ----------
    polygon_region : shapely.geometry.Polygon
        Region to calculate centroid for
    density_fn : callable
        Function that takes (x, y) arrays and returns density values
    is_uniform_density : bool
        If True, uses geometric centroid instead of weighted centroid
    total_area : float
        Total area of the boundary polygon
    
    Returns
    -------
    tuple
        (x, y) coordinates of the weighted centroid
    
    Notes
    -----
    For uniform density, returns the geometric centroid.
    Uses adaptive grid resolution based on region size for numerical integration.
    """
    if is_uniform_density:
        return polygon_region.centroid.coords[0]
    
    # Use adaptive grid resolution
    samples = compute_adaptive_samples(polygon_region, total_area)
    
    # Create density grid
    X, Y, density_vals, dx, dy = create_density_grid(polygon_region.bounds, samples, density_fn)
    
    # Prepare points for containment test
    points = np.column_stack((X.flatten(), Y.flatten()))
    
    # Vectorized point-in-polygon test
    prepared_region = prepared.prep(polygon_region)
    mask = np.array([prepared_region.contains(Point(p)) for p in points], dtype=float)
    mask = mask.reshape(X.shape)
    
    # Compute weighted centroid
    x_centroid, y_centroid = compute_weighted_centroid(X, Y, density_vals, mask, dx, dy)
    
    if x_centroid == 0.0 and y_centroid == 0.0:
        return polygon_region.centroid.coords[0]
        
    return (x_centroid, y_centroid)

def calculate_energy(points, voronoi, polygon, density_fn, is_uniform_density):
    """
    Calculate Lloyd's energy for the current point configuration.
    
    Computes the total energy of the configuration defined as the sum of second
    moments of each Voronoi cell, weighted by the density function.
    
    Parameters
    ----------
    points : ndarray
        Array of generator points, shape (n_points, 2)
    voronoi : scipy.spatial.Voronoi
        Voronoi diagram for current points
    polygon : shapely.geometry.Polygon
        Boundary polygon
    density_fn : callable
        Function that takes (x, y) arrays and returns density values
    is_uniform_density : bool
        If True, uses uniform density weighting
    
    Returns
    -------
    float
        Total energy of the configuration
    """
    energy = 0.0
    for point_idx, region_idx in enumerate(voronoi.point_region):
        vertices = voronoi.regions[region_idx]
        if not all(v >= 0 for v in vertices):
            continue
            
        # Create Voronoi cell polygon
        region = [voronoi.vertices[i] for i in vertices]
        cell = Polygon(region)
        if not cell.is_valid:
            cell = cell.buffer(0)
            
        # Clip with boundary
        cell = cell.intersection(polygon)
        if cell.is_empty:
            continue
            
        # Calculate second moment relative to generator point
        bounds = cell.bounds
        x_min, y_min, x_max, y_max = bounds
        
        # Use numerical integration for energy calculation
        x = np.linspace(x_min, x_max, 20)
        y = np.linspace(y_min, y_max, 20)
        X, Y = np.meshgrid(x, y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Points for containment test
        points_grid = np.column_stack((X.flatten(), Y.flatten()))
        mask = np.array([cell.contains(Point(p)) for p in points_grid]).reshape(X.shape)
        
        # Calculate squared distances
        dx_squared = (X - points[point_idx, 0])**2
        dy_squared = (Y - points[point_idx, 1])**2
        squared_distances = dx_squared + dy_squared
        
        if is_uniform_density:
            energy += np.sum(squared_distances * mask) * dx * dy
        else:
            density_vals = density_fn(X, Y)
            energy += np.sum(squared_distances * mask * density_vals) * dx * dy
            
    return energy

def compute_error_measure(polygon, points, centroids, cell_masses, density_fn, is_uniform_density):
    """
    Compute the non-dimensionalized error measure with a simple cached density integral.
    
    This function uses a single cached value for the density integral, which is computed
    only once for a given domain and density function combination. The cached value is
    stored as a function attribute using Python's ability to attach attributes to functions.
    """
    n_points = len(points)
    total_area = polygon.area
    
    # Compute gradient norm using masses
    point_diffs = points - centroids
    squared_diffs = np.sum(point_diffs * point_diffs, axis=1)
    gradient_norm = np.sqrt(np.sum(cell_masses * cell_masses * squared_diffs))
    
    # Handle density integral calculation
    if is_uniform_density:
        density_integral = total_area
    else:
        # Check if we have already computed the density integral
        if not hasattr(compute_error_measure, '_cached_integral'):
            # If this is our first time, compute the integral
            samples = compute_adaptive_samples(polygon, total_area, 
                                            min_samples=50, max_samples=100)
            
            # Create density grid
            X, Y, density_vals, dx, dy = create_density_grid(
                polygon.bounds, samples, density_fn
            )
            
            # Create domain mask
            points = np.column_stack((X.flatten(), Y.flatten()))
            prepared_domain = prepared.prep(polygon)
            mask = np.array([prepared_domain.contains(Point(p)) 
                           for p in points], dtype=float)
            mask = mask.reshape(X.shape)
            
            # Compute and cache the integral value
            masked_density = density_vals * mask
            compute_error_measure._cached_integral = np.sum(masked_density) * dx * dy
        
        density_integral = compute_error_measure._cached_integral
    
    # Compute non-dimensionalized error
    error = (n_points * gradient_norm) / (np.sqrt(total_area) * density_integral)
    
    return error

def lloyd_with_density(polygon, seed_points, density_function=None, max_iterations=10000000, 
                      tol=1e-5, use_decay=True, decay_start=2.0, decay_mechanism="exponential",
                      grad_increase_tol=1.2, history_buffer_size=10):
    """
    This function provides a structured interface to run Lloyd's algorithm (centroidal
    Voronoi tessellation) on a given polygon domain, using either uniform or
    non-uniform density functions for weighting.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The domain in which to compute the centroidal Voronoi tessellation.
    seed_points : array_like
        Initial seed points (n_points x 2) inside the polygon.
    density_function : callable, optional
        A function f(x, y) -> float that defines pointwise density. If None,
        a uniform density of 1 is assumed.
        Example density function:
        
        .. code-block:: python
        
            def hexagonDensity(x, y):
                return np.exp(-20 * (x**2 + y**2)) + 0.05 * np.sin(np.pi * x)**2 * np.sin(np.pi * y)**2
                
    max_iterations : int, optional
        Maximum number of Lloyd iterations (default=1e7).
    tol : float, optional
        Convergence tolerance for the error measure (default=1e-5).
    use_decay : bool, optional
        Whether to apply a decay factor to point movements (default=True).
    decay_start : float, optional
        Starting value for decay (default=2.0). Decay moves from decay_start -> 1.
    decay_mechanism : {'exponential', 'linear'}, optional
        Decay mode for point adjustments (default='exponential').
    grad_increase_tol : float, optional
        Factor of tolerance for detecting error growth (default=1.2).
    history_buffer_size : int, optional
        Maximum size of the point configuration buffer (default=10).

    Returns
    -------
    tuple
        A tuple (points, metrics) where:
        
        - points: The final point configuration as a numpy array
        - metrics: A dictionary containing convergence information and error history

    Notes
    -----
    1. The algorithm uses a numerical integration approach if a non-uniform density
       is provided. This can be slower but gives more accurate centroid locations.
    2. The algorithm stops if:
       - The error measure is below `tol`.
       - The error measure grows substantially beyond its minimum observed value.
       - The maximum iteration limit is reached.
    """
    # Clear any cached integral value at the start
    if hasattr(compute_error_measure, '_cached_integral'):
        delattr(compute_error_measure, '_cached_integral')
    
    # Initial setup remains the same
    is_uniform_density = density_function is None
    density_fn = (lambda x, y: np.ones_like(x)) if is_uniform_density else density_function
    
    seed_points = np.array(seed_points)
    total_area = polygon.area
    # Calculate expected average cell area
    # expected_cell_area = total_area / len(seed_points)

    # Initialize buffer and metrics
    point_buffer = PointHistoryBuffer(
        max_size=history_buffer_size,
        point_shape=seed_points.shape,
        dtype=seed_points.dtype
    )
    
    metrics = {
        'error_values': [],
        'convergence_status': None,
        'iterations': 0,
        'time_taken': 0,
        'min_error': float('inf'),
        'min_error_iteration': 0,
    }

    point_buffer.append(seed_points, 0, float('inf'))
    prepared_polygon = prepared.prep(polygon)
    
    # Setup decay
    if use_decay:
        if decay_mechanism.lower() == "linear":
            decay_values = decay_start + (1 - decay_start) * (np.arange(max_iterations) / (max_iterations - 1))
        else:
            decay_values = np.exp(-np.linspace(0, max_iterations, max_iterations) / 
                                max_iterations * np.log(0.1)) * (decay_start - 1) + 1
    
    start_time = time.time()
    
    try:
        for iteration in range(max_iterations):
            vor = Voronoi(seed_points)
            new_points = np.empty_like(seed_points)
            centroids = np.empty_like(seed_points)
            cell_masses = np.zeros(len(seed_points))
            
            # Process Voronoi cells and collect necessary information
            for point_idx, region_idx in enumerate(vor.point_region):
                vertices = vor.regions[region_idx]
                if not all(v >= 0 for v in vertices):
                    new_points[point_idx] = seed_points[point_idx]
                    centroids[point_idx] = seed_points[point_idx]
                    continue
                
                # Create and validate Voronoi cell
                region = [vor.vertices[i] for i in vertices]
                polygon_voronoi = Polygon(region)
                
                if not polygon_voronoi.is_valid:
                    polygon_voronoi = polygon_voronoi.buffer(0)
                
                clipped_polygon = polygon_voronoi.intersection(polygon)
                if clipped_polygon.is_empty:
                    new_points[point_idx] = seed_points[point_idx]
                    centroids[point_idx] = seed_points[point_idx]
                    continue
                
                try:
                    if is_uniform_density:
                        # For uniform density, use geometric centroid and area
                        centroids[point_idx] = clipped_polygon.centroid.coords[0]
                        cell_masses[point_idx] = clipped_polygon.area
                    else:
                        # Calculate weighted centroid for non-uniform density
                        # Calculate centroid and collect cell mass
                        samples = compute_adaptive_samples(clipped_polygon, total_area)
                        X, Y, density_vals, dx, dy = create_density_grid(
                            clipped_polygon.bounds, samples, density_fn
                        )
                        
                        points = np.column_stack((X.flatten(), Y.flatten()))
                        prepared_cell = prepared.prep(clipped_polygon)
                        mask = np.array([prepared_cell.contains(Point(p)) for p in points], dtype=float)
                        mask = mask.reshape(X.shape)
                        
                        masked_density = density_vals * mask
                        cell_masses[point_idx] = np.sum(masked_density) * dx * dy
                        
                        centroid = compute_weighted_centroid(X, Y, density_vals, mask, dx, dy)
                        centroids[point_idx] = centroid
                except Exception as e:
                    warnings.warn(f"Error in cell calculations: {str(e)}. Using geometric centroid.")
                    centroids[point_idx] = clipped_polygon.centroid.coords[0]
                    cell_masses[point_idx] = clipped_polygon.area
                
                # Apply movement with decay
                if prepared_polygon.contains(Point(centroids[point_idx])):
                    if use_decay:
                        decay = decay_values[iteration]
                        new_points[point_idx] = seed_points[point_idx] + \
                            (centroids[point_idx] - seed_points[point_idx]) * decay
                    else:
                        new_points[point_idx] = centroids[point_idx]
                else:
                    new_points[point_idx] = seed_points[point_idx]
            
            # Compute error measure
            error = compute_error_measure(
                polygon, seed_points, centroids, cell_masses,
                density_fn, is_uniform_density
            )
            
            metrics['error_values'].append(error)
            # Update history
            point_buffer.append(new_points, iteration + 1, error)
            
            if error < metrics['min_error']:
                metrics['min_error'] = error
                metrics['min_error_iteration'] = iteration
            
            # Check convergence
            if error < tol:
                metrics['convergence_status'] = 'Converged to tolerance'
                metrics['iterations'] = iteration + 1
                metrics['time_taken'] = time.time() - start_time
                print(f"Converged after {iteration+1} iterations with error: {error:.5e}")
                print(f"Total time: {metrics['time_taken']:.5f} seconds")
                return point_buffer.get_optimal_config(), metrics
            
            # Check for divergence
            elif (error > grad_increase_tol * metrics['min_error'] and 
                  iteration > metrics['min_error_iteration'] + 5):
                metrics['convergence_status'] = 'Error increased significantly'
                metrics['iterations'] = iteration + 1
                metrics['time_taken'] = time.time() - start_time
                print(f"Minimum error {metrics['min_error']:.5e} found at iteration {metrics['min_error_iteration']+1}")
                print(f"Total time: {metrics['time_taken']:.5f} seconds")
                return point_buffer.get_optimal_config(), metrics
            
            seed_points = new_points
            
        # Handle maximum iterations case
        metrics['convergence_status'] = 'Maximum iterations reached'
        metrics['iterations'] = max_iterations
        metrics['time_taken'] = time.time() - start_time
        print(f"Reached maximum iterations ({max_iterations}) with norm: {error:.5e}")
        print(f"Total time: {metrics['time_taken']:.5f} seconds")
        return point_buffer.get_optimal_config(), metrics
        
    except Exception as e:
        metrics['convergence_status'] = f'Error: {str(e)}'
        metrics['time_taken'] = time.time() - start_time
        raise
