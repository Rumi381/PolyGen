import os
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from scipy.stats.qmc import Sobol, Halton
import time

class ConstrainedPointGenerator:
    """
    A class to generate constrained point distributions inside a polygonal region using various methods,
    including Poisson Disc Sampling and quasi-random sequences (Sobol and Halton).

    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        A Shapely polygon or multipolygon that defines the region within which the points are generated.
    N : int
        The target number of points to generate.
    seed : int or None, optional
        Seed for the random number generator. Default is None.
    k : int, optional
        Number of attempts to place a new point during Poisson Disc Sampling. Default is 30.
    margin : float, optional
        A small margin to shrink the polygon before generating points. This helps to avoid boundary points.
        Default is 0.01.
    tolerance : float, optional
        Tolerance for the number of generated points. Default is 0.01.
    max_iterations : int, optional
        Maximum number of iterations for radius adjustment during Poisson Disc Sampling. Default is 50.
    workers : int or None, optional
        Number of workers to use for parallel processing. If None, the number of workers is set based on
        the number of points (1 for N < 1000, or the number of CPU cores for N >= 1000). Default is None.
    optimization : str or None, optional
        Optimization scheme for quasi-random sequences (Sobol or Halton). Default is None.

    Attributes
    ----------
    polygon : Polygon or MultiPolygon
        The polygonal region within which points are generated.
    N : int
        The target number of points.
    rng : numpy.random.Generator
        Random number generator used for point generation.
    margin : float
        The margin used to shrink the polygon before point generation.
    tolerance : float
        The tolerance level for the number of points generated.
    max_iterations : int
        The maximum number of iterations for adjusting the radius during Poisson Disc Sampling.
    workers : int
        Number of workers for parallel processing.
    optimization : str or None
        Optimization scheme for quasi-random sequences (if any).

    Methods
    -------
    generate_poisson_points()
        Generate N points using Poisson Disc Sampling inside the polygon.
    generate_sequence_points(use_sobol=False)
        Generate N points using a quasi-random sequence (Sobol or Halton) inside the polygon.
    _generate_points(radius, shrunk_polygon)
        Generate points using Poisson Disc Sampling for a given radius within a shrunk polygon.
    _get_nearby_points(occupied_cells, grid, points, row, col, radius, cell_size)
        Helper method to return nearby points within a given grid for Poisson Disc Sampling.

    Notes
    -----
    Poisson Disc Sampling generates points such that no two points are closer than a specified radius, making it
    useful for applications like blue-noise sampling or spatial point distributions with minimum separation.

    Quasi-random sequences (Sobol and Halton) generate points that cover the space more uniformly than pure random
    sampling, making them suitable for integration and optimization problems.

    Examples
    --------
    Poisson Disc Sampling:
    
    >>> from shapely.geometry import Polygon
    >>> import numpy as np
    >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> N = 100
    >>> generator = ConstrainedPointGenerator(polygon, N)
    >>> points = generator.generate_poisson_points()
    >>> print(points.shape)
    (100, 2)
    
    Generating points using Halton sequence:
    
    >>> points_halton = generator.generate_sequence_points(use_sobol=False)
    >>> print(points_halton.shape)
    (100, 2)
    
    Generating points using Sobol sequence:
    
    >>> points_sobol = generator.generate_sequence_points(use_sobol=True)
    >>> print(points_sobol.shape)
    (128, 2)  # Sobol generates points in powers of two

    Performance Tips
    ----------------
    - For large numbers of points (N >= 1000), use multiple workers for parallel processing. By default, the number
      of workers is automatically set based on your CPU cores when N >= 1000.
    - The Sobol sequence is optimized for uniform space filling but can be slower when generating small numbers
      of points compared to Halton.
    """
    def __init__(self, polygon, N, seed=None, k=30, margin=0.01, tolerance=0.01, max_iterations=50, workers=None, optimization=None):
        self.polygon = polygon
        self.N = N
        self.rng = np.random.default_rng(seed)
        self.k = k
        self.margin = margin
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.seed = seed
        self.workers = workers if workers is not None else (os.cpu_count() if N >= 1000 else 1)
        self.optimization = optimization

    def _generate_points(self, radius, shrunk_polygon):
        """
        Generate points using Poisson Disc Sampling with specified radius.
        
        Parameters
        ----------
        radius : float
            Minimum distance between points
        shrunk_polygon : Polygon
            Polygon shrunk by margin to avoid boundary points
            
        Returns
        -------
        numpy.ndarray
            Array of generated points, shape (M, 2)
        """
        minx, miny, maxx, maxy = shrunk_polygon.bounds
        width, height = maxx - minx, maxy - miny
        cell_size = radius / np.sqrt(2)
        cols, rows = int(np.ceil(width / cell_size)), int(np.ceil(height / cell_size))
        
        # Initialize grid with empty cells (-1 indicates no point)
        grid = np.full((rows, cols), -1, dtype=int)
        points = []
        active = []
        
        # List to track occupied grid cells
        occupied_cells = set()

        # Generate the first point
        while True:
            x, y = self.rng.uniform(minx, maxx), self.rng.uniform(miny, maxy)
            if shrunk_polygon.contains(Point(x, y)):
                points.append((x, y))
                active.append(0)
                col, row = int((x - minx) / cell_size), int((y - miny) / cell_size)
                grid[row, col] = 0
                occupied_cells.add((row, col))
                break

        # Generate more points
        while active:
            idx = self.rng.choice(active)
            found = False
            
            for _ in range(self.k):
                angle = self.rng.uniform(0, 2 * np.pi)
                r = self.rng.uniform(radius, 2 * radius)
                new_x, new_y = points[idx][0] + r * np.cos(angle), points[idx][1] + r * np.sin(angle)
                
                if minx <= new_x < maxx and miny <= new_y < maxy:
                    col, row = int((new_x - minx) / cell_size), int((new_y - miny) / cell_size)
                    
                    if grid[row, col] == -1:
                        new_point = Point(new_x, new_y)
                        if shrunk_polygon.contains(new_point):
                            # Optimize neighborhood search by limiting it to occupied cells
                            nearby_points = self._get_nearby_points(occupied_cells, grid, points, row, col, radius, cell_size)
                            if all((new_x - points[i][0])**2 + (new_y - points[i][1])**2 >= radius**2 for i in nearby_points):
                                points.append((new_x, new_y))
                                grid[row, col] = len(points) - 1
                                active.append(len(points) - 1)
                                occupied_cells.add((row, col))  # Mark the cell as occupied
                                found = True
                                break

            if not found:
                active.remove(idx)

        return np.array(points)

    def _get_nearby_points(self, occupied_cells, grid, points, row, col, radius, cell_size):
        """
        Given the current grid cell (row, col), return the indices of nearby points that
        might be within the search radius.

        Parameters
        ----------
        occupied_cells : set
            A set of occupied cells within the grid.
        grid : numpy.ndarray
            The grid used to track point placements.
        points : numpy.ndarray
            The list of generated points.
        row : int
            Row index of the current cell.
        col : int
            Column index of the current cell.
        radius : float
            The search radius around the current cell.
        cell_size : float
            The size of each cell in the grid.

        Returns
        -------
        nearby_points : list of int
            List of indices of points near the current cell.
        """
        search_radius = int(np.ceil(radius / cell_size))  # Number of cells to search around the current cell
        nearby_points = []

        # Only search occupied cells within the search radius
        for i in range(max(0, row - search_radius), min(grid.shape[0], row + search_radius + 1)):
            for j in range(max(0, col - search_radius), min(grid.shape[1], col + search_radius + 1)):
                if (i, j) in occupied_cells:
                    point_idx = grid[i, j]
                    if point_idx != -1:
                        nearby_points.append(point_idx)
        
        return nearby_points

    def generate_poisson_points(self):
        """
        :no-index:
        Generate N points using Poisson Disc Sampling inside the polygon.

        This method applies binary search over the radius to ensure that the number
        of generated points is as close to N as possible within the given tolerance.

        Returns
        -------
        points : numpy.ndarray, shape (N, 2)
            Array of generated points inside the polygon.

        Raises
        ------
        ValueError
            If the input polygon is invalid or the margin is too large.

        Examples
        --------
        >>> from shapely.geometry import Polygon
        >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> generator = ConstrainedPointGenerator(polygon, 100)
        >>> points = generator.generate_poisson_points()
        >>> print(points.shape)
        (100, 2)

        Performance Tip
        ---------------
        If the number of generated points is far from N, try increasing the number of iterations or reducing
        the margin to improve accuracy.
        """
        if not isinstance(self.polygon, (Polygon, MultiPolygon)):
            raise ValueError("The input must be a Shapely Polygon or MultiPolygon.")
        
        if self.N <= 0:
            raise ValueError("N must be a positive integer.")
        
        shrunk_polygon = self.polygon.buffer(-self.margin)
        
        if shrunk_polygon.is_empty or shrunk_polygon.area <= 0:
            raise ValueError("The margin is too large, the polygon has disappeared or is too small after buffering.")
        
        start_time = time.time()
        initial_radius = np.sqrt(shrunk_polygon.area / (self.N * np.pi)) * 2
        radius_low, radius_high = initial_radius * 0.5, initial_radius * 2
        best_points = None
        best_count = 0
        
        for _ in range(self.max_iterations):
            radius = (radius_low + radius_high) / 2
            points = self._generate_points(radius, shrunk_polygon)
            count = len(points)
            
            if abs(count - self.N) <= self.tolerance * self.N:
                break
            
            if abs(count - self.N) < abs(best_count - self.N):
                best_points = points
                best_count = count
            
            if count < self.N:
                radius_high = radius
            else:
                radius_low = radius
        
        elapsed_time = time.time() - start_time
        print(f"\nPoisson Disc Sampling completed in {elapsed_time:.2f} seconds.")
        print(f"Generated {count} points (target: {self.N}).")
        return points if abs(count - self.N) <= self.tolerance * self.N else best_points

    def generate_sequence_points(self, use_sobol=False):
        """
        :no-index:
        Generate N points using a quasi-random sequence (Sobol or Halton) inside the polygon.

        Parameters
        ----------
        use_sobol : bool, optional
            If True, use the Sobol sequence for point generation. Otherwise, use the Halton sequence.
            Default is False (use Halton).

        Returns
        -------
        points : numpy.ndarray, shape (N, 2)
            Array of generated points inside the polygon.

        Raises
        ------
        ValueError
            If the input polygon is invalid or the margin is too large.

        Examples
        --------
        Generating points using Sobol sequence:
        
        >>> generator = ConstrainedPointGenerator(polygon, 128)
        >>> points_sobol = generator.generate_sequence_points(use_sobol=True)
        >>> print(points_sobol.shape)
        (128, 2)

        Generating points using Halton sequence:
        
        >>> points_halton = generator.generate_sequence_points(use_sobol=False)
        >>> print(points_halton.shape)
        (100, 2)

        Performance Tip
        ---------------
        If using the Sobol sequence, note that it generates points in powers of two. Ensure that N is a power
        of two or slightly adjust the target number of points for better performance.
        """
        if not isinstance(self.polygon, (Polygon, MultiPolygon)):
            raise ValueError("The input must be a Shapely Polygon or MultiPolygon.")
        
        if self.N <= 0:
            raise ValueError("N must be a positive integer.")
        
        shrunk_polygon = self.polygon.buffer(-self.margin)
        
        if shrunk_polygon.is_empty or shrunk_polygon.area <= 0:
            raise ValueError("The margin is too large, the polygon has disappeared or is too small after buffering.")
        
        start_time = time.time()
        
        if use_sobol:
            N = 1 << (self.N - 1).bit_length()  # Next power of 2
            sampler = Sobol(d=2, seed=self.seed, optimization=self.optimization)
        else:
            N = self.N
            sampler = Halton(d=2, seed=self.seed, optimization=self.optimization)
        
        min_x, min_y, max_x, max_y = shrunk_polygon.bounds
        generated_points = []
        
        while len(generated_points) < N:
            batch_size = int(1.2 * (N - len(generated_points)))
            sample = sampler.random(n=batch_size, workers=self.workers)
            
            sample[:, 0] = sample[:, 0] * (max_x - min_x) + min_x
            sample[:, 1] = sample[:, 1] * (max_y - min_y) + min_y
            
            for point in sample:
                if shrunk_polygon.contains(Point(point)):
                    generated_points.append(point)
                if len(generated_points) >= N:
                    break
        
        result = np.array(generated_points[:self.N])
        
        elapsed_time = time.time() - start_time
        print(f"Point generation using quasi-random generator completed in {elapsed_time:.5f} seconds.")
        print(f"Generated {result.shape[0]} points (target: {self.N}).")
        
        return result

# Helper functions to create and use the ConstrainedPointGenerator
def generate_poisson_points(domain, N, seed=None, k=30, margin=0.01, tolerance=0.01, max_iterations=50):
    """
    Wrapper function for ConstrainedPointGenerator.generate_poisson_points().
    :no-index:
    
    For detailed documentation, see :meth:`ConstrainedPointGenerator.generate_poisson_points`.
    
    Parameters
    ----------
    domain : Polygon or MultiPolygon
        A Shapely polygon or multipolygon that defines the boundary.
    N : int
        The target number of points to generate.
    seed : int or None, optional
        Random number generator seed.
    k : int, optional
        Number of point placement attempts.
    margin : float, optional
        Boundary margin.
    tolerance : float, optional
        Point count tolerance.
    max_iterations : int, optional
        Maximum radius adjustment iterations.

    Returns
    -------
    numpy.ndarray
        Array of generated points.
    """
    generator = ConstrainedPointGenerator(domain, N, seed, k, margin, tolerance, max_iterations)
    return generator.generate_poisson_points()

def generate_sequence_points(domain, N, seed=None, margin=0.01, use_sobol=False, workers=None, optimization=None):
    """
    Wrapper function for ConstrainedPointGenerator.generate_sequence_points().
    :no-index:
    
    For detailed documentation, see :meth:`ConstrainedPointGenerator.generate_sequence_points`.
    
    Parameters
    ----------
    domain : Polygon or MultiPolygon
        A Shapely polygon or multipolygon that defines the boundary.
    N : int
        The target number of points to generate.
    seed : int or None, optional
        Random number generator seed.
    margin : float, optional
        Boundary margin.
    use_sobol : bool, optional
        Whether to use Sobol sequence.
    workers : int or None, optional
        Number of parallel workers.
    optimization : str or None, optional
        Sequence optimization scheme.

    Returns
    -------
    numpy.ndarray
        Array of generated points.
    """
    generator = ConstrainedPointGenerator(domain, N, seed, margin=margin, workers=workers, optimization=optimization)
    return generator.generate_sequence_points(use_sobol)
