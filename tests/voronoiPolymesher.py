from typing import List, Tuple, Dict, Union
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, Point

class PolyMesherVoronoiGenerator:
    """
    A class for generating PolyMesher-style Voronoi tessellations within bounded regions.
    
    This class handles the generation of Voronoi cells using the reflection method
    from PolyMesher instead of clipping against the domain boundary.
    
    Parameters
    ----------
    reflection_factor : float, optional
        Factor controlling the width of the reflection band, default=1.5
    
    Notes
    -----
    Unlike the original VoronoiGenerator, this implementation:
    - Uses point reflections instead of clipping
    - Returns only cells corresponding to the original seed points
    - Follows the PolyMesher methodology for boundary handling
    """
    
    def __init__(self, reflection_factor: float = 1.5):
        self.reflection_factor = reflection_factor
    
    def _compute_distance_components(self, 
                                    points: np.ndarray, 
                                    polygon: Union[Polygon, MultiPolygon]
                                    ) -> np.ndarray:
        """
        Compute signed distance components to each segment of the polygon boundary.
        
        Parameters
        ----------
        points : ndarray
            Array of point coordinates
        polygon : Union[Polygon, MultiPolygon]
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
        d[:, -1] = np.max(d[:, :-1], axis=1)
        
        return d
    
    def _generate_reflections(self, 
                             points: np.ndarray, 
                             polygon: Union[Polygon, MultiPolygon],
                             alpha: float
                             ) -> np.ndarray:
        """
        Generate reflections of points near the boundary of the domain.
        
        Parameters
        ----------
        points : ndarray
            Array of point coordinates
        polygon : Union[Polygon, MultiPolygon]
            The domain boundary
        alpha : float
            Parameter controlling how close to boundary points need to be for reflection
        
        Returns
        -------
        ndarray
            Array of reflection points
        """
        n_points = len(points)
        eps = 1e-8  # Small value for numerical differentiation
        eta = 0.9   # Threshold for reflection acceptance (from PolyMesher)
        
        # Get distance components
        d = self._compute_distance_components(points, polygon)
        n_bdry_segs = d.shape[1] - 1
        
        # Compute normal vectors via numerical differentiation
        points_x_shift = points.copy()
        points_x_shift[:, 0] += eps
        d_x_shift = self._compute_distance_components(points_x_shift, polygon)
        n1 = (d_x_shift - d) / eps
        
        points_y_shift = points.copy()
        points_y_shift[:, 1] += eps
        d_y_shift = self._compute_distance_components(points_y_shift, polygon)
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
                    reflection_x = points[i, 0] - 2 * n1[i, j] * d[i, j]
                    reflection_y = points[i, 1] - 2 * n2[i, j] * d[i, j]
                    reflections.append((reflection_x, reflection_y))
                    original_distances.append(np.abs(d[i, j]))
        
        if not reflections:
            return np.zeros((0, 2))
        
        # Convert to numpy array
        reflections = np.array(reflections)
        original_distances = np.array(original_distances)
        
        # Check validity
        d_reflections = self._compute_distance_components(reflections, polygon)
        
        # Apply validity criteria from PolyMesher
        valid = (d_reflections[:, -1] > 0) & (np.abs(d_reflections[:, -1]) >= eta * original_distances)
        
        if not np.any(valid):
            return np.zeros((0, 2))
        
        # Return valid reflections
        valid_reflections = reflections[valid]
        
        # Remove duplicates
        unique_reflections = np.unique(valid_reflections, axis=0)
        
        return unique_reflections
    
    def _create_voronoi_polygons(self, vor: Voronoi, n_original: int) -> List[Polygon]:
        """
        Create Shapely polygons from Voronoi regions.
        
        Parameters
        ----------
        vor : scipy.spatial.Voronoi
            Voronoi tessellation object
        n_original : int
            Number of original seed points (excludes reflections)
            
        Returns
        -------
        List[Polygon]
            List of Voronoi cells as polygons, only for original points
        """
        polygons = []
        
        # Process only regions for original points
        for p_idx in range(n_original):
            region_idx = vor.point_region[p_idx]
            vertices_idx = vor.regions[region_idx]
            
            # Skip unbounded regions
            if -1 in vertices_idx or len(vertices_idx) == 0:
                continue
            
            # Extract vertices for this region
            vertices = [vor.vertices[i] for i in vertices_idx]
            
            try:
                # Create polygon
                polygon = Polygon(vertices)
                
                # Validate and fix if necessary
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                
                if not polygon.is_empty:
                    polygons.append(polygon)
            except (ValueError, TypeError):
                continue
        
        return polygons
    
    def generate_cells(self, 
                      domain: Union[Polygon, MultiPolygon],
                      points: np.ndarray
                      ) -> List[Polygon]:
        """
        Generate Voronoi cells using the PolyMesher reflection approach.
        
        Parameters
        ----------
        domain : Union[Polygon, MultiPolygon]
            The boundary within which to generate Voronoi cells
        points : np.ndarray
            Array of shape (N, 2) containing generator points
            
        Returns
        -------
        List[Polygon]
            List of Voronoi cells as Shapely Polygons
            
        Notes
        -----
        The process follows PolyMesher's approach:
        1. Calculate the reflection distance parameter (alpha)
        2. Generate reflections of points near the boundary
        3. Compute the Voronoi diagram with original points and reflections
        4. Extract only the cells corresponding to original points
        """
        # Validate input
        if not isinstance(domain, (Polygon, MultiPolygon)):
            raise ValueError("Domain must be a Shapely Polygon or MultiPolygon")
        
        if not isinstance(points, np.ndarray) or points.shape[1] != 2:
            raise ValueError("Points must be a numpy array of shape (N, 2)")
        
        # Number of original points
        n_points = len(points)
        
        # Calculate Alpha (reflection distance parameter) based on PolyMesher approach
        total_area = domain.area
        alpha = self.reflection_factor * np.sqrt(total_area / n_points)
        
        # Generate reflections of points near the boundary
        reflections = self._generate_reflections(points, domain, alpha)
        
        # Compute Voronoi diagram with original points and reflections
        if len(reflections) > 0:
            vor = Voronoi(np.vstack([points, reflections]))
        else:
            vor = Voronoi(points)
        
        # Extract Voronoi cells as polygons
        polygons = self._create_voronoi_polygons(vor, n_points)
        
        return polygons