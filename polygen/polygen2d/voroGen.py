from typing import List, Union
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon

class VoronoiGenerator:
    """
    A class for generating constrained Voronoi tessellations within bounded regions.
    
    This class handles the generation of Voronoi cells within a specified boundary,
    including proper handling of infinite regions and geometric validity checks.
    
    Parameters
    ----------
    buffer_factor : float, optional
        Factor to extend bounding box beyond domain bounds, default=1.0
    
    Notes
    -----
    The Voronoi diagram V(P) for a set of points P = {p₁, ..., pₙ} partitions
    the plane into n cells, where each cell V(pᵢ) consists of all points closer
    to pᵢ than to any other point in P:
    
    .. math::
        V(pᵢ) = {x | ||x - pᵢ|| ≤ ||x - pⱼ|| for all j ≠ i}
    
    This implementation handles:
    - Infinite Voronoi regions using a bounding box
    - Invalid geometries through filtering
    - Proper cell clipping against the domain boundary
    """
    
    def __init__(self, buffer_factor: float = 1.0):
        self.buffer_factor = buffer_factor
    
    def _create_bounding_box(self, domain: Union[Polygon, MultiPolygon]) -> Polygon:
        """
        Create a bounding box for clipping infinite Voronoi regions.
        
        Parameters
        ----------
        domain : Union[Polygon, MultiPolygon]
            The domain to create a bounding box for
            
        Returns
        -------
        Polygon
            Bounding box extending beyond domain bounds
        """
        min_x, min_y, max_x, max_y = domain.bounds
        buffer = self.buffer_factor
        return Polygon([
            (min_x - buffer, min_y - buffer),
            (min_x - buffer, max_y + buffer),
            (max_x + buffer, max_y + buffer),
            (max_x + buffer, min_y - buffer)
        ])
    
    def _process_voronoi_regions(self, 
                               vor: Voronoi, 
                               bbox: Polygon
                               ) -> List[Polygon]:
        """
        Process Voronoi regions including handling of infinite regions.
        
        Parameters
        ----------
        vor : scipy.spatial.Voronoi
            Voronoi tessellation object
        bbox : Polygon
            Bounding box for clipping infinite regions
            
        Returns
        -------
        List[Polygon]
            List of processed Voronoi regions as polygons
            
        Notes
        -----
        Handles infinite regions by:
        1. Computing far points using bisector normals
        2. Ordering vertices counterclockwise
        3. Clipping against bounding box
        """
        new_regions = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)
        radius = np.ptp(vor.points, axis=0).max()

        # Build ridge dictionary
        ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            ridges.setdefault(p1, []).append((p2, v1, v2))
            ridges.setdefault(p2, []).append((p1, v1, v2))

        # Process each region
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]
            
            # Handle finite regions
            if all(v >= 0 for v in vertices):
                new_regions.append(vertices)
                continue

            # Process infinite regions
            ridges_p1 = ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges_p1:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue

                # Compute far point
                t = vor.points[p2] - vor.points[p1]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n))
                far_point = vor.vertices[v2] + direction * n * radius

                # Add new vertex
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # Sort vertices counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            new_regions.append(new_region.tolist())

        # Create and clip polygons
        polygons = []
        for region in new_regions:
            try:
                polygon = Polygon([new_vertices[v] for v in region])
                clipped = polygon.intersection(bbox)
                if clipped.is_valid and not clipped.is_empty:
                    polygons.append(clipped)
            except (ValueError, TypeError):
                continue

        return polygons
    
    def generate_cells(self, 
                      domain: Union[Polygon, MultiPolygon],
                      points: np.ndarray
                      ) -> List[Polygon]:
        """
        Generate constrained Voronoi cells within the given domain.
        
        Parameters
        ----------
        domain : Union[Polygon, MultiPolygon]
            The boundary within which to generate Voronoi cells
        points : np.ndarray
            Array of shape (N, 2) containing generator points
            
        Returns
        -------
        List[Polygon]
            List of Voronoi cells clipped to the domain
            
        Examples
        --------
        >>> from shapely.geometry import Polygon
        >>> import numpy as np
        >>> domain = Polygon([(0,0), (1,0), (1,1), (0,1)])
        >>> points = np.random.rand(10, 2)
        >>> generator = VoronoiGenerator()
        >>> cells = generator.generate_cells(domain, points)
        
        Notes
        -----
        The process involves:
        1. Computing the Voronoi tessellation
        2. Handling infinite regions with a bounding box
        3. Clipping cells to the domain boundary
        4. Filtering invalid geometries
        
        For numerical stability, points should be reasonably distributed
        within the domain bounds.
        """
        # Validate input
        if not isinstance(domain, (Polygon, MultiPolygon)):
            raise ValueError("Domain must be a Shapely Polygon or MultiPolygon")
        
        if not isinstance(points, np.ndarray) or points.shape[1] != 2:
            raise ValueError("Points must be a numpy array of shape (N, 2)")
            
        # Generate Voronoi tessellation
        vor = Voronoi(points)
        
        # Create bounding box and process regions
        bbox = self._create_bounding_box(domain)
        cells = self._process_voronoi_regions(vor, bbox)
        
        # Clip to domain and filter invalid cells
        clipped_cells = []
        for cell in cells:
            try:
                clipped = cell.intersection(domain)
                if clipped.is_valid and not clipped.is_empty:
                    clipped_cells.append(clipped)
            except (ValueError, TypeError):
                continue
                
        return clipped_cells
