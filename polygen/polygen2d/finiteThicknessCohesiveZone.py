from typing import List, Tuple, Dict, Set
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
# import numpy as np
import copy
# from scipy.spatial import Delaunay
from shapely import buffer

class CohesiveZoneAdjuster:
    """
    An improved class for adjusting Voronoi cells to introduce finite-thickness cohesive zones.
    
    This implementation addresses two key issues:
    1. Preserves the original boundary of the domain
    2. Creates uniform thickness gaps between adjacent cells
    
    Parameters
    ----------
    tolerance : float, optional
        Convergence tolerance for area ratio (default: 0.005)
    max_iterations : int, optional
        Maximum number of adjustment iterations (default: 10)
    verbose : bool, optional
        Whether to print progress information (default: False)
        
    Attributes
    ----------
    history : List[Tuple[float, float]]
        History of (thickness, ratio) pairs from iterations
    """
    
    def __init__(
        self,
        tolerance: float = 0.005,
        max_iterations: int = 10,
        verbose: bool = False
    ):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.history: List[Tuple[float, float]] = []
    
    def _find_boundary_cells(self, cells: List[Polygon], domain_boundary: Polygon) -> Set[int]:
        """
        Identify cells that are on the boundary of the domain.
        
        Parameters
        ----------
        cells : List[Polygon]
            Input Voronoi cells
        domain_boundary : Polygon
            The boundary polygon of the entire domain
            
        Returns
        -------
        Set[int]
            Indices of cells that are on the boundary
        """
        boundary_edges = domain_boundary.boundary
        boundary_cell_indices = set()
        
        for i, cell in enumerate(cells):
            if cell.boundary.intersects(boundary_edges):
                boundary_cell_indices.add(i)
                
        return boundary_cell_indices
    
    # def _find_adjacent_cells(self, cells: List[Polygon]) -> Dict[int, Set[int]]:
    #     """
    #     Find adjacency using Delaunay triangulation of cell centroids.
        
    #     Parameters
    #     ----------
    #     cells : List[Polygon]
    #         Input Voronoi cells
            
    #     Returns
    #     -------
    #     Dict[int, Set[int]]
    #         Dictionary mapping cell indices to sets of adjacent cell indices
    #     """
    #     # Extract centroids
    #     centroids = np.array([(cell.centroid.x, cell.centroid.y) for cell in cells])
        
    #     # Create Delaunay triangulation
    #     tri = Delaunay(centroids)
        
    #     # Initialize adjacency dictionary
    #     adjacency = {i: set() for i in range(len(cells))}
        
    #     # Process each triangle to establish adjacency
    #     for simplex in tri.simplices:
    #         # Each simplex (triangle) defines three adjacency relationships
    #         i, j, k = simplex
    #         adjacency[i].add(j)
    #         adjacency[i].add(k)
    #         adjacency[j].add(i)
    #         adjacency[j].add(k)
    #         adjacency[k].add(i)
    #         adjacency[k].add(j)
        
    #     return adjacency
    
    # def _compute_edge_bisectors(
    #     self, 
    #     cells: List[Polygon],
    #     distance: float
    # ) -> Dict[int, Polygon]:
    #     """
    #     Compute the inward offset for each cell based on a uniform distance.
        
    #     Parameters
    #     ----------
    #     cells : List[Polygon]
    #         Input Voronoi cells
    #     distance : float
    #         Half of the desired gap thickness
            
    #     Returns
    #     -------
    #     Dict[int, Polygon]
    #         Dictionary mapping cell indices to their offset polygons
    #     """
    #     offset_cells = {}
        
    #     for i, cell in enumerate(cells):
    #         # Compute the negative buffer (inward offset)
    #         # This creates a uniform inward offset regardless of cell shape
    #         offset = cell.buffer(-distance, join_style=2, mitre_limit=5.0)
            
    #         # Handle potential geometric errors or empty offsets
    #         if offset.is_empty or not offset.is_valid:
    #             # For very small cells that would disappear, keep a minimal cell
    #             centroid = cell.centroid
    #             offset = Point(centroid).buffer(distance/4)
            
    #         offset_cells[i] = offset
                
    #     return offset_cells
    
    def _compute_edge_bisectors(
        self, 
        cells: List[Polygon],
        distance: float
    ) -> Dict[int, Polygon]:
        """
        Compute the inward offset for each cell using vectorized buffer operations.
        
        Parameters
        ----------
        cells : List[Polygon]
            Input Voronoi cells
        distance : float
            Half of the desired gap thickness
            
        Returns
        -------
        Dict[int, Polygon]
            Dictionary mapping cell indices to their offset polygons
        """
        
        # Perform vectorized buffer operation on all cells at once
        offset_geoms = buffer(cells, -distance, join_style=2, mitre_limit=5.0)
        
        # Create dictionary of offset cells
        offset_cells = {}
        
        # Process the results
        for i, offset in enumerate(offset_geoms):
            # Handle potential geometric errors or empty offsets
            if offset is None or offset.is_empty or not offset.is_valid:
                # For very small cells that would disappear, keep a minimal cell
                centroid = cells[i].centroid
                offset = Point(centroid).buffer(distance/4)
            
            offset_cells[i] = offset
                
        return offset_cells
    
    def _adjust_boundary_cells(
        self,
        cells: List[Polygon],
        offset_cells: Dict[int, Polygon],
        boundary_indices: Set[int],
        domain_boundary: Polygon,
        distance: float
    ) -> Dict[int, Polygon]:
        """
        Adjust boundary cells to preserve the original domain boundary.
        
        Parameters
        ----------
        cells : List[Polygon]
            Original Voronoi cells
        offset_cells : Dict[int, Polygon]
            Inward offset cells
        boundary_indices : Set[int]
            Indices of cells that are on the boundary
        domain_boundary : Polygon
            The boundary polygon of the entire domain
        distance : float
            Half of the desired gap thickness
            
        Returns
        -------
        Dict[int, Polygon]
            Adjusted offset cells that preserve the domain boundary
        """
        adjusted_offset_cells = copy.deepcopy(offset_cells)
        
        for i in boundary_indices:
            # For boundary cells, we need special handling
            original_cell = cells[i]
            
            # Calculate the intersection of the cell with the domain boundary
            boundary_segment = original_cell.intersection(domain_boundary.boundary)
            
            # If this cell has a portion on the domain boundary
            if not boundary_segment.is_empty:
                # Create an inward buffer only for the boundary segment
                inward_boundary = None
                
                # Handle different geometry types
                if isinstance(boundary_segment, LineString):
                    # Create an inward offset from the boundary line
                    inward_boundary = boundary_segment.buffer(distance, single_sided=True)
                else:
                    # For more complex boundary segments, use a different approach
                    # This handles MultiLineString and other geometry types
                    inward_boundary = boundary_segment.buffer(distance)
                
                # Intersect with the original cell to ensure we stay within it
                inward_boundary = inward_boundary.intersection(original_cell)
                
                # The adjusted cell is the difference between original offset and
                # the inward boundary buffer
                if inward_boundary.is_valid and not inward_boundary.is_empty:
                    if i in adjusted_offset_cells and adjusted_offset_cells[i].is_valid:
                        # Remove the inward boundary from the offset cell
                        adjusted_offset_cells[i] = adjusted_offset_cells[i].difference(inward_boundary)
                        
                        # Ensure the result is valid
                        if not adjusted_offset_cells[i].is_valid or adjusted_offset_cells[i].is_empty:
                            # Fallback: use a simpler approach
                            adjusted_offset_cells[i] = original_cell.buffer(-distance, join_style=2)
        
        return adjusted_offset_cells
    
    def _calculate_area_ratio(
        self,
        original_cells: List[Polygon],
        adjusted_cells: List[Polygon]
    ) -> float:
        """
        Calculate the area ratio between adjusted and original cells.
        
        Parameters
        ----------
        original_cells : List[Polygon]
            Original Voronoi cells
        adjusted_cells : List[Polygon]
            Adjusted Voronoi cells
            
        Returns
        -------
        float
            Area ratio (adjusted area / original area)
        """
        original_area = sum(cell.area for cell in original_cells)
        adjusted_area = sum(cell.area for cell in adjusted_cells)
        
        return adjusted_area / original_area
    
    def _interpolate_thickness(
        self,
        target_ratio: float
    ) -> float:
        """
        Interpolate thickness using Lagrange polynomials.
        
        Parameters
        ----------
        target_ratio : float
            Target area ratio to achieve
            
        Returns
        -------
        float
            Interpolated thickness value
        """
        if len(self.history) < 2:
            # Linear scaling for single point
            thickness, ratio = self.history[0]
            return thickness * target_ratio / ratio
            
        elif len(self.history) == 2:
            # Linear interpolation
            (t1, r1), (t2, r2) = self.history[-2:]
            return t1 + (target_ratio - r1) * (t2 - t1) / (r2 - r1)
            
        else:
            # Quadratic interpolation using last 3 points
            points = self.history[-3:]
            thicknesses, ratios = zip(*points)
            
            # Lagrange basis polynomials
            L = []
            for i in range(3):
                product = 1.0
                for j in range(3):
                    if i != j:
                        product *= ((target_ratio - ratios[j]) / 
                                  (ratios[i] - ratios[j]))
                L.append(product)
                
            return sum(L[i] * thicknesses[i] for i in range(3))
    
    def adjust_fixed_thickness(
        self,
        cells: List[Polygon],
        thickness: float,
        preserve_boundary: bool = True
    ) -> List[Polygon]:
        """
        Adjust cells using a fixed thickness value.
        
        Parameters
        ----------
        cells : List[Polygon]
            Input Voronoi cells
        thickness : float
            Fixed gap thickness to apply
        preserve_boundary : bool, optional
            Whether to preserve the original domain boundary (default: True)
            
        Returns
        -------
        List[Polygon]
            Adjusted Voronoi cells
        """
        # Calculate the domain boundary before adjustment
        domain_boundary = unary_union(cells)
        
        # Half of the thickness for each side of the gap
        half_thickness = thickness / 2.0
        
        # Find boundary cells
        boundary_indices = self._find_boundary_cells(cells, domain_boundary) if preserve_boundary else set()
        
        # Find adjacency relationships using optimized method
        # adjacency = self._find_adjacent_cells(cells)
        
        # Compute inward offsets
        # offset_cells = self._compute_edge_bisectors(cells, adjacency, half_thickness)
        offset_cells = self._compute_edge_bisectors(cells, half_thickness)
        
        # Adjust boundary cells if needed
        if preserve_boundary:
            offset_cells = self._adjust_boundary_cells(
                cells, offset_cells, boundary_indices, domain_boundary, half_thickness
            )
        
        # Convert the dictionary to a list in the original order
        adjusted_cells = [offset_cells.get(i, Polygon()) for i in range(len(cells))]
        
        # Filter out invalid or empty polygons
        adjusted_cells = [cell for cell in adjusted_cells if cell.is_valid and not cell.is_empty]
        
        # Calculate the achieved area ratio
        achieved_ratio = self._calculate_area_ratio(cells, adjusted_cells)
        
        if self.verbose:
            print(f"Applied thickness: {thickness:.4f}")
            print(f"Achieved area ratio: {achieved_ratio:.4f}")
            
        return adjusted_cells
    
    def adjust_target_ratio(
        self,
        cells: List[Polygon],
        target_ratio: float,
        initial_thickness: float = 0.1,
        preserve_boundary: bool = True
    ) -> List[Polygon]:
        """
        Iteratively adjust cells to achieve target area ratio.
        
        Parameters
        ----------
        cells : List[Polygon]
            Input Voronoi cells
        target_ratio : float
            Target area ratio to achieve
        initial_thickness : float, optional
            Starting thickness value, default=0.1
        preserve_boundary : bool, optional
            Whether to preserve the original domain boundary (default: True)
            
        Returns
        -------
        List[Polygon]
            Adjusted Voronoi cells
        """
        self.history.clear()
        current_thickness = initial_thickness
        
        for iteration in range(self.max_iterations):
            # Adjust cells with current thickness
            adjusted_cells = self.adjust_fixed_thickness(
                cells, current_thickness, preserve_boundary
            )
            
            # Calculate achieved ratio
            achieved_ratio = self._calculate_area_ratio(cells, adjusted_cells)
            
            # Update history
            self.history.append((current_thickness, achieved_ratio))
            
            # Check convergence
            if abs(achieved_ratio - target_ratio) <= self.tolerance:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                    print(f"Final thickness: {current_thickness:.4f}")
                    print(f"Achieved ratio: {achieved_ratio:.4f}")
                return adjusted_cells
                
            # Update thickness through interpolation
            current_thickness = self._interpolate_thickness(target_ratio)
            
            if self.verbose:
                print(f"Iteration {iteration + 1}:")
                print(f"  Thickness: {current_thickness:.4f}")
                print(f"  Area ratio: {achieved_ratio:.4f}")
                
        # Maximum iterations reached
        if self.verbose:
            print("Warning: Maximum iterations reached")
            print(f"Best ratio: {achieved_ratio:.4f}")
            print(f"Final thickness: {current_thickness:.4f}")
            
        return adjusted_cells