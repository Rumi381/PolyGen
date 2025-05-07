from typing import List, Tuple, Union, Optional, Dict
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union

class MeshOptimizer:
    """
    A class for optimizing mesh geometries by collapsing short edges in Voronoi cells
    and polygon boundaries.

    This class provides functionality to analyze and optimize geometric meshes by
    identifying and collapsing short edges while preserving the overall structure
    of the mesh.

    Parameters
    ----------
    min_edge_length : float, optional
        Minimum edge length threshold for optimization, by default None
    verbose : bool, optional
        Whether to print detailed information during processing, by default False

    Attributes
    ----------
    _min_edge_length : float
        Stored minimum edge length threshold
    _verbose : bool
        Flag for verbose output
    """

    def __init__(self, min_edge_length: Optional[float] = None, verbose: bool = False):
        self._min_edge_length = min_edge_length
        self._verbose = verbose
        self._point_map: Dict = {}

    def _collect_polygon_edges(self, 
                             polygon: Union[Polygon, MultiPolygon]
                             ) -> List[Tuple[float, LineString, str, int]]:
        """
        Collect edges from a polygon with their associated metadata.

        Parameters
        ----------
        polygon : Union[Polygon, MultiPolygon]
            Input polygon to collect edges from

        Returns
        -------
        List[Tuple[float, LineString, str, int]]
            List of tuples containing (edge_length, line, boundary_type, index)
        """
        edges = []
        
        def process_ring(coords, ring_type, ring_index=None):
            for i in range(len(coords) - 1):
                line = LineString([coords[i], coords[i + 1]])
                edge_info = (line.length, line, 
                           (ring_type, ring_index) if ring_index is not None else ring_type, 
                           i)
                edges.append(edge_info)

        if isinstance(polygon, Polygon):
            process_ring(list(polygon.exterior.coords), 'exterior')
            for idx, interior in enumerate(polygon.interiors):
                process_ring(list(interior.coords), 'interior', idx)
        else:  # MultiPolygon
            for poly_idx, poly in enumerate(polygon.geoms):
                process_ring(list(poly.exterior.coords), 'exterior', poly_idx)
                for int_idx, interior in enumerate(poly.interiors):
                    process_ring(list(interior.coords), 'interior', (poly_idx, int_idx))

        return edges

    def _find_root(self, vertex: Tuple[float, float]) -> Tuple[float, float]:
        """
        Find the root vertex in the union-find data structure with path compression.

        Parameters
        ----------
        vertex : Tuple[float, float]
            Input vertex coordinates

        Returns
        -------
        Tuple[float, float]
            Root vertex coordinates
        """
        root = vertex
        while root in self._point_map:
            root = self._point_map[root]
        
        # Path compression
        while vertex != root:
            parent = self._point_map[vertex]
            self._point_map[vertex] = root
            vertex = parent
            
        return root

    def _create_vertex_mapping(self, 
                             edges_to_collapse: List[Tuple[float, LineString, str, int]]
                             ) -> None:
        """
        Create vertex mapping for edge collapse operations.

        Parameters
        ----------
        edges_to_collapse : List[Tuple[float, LineString, str, int]]
            List of edges to be collapsed
        """
        self._point_map.clear()
        merge_list = {}
        
        # Create initial merge list
        for _, line, _, _ in edges_to_collapse:
            start, end = line.coords[0], line.coords[1]
            merge_list[end] = start

        # Create unified vertex mapping
        for old_point, new_point in merge_list.items():
            root_old = self._find_root(old_point)
            root_new = self._find_root(new_point)
            if root_old != root_new:
                self._point_map[root_old] = root_new

    def analyze_voronoi_edges(self, 
                            voronoi_cells: List[Union[Polygon, MultiPolygon]], 
                            n: Optional[int] = None, 
                            threshold: Optional[float] = None) -> None:
        """
        Analyze and print information about short edges in Voronoi cells.

        Parameters
        ----------
        voronoi_cells : List[Union[Polygon, MultiPolygon]]
            List of Voronoi cells to analyze
        n : int, optional
            Number of shortest edges to analyze, by default None
        threshold : float, optional
            Length threshold for edge analysis, by default None

        Raises
        ------
        ValueError
            If neither or both n and threshold are specified
        """
        if (n is None and threshold is None) or (n is not None and threshold is not None):
            raise ValueError("Specify exactly one of 'n' or 'threshold'")

        edges = []
        for cell in voronoi_cells:
            edges.extend(self._collect_polygon_edges(cell))
        
        edges.sort(key=lambda x: x[0])
        edges_to_print = edges[:n] if n is not None else [e for e in edges if e[0] < threshold]

        if not edges_to_print:
            print(f'No edges found {"shorter than " + str(threshold) if threshold else ""}')
            return

        print(f'{"First " + str(n) if n else "All"} edge lengths in Voronoi cells:')
        for idx, (length, _, _, _) in enumerate(edges_to_print):
            print(f"Edge {idx}: {length:.6f}")

    def optimize_voronoi_cells(self, 
                             voronoi_cells: List[Union[Polygon, MultiPolygon]], 
                             n: Optional[int] = None, 
                             threshold: Optional[float] = None
                             ) -> List[Union[Polygon, MultiPolygon]]:
        """
        Optimize Voronoi cells by collapsing short edges.

        Parameters
        ----------
        voronoi_cells : List[Union[Polygon, MultiPolygon]]
            List of Voronoi cells to optimize
        n : int, optional
            Number of shortest edges to collapse, by default None
        threshold : float, optional
            Length threshold for edge collapse, by default None

        Returns
        -------
        List[Union[Polygon, MultiPolygon]]
            Optimized Voronoi cells

        Raises
        ------
        ValueError
            If neither or both n and threshold are specified
        """
        if (n is None and threshold is None) or (n is not None and threshold is not None):
            raise ValueError("Specify exactly one of 'n' or 'threshold'")

        # Collect and sort edges
        edges = []
        for cell in voronoi_cells:
            edges.extend(self._collect_polygon_edges(cell))
        edges.sort(key=lambda x: x[0])

        # Select edges to collapse
        edges_to_collapse = edges[:n] if n is not None else [e for e in edges if e[0] < threshold]

        if not edges_to_collapse:
            if self._verbose:
                print('No edges to collapse, returning original cells')
            return voronoi_cells

        # Create vertex mapping and optimize cells
        self._create_vertex_mapping(edges_to_collapse)
        
        modified_cells = []
        for cell in voronoi_cells:
            if isinstance(cell, Polygon):
                new_coords = [self._find_root(coord) for coord in cell.exterior.coords[:-1]]
                modified_cells.append(Polygon(new_coords))
            else:  # MultiPolygon
                new_polys = []
                for poly in cell.geoms:
                    new_coords = [self._find_root(coord) for coord in poly.exterior.coords[:-1]]
                    new_polys.append(Polygon(new_coords))
                modified_cells.append(MultiPolygon(new_polys))

        return modified_cells

    def analyze_boundary_edges(self, 
                             polygon: Union[Polygon, MultiPolygon], 
                             n: Optional[int] = None, 
                             threshold: Optional[float] = None) -> None:
        """
        Analyze and print information about short edges in polygon boundaries.

        Parameters
        ----------
        polygon : Union[Polygon, MultiPolygon]
            Polygon to analyze
        n : int, optional
            Number of shortest edges to analyze, by default None
        threshold : float, optional
            Length threshold for edge analysis, by default None

        Raises
        ------
        ValueError
            If neither or both n and threshold are specified
        """
        if (n is None and threshold is None) or (n is not None and threshold is not None):
            raise ValueError("Specify exactly one of 'n' or 'threshold'")

        edges = self._collect_polygon_edges(polygon)
        edges.sort(key=lambda x: x[0])
        
        edges_to_print = edges[:n] if n is not None else [e for e in edges if e[0] < threshold]

        if not edges_to_print:
            print(f'No edges found {"shorter than " + str(threshold) if threshold else ""}')
            return

        print(f'{"First " + str(n) if n else "All"} edge lengths in boundary:')
        for idx, (length, _, _, _) in enumerate(edges_to_print):
            print(f"Edge {idx}: {length:.6f}")

    def optimize_boundary(self, 
                        polygon: Union[Polygon, MultiPolygon], 
                        n: Optional[int] = None, 
                        threshold: Optional[float] = None
                        ) -> Union[Polygon, MultiPolygon]:
        """
        Optimize polygon boundaries by collapsing short edges.

        Parameters
        ----------
        polygon : Union[Polygon, MultiPolygon]
            Polygon to optimize
        n : int, optional
            Number of shortest edges to collapse, by default None
        threshold : float, optional
            Length threshold for edge collapse, by default None

        Returns
        -------
        Union[Polygon, MultiPolygon]
            Optimized polygon

        Raises
        ------
        ValueError
            If neither or both n and threshold are specified
        """
        if (n is None and threshold is None) or (n is not None and threshold is not None):
            raise ValueError("Specify exactly one of 'n' or 'threshold'")

        # Collect and sort edges
        edges = self._collect_polygon_edges(polygon)
        edges.sort(key=lambda x: x[0])

        # Select edges to collapse
        edges_to_collapse = edges[:n] if n is not None else [e for e in edges if e[0] < threshold]

        if not edges_to_collapse:
            if self._verbose:
                print('No edges to collapse, returning original polygon')
            return polygon

        # Create vertex mapping
        self._create_vertex_mapping(edges_to_collapse)

        # Update polygon vertices
        def update_polygon(poly: Polygon) -> Polygon:
            new_exterior = [self._find_root(coord) for coord in poly.exterior.coords[:-1]]
            new_interiors = []
            for interior in poly.interiors:
                new_interior = [self._find_root(coord) for coord in interior.coords[:-1]]
                new_interiors.append(new_interior)
            return Polygon(new_exterior, new_interiors)

        if isinstance(polygon, Polygon):
            return update_polygon(polygon)
        else:  # MultiPolygon
            modified_polys = [update_polygon(poly) for poly in polygon.geoms]
            return unary_union(modified_polys)
