from typing import List, Tuple, Dict, Optional, Set, Union
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon, MultiPolygon

class VoronoiEdgeExtractor:
    """
    A class for extracting and classifying edges from Voronoi tessellations.
    
    This class provides functionality to identify boundary and internal edges
    from Voronoi cells, with support for both modified (with gaps) and original
    cell configurations.
    
    Parameters
    ----------
    use_kdtree : bool, optional
        Whether to use KD-tree for edge mapping optimization, default=True
    
    Attributes
    ----------
    _edge_cache : Dict
        Cache of classified edges for each cell set
        
    Notes
    -----
    Edge classification can be performed in two modes:
    1. Using only modified cells (basic classification)
    2. Using both modified and original cells (advanced classification)
    
    The KD-tree optimization significantly improves performance for large
    tessellations by reducing edge mapping complexity from O(n²) to O(n log n).
    """
    
    def __init__(self, use_kdtree: bool = True):
        self.use_kdtree = use_kdtree
        self._edge_cache: Dict = {}
        
    def _normalize_edge(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Normalize edge orientation for consistent comparison.
        
        Parameters
        ----------
        p1, p2 : Tuple[float, float]
            Edge endpoint coordinates
            
        Returns
        -------
        Tuple[Tuple[float, float], Tuple[float, float]]
            Normalized edge with consistent orientation
        """
        if (p1[0] < p2[0]) or (p1[0] == p2[0] and p1[1] <= p2[1]):
            return (p1, p2)
        return (p2, p1)
        
    def _extract_polygon_edges(
        self,
        polygon: Polygon
    ) -> Set[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Extract edges from a single polygon.
        
        Parameters
        ----------
        polygon : Polygon
            Input polygon
            
        Returns
        -------
        Set[Tuple[Tuple[float, float], Tuple[float, float]]]
            Set of normalized edges
        """
        edges = set()
        # Process exterior
        coords = list(polygon.exterior.coords)
        for i in range(len(coords) - 1):
            edge = self._normalize_edge(coords[i], coords[i + 1])
            edges.add(edge)
            
        # Process interiors
        for interior in polygon.interiors:
            coords = list(interior.coords)
            for i in range(len(coords) - 1):
                edge = self._normalize_edge(coords[i], coords[i + 1])
                edges.add(edge)
                
        return edges
    
    def _classify_cell_edges(
        self,
        cells: List[Union[Polygon, MultiPolygon]]
    ) -> Dict[Tuple[Tuple[float, float], Tuple[float, float]], int]:
        """
        Count occurrences of each edge in cell set.
        
        Parameters
        ----------
        cells : List[Union[Polygon, MultiPolygon]]
            List of cells to process
            
        Returns
        -------
        Dict[Tuple[Tuple[float, float], Tuple[float, float]], int]
            Dictionary mapping edges to their occurrence count
        """
        edge_count = defaultdict(int)
        
        for cell in cells:
            if isinstance(cell, Polygon):
                edges = self._extract_polygon_edges(cell)
            else:  # MultiPolygon
                edges = set()
                for poly in cell.geoms:
                    edges.update(self._extract_polygon_edges(poly))
                    
            for edge in edges:
                edge_count[edge] += 1
                
        return edge_count
    
    def _create_edge_mapping(
        self,
        modified_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        original_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> Dict[Tuple[Tuple[float, float], Tuple[float, float]], 
              Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Create mapping between modified and original edges.
        
        Parameters
        ----------
        modified_edges : List[Tuple[Tuple[float, float], Tuple[float, float]]]
            Edges from modified cells
        original_edges : List[Tuple[Tuple[float, float], Tuple[float, float]]]
            Edges from original cells
            
        Returns
        -------
        Dict
            Mapping from modified to original edges
        """
        if not self.use_kdtree:
            # Simple nearest edge mapping
            mapping = {}
            for mod_edge in modified_edges:
                mod_mid = np.array([
                    (mod_edge[0][0] + mod_edge[1][0])/2,
                    (mod_edge[0][1] + mod_edge[1][1])/2
                ])
                
                min_dist = float('inf')
                nearest_edge = None
                
                for orig_edge in original_edges:
                    orig_mid = np.array([
                        (orig_edge[0][0] + orig_edge[1][0])/2,
                        (orig_edge[0][1] + orig_edge[1][1])/2
                    ])
                    dist = np.linalg.norm(mod_mid - orig_mid)
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_edge = orig_edge
                        
                mapping[mod_edge] = nearest_edge
            return mapping
            
        else:
            # KD-tree optimized mapping
            orig_edges_array = np.array(original_edges)
            orig_midpoints = np.array([
                [(edge[0][0] + edge[1][0])/2, (edge[0][1] + edge[1][1])/2]
                for edge in original_edges
            ])
            
            kdtree = cKDTree(orig_midpoints)
            mapping = {}
            
            for mod_edge in modified_edges:
                mod_midpoint = np.array([
                    (mod_edge[0][0] + mod_edge[1][0])/2,
                    (mod_edge[0][1] + mod_edge[1][1])/2
                ])
                
                _, idx = kdtree.query(mod_midpoint)
                orig_edge = tuple(map(tuple, orig_edges_array[idx]))
                mapping[mod_edge] = orig_edge
                
            return mapping
    
    def extract_edges(
        self,
        modified_cells: List[Union[Polygon, MultiPolygon]],
        original_cells: Optional[List[Union[Polygon, MultiPolygon]]] = None
    ) -> Tuple[List[LineString], List[LineString]]:
        """
        Extract and classify edges from Voronoi cells.
        
        Parameters
        ----------
        modified_cells : List[Union[Polygon, MultiPolygon]]
            Modified Voronoi cells (with gaps)
        original_cells : Optional[List[Union[Polygon, MultiPolygon]]], optional
            Original Voronoi cells without gaps
            
        Returns
        -------
        Tuple[List[LineString], List[LineString]]
            Lists of boundary and internal edges
            
        Examples
        --------
        >>> extractor = VoronoiEdgeExtractor()
        >>> boundary_edges, internal_edges = extractor.extract_edges(
        ...     modified_cells, original_cells
        ... )
        """
        # Classify modified cell edges
        modified_edge_count = self._classify_cell_edges(modified_cells)
        modified_edges = list(modified_edge_count.keys())
        
        if original_cells is not None:
            # Use original cells for classification
            original_edge_count = self._classify_cell_edges(original_cells)
            original_edges = list(original_edge_count.keys())
            
            # Create edge mapping
            edge_mapping = self._create_edge_mapping(
                modified_edges,
                original_edges
            )
            
            # Classify edges based on original topology
            boundary_edges = []
            internal_edges = []
            
            for mod_edge, orig_edge in edge_mapping.items():
                if original_edge_count[orig_edge] > 1:
                    internal_edges.append(LineString(mod_edge))
                else:
                    boundary_edges.append(LineString(mod_edge))
                    
        else:
            # Classify based on modified cells only
            boundary_edges = []
            internal_edges = []
            
            for edge, count in modified_edge_count.items():
                if count > 1:
                    internal_edges.append(LineString(edge))
                else:
                    boundary_edges.append(LineString(edge))
                    
        return boundary_edges, internal_edges
