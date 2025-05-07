import os
import meshio
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, LineString
from typing import List, Dict, Optional, Union
from .extractVoronoiEdges import VoronoiEdgeExtractor

class IO:
    """
    A comprehensive class for handling input/output operations for geometric data.
    
    This class provides methods for:
    1. Loading mesh files into Shapely polygons
    2. Saving Voronoi cells to various formats
    3. Exporting structured data for further analysis
    4. Handling boundary and internal edge data
    """
    
    #: Default precision for coordinate values in output files.
    #: :no-index:
    DEFAULT_PRECISION = 8
    
    #: Default directory for Voronoi mesh outputs.
    #: :no-index:
    VORONOI_OUTPUT_DIR = "meshFiles"
    
    #: Default directory for data file outputs.
    #: :no-index:
    DATA_OUTPUT_DIR = "voronoiDataFiles"
    
    @classmethod
    def _ensure_directory(cls, directory: str) -> None:
        """Create directory if it doesn't exist."""
        os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def _round_coord(coord: tuple, precision: int) -> tuple:
        """Round coordinates to specified precision."""
        return tuple(round(x, precision) for x in coord)
    
    @staticmethod
    def _clean_polygon_vertices(coords: List[tuple], precision: int) -> Optional[List[tuple]]:
        """
        Clean polygon vertices by removing duplicates and ensuring valid geometry.
        
        Parameters
        ----------
        coords : List[tuple]
            List of vertex coordinates
        precision : int
            Decimal precision for coordinate rounding
            
        Returns
        -------
        Optional[List[tuple]]
            Cleaned vertex list or None if invalid
        """
        rounded_coords = [IO._round_coord(coord, precision) for coord in coords]
        cleaned_coords = []
        
        for i in range(len(rounded_coords)):
            current = rounded_coords[i]
            next_vertex = rounded_coords[(i + 1) % len(rounded_coords)]
            if current != next_vertex:
                cleaned_coords.append(current)
                
        if len(cleaned_coords) < 3 or len(set(cleaned_coords)) != len(cleaned_coords):
            return None
            
        return cleaned_coords
    
    @staticmethod
    def _process_boundary(
        coords: List[tuple],
        lines_to_save: List[tuple],
        lines_list: List[tuple]
    ) -> None:
        """Process boundary lines from coordinates."""
        for i in range(len(coords) - 1):
            line = LineString([coords[i], coords[i + 1]])
            line_tuple = (
                (line.coords[0][0], line.coords[0][1]),
                (line.coords[1][0], line.coords[1][1])
            )
            lines_to_save.append(line_tuple)
            lines_list.append(line_tuple)
    
    @classmethod
    def load_polygon_from_file(cls, input_source: Union[str, Dict]) -> Union[Polygon, MultiPolygon]:
        """
        Load a polygon from file or predefined geometric object.
        :no-index:

        Parameters
        ----------
        input_source : Union[str, Dict]
            File path or geometric object.

        Returns
        -------
        Union[Polygon, MultiPolygon]
            Loaded geometry.
        """
        try:
            if isinstance(input_source, dict) and 'polygon' in input_source:
                if not input_source['polygon'].is_valid:
                    raise ValueError("Invalid polygon in geometric object")
                return input_source['polygon']
                
            elif isinstance(input_source, str):
                if not os.path.exists(input_source):
                    raise FileNotFoundError(f"File not found: {input_source}")
                    
                file_extension = os.path.splitext(input_source)[1].lstrip('.').lower()
                meshio_formats = sorted(list(meshio._helpers.extension_to_filetypes.keys()))
                supported_extensions = {ext.lstrip('.') for ext in meshio_formats}
                
                if file_extension not in supported_extensions:
                    raise ValueError(
                        f"Unsupported format: {file_extension}. "
                        f"Supported: {', '.join(sorted(supported_extensions))}"
                    )
                
                mesh = meshio.read(input_source)
                if len(mesh.points) == 0:
                    raise ValueError("Empty vertex data")
                
                polygons = []
                valid_cell_types = {'triangle', 'quad', 'polygon'}
                
                for cell_block in mesh.cells:
                    if cell_block.type in valid_cell_types:
                        vertices = mesh.points[:, :2] if mesh.points.shape[1] > 2 else mesh.points
                        for i, face in enumerate(cell_block.data):
                            face_vertices = vertices[face]
                            if len(face_vertices) >= 3:
                                try:
                                    poly = Polygon(face_vertices)
                                    if poly.is_valid and not poly.is_empty:
                                        polygons.append(poly)
                                except Exception as e:
                                    print(f"Warning: Invalid face {i}: {str(e)}")
                
                if not polygons:
                    raise ValueError("No valid polygons in file")
                
                unified = unary_union(polygons)
                if not unified.is_valid:
                    raise RuntimeError("Invalid unified polygon")
                
                print(f"Loaded {len(polygons)} polygons from {input_source}")
                return unified
                
            else:
                raise ValueError("Input must be file path or geometric object")
                
        except Exception as e:
            print(f"\nError ({type(e).__name__}):")
            print(f"{'='*(len(type(e).__name__)+8)}")
            print(str(e))
            raise
    
    @classmethod
    def save_voronoi_to_obj(
        cls,
        voronoi_cells: List[Polygon],
        filename: str,
        output_dir: Optional[str] = None,
        precision: int = DEFAULT_PRECISION
    ) -> None:
        """
        Save Voronoi cells to OBJ format.
        :no-index:

        Parameters
        ----------
        voronoi_cells : List[Polygon]
            Voronoi cells to save.
        filename : str
            Output filename.
        output_dir : Optional[str]
            Output directory.
        precision : int
            Coordinate precision.
        """
        output_dir = output_dir or cls.VORONOI_OUTPUT_DIR
        cls._ensure_directory(output_dir)
        
        filepath = os.path.join(output_dir, filename)
        if not filepath.lower().endswith('.obj'):
            filepath += '.obj'
            
        print(f"Saving Voronoi mesh to: {filepath}")
        
        all_vertices = set()
        valid_cells = []
        cell_vertex_lists = []
        
        for i, cell in enumerate(voronoi_cells):
            if not cell.is_valid:
                print(f"Warning: Skipping invalid cell {i}")
                continue
                
            coords = list(cell.exterior.coords)[:-1]
            cleaned = cls._clean_polygon_vertices(coords, precision)
            
            if cleaned is None:
                print(f"Warning: Skipping degenerate cell {i}")
                continue
                
            all_vertices.update(cleaned)
            cell_vertex_lists.append(cleaned)
            valid_cells.append(cell)
            
        vertices_list = sorted(list(all_vertices))
        vertex_to_index = {v: i+1 for i, v in enumerate(vertices_list)}
        
        with open(filepath, 'w') as f:
            f.write(f"# Voronoi cells\n")
            f.write(f"# Original cells: {len(voronoi_cells)}\n")
            f.write(f"# Valid cells: {len(valid_cells)}\n")
            f.write(f"# Vertices: {len(vertices_list)}\n")
            f.write(f"# Precision: {precision}\n\n")
            
            for vertex in vertices_list:
                if len(vertex) == 2:
                    f.write(f"v {vertex[0]:.{precision}f} {vertex[1]:.{precision}f} 0.0\n")
                else:
                    f.write(f"v {vertex[0]:.{precision}f} {vertex[1]:.{precision}f} "
                           f"{vertex[2]:.{precision}f}\n")
                           
            f.write("\n")
            
            for i, vertex_list in enumerate(cell_vertex_lists):
                indices = [vertex_to_index[v] for v in vertex_list]
                if len(set(indices)) == len(indices):
                    f.write(f"f {' '.join(map(str, indices))}\n")
                else:
                    print(f"Warning: Skipping cell {i} with duplicate vertices")
                    
        print(f"Successfully saved Voronoi mesh to: {filepath}")
    
    @classmethod
    def save_voronoi_data(
        cls,
        target_cells: List[Polygon],
        filename: str = 'voronoi_data.py',
        data_name: str = 'voronoi_lines',
        original_cells: Optional[List[Polygon]] = None
    ) -> None:
        """
        Save structured Voronoi data including edges.
        :no-index:
        
        Parameters
        ----------
        target_cells : List[Polygon]
            Voronoi cells to save
        filename : str
            Output filename
        data_name : str
            Base name for data variables
        original_cells : Optional[List[Polygon]]
            Original cells for edge classification
        """
        cls._ensure_directory(cls.DATA_OUTPUT_DIR)
        filepath = os.path.join(cls.DATA_OUTPUT_DIR, filename)
        print(f"Saving Voronoi data to: {filepath}")
        
        lines_to_save = []
        boundary_lines = []
        internal_lines = []
        
        # Process all cell edges
        def process_polygon(poly):
            exterior_coords = list(poly.exterior.coords)
            for i in range(len(exterior_coords) - 1):
                line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                lines_to_save.append((
                    (line.coords[0][0], line.coords[0][1]),
                    (line.coords[1][0], line.coords[1][1])
                ))
        
        for cell in target_cells:
            if isinstance(cell, Polygon):
                process_polygon(cell)
            elif isinstance(cell, MultiPolygon):
                for poly in cell.geoms:
                    process_polygon(poly)
        
        # Extract and classify edges
        extractor = VoronoiEdgeExtractor(use_kdtree=True)
        boundary_edges, internal_edges = extractor.extract_edges(
            modified_cells=target_cells,
            original_cells=original_cells
        )
        
        # Process boundary edges
        for line in boundary_edges:
            boundary_lines.append((
                (line.coords[0][0], line.coords[0][1]),
                (line.coords[1][0], line.coords[1][1])
            ))
        
        # Process internal edges
        for line in internal_edges:
            internal_lines.append((
                (line.coords[0][0], line.coords[0][1]),
                (line.coords[1][0], line.coords[1][1])
            ))
        
        # Write data
        with open(filepath, 'w') as f:
            f.write(f"{data_name} = {repr(lines_to_save)}\n")
            f.write(f"{data_name}_boundaries = {repr(boundary_lines)}\n")
            f.write(f"{data_name}_internalEdges = {repr(internal_lines)}\n")
        print(f"Successfully saved Voronoi data to: {filepath}")
    
    @classmethod
    def save_polygon_data(
        cls,
        polygon: Union[Polygon, MultiPolygon],
        filename: str = 'polygon_data.py',
        data_name: str = 'polygon_boundaries'
    ) -> None:
        """
        Save polygon boundary data.
        :no-index:
        
        Parameters
        ----------
        polygon : Union[Polygon, MultiPolygon]
            Polygon to save
        filename : str
            Output filename
        data_name : str
            Base name for data variables
        """
        cls._ensure_directory(cls.DATA_OUTPUT_DIR)
        filepath = os.path.join(cls.DATA_OUTPUT_DIR, filename)
        print(f"Saving boundary data to: {filepath}")
        
        lines_to_save = []
        exterior_lines = []
        interior_lines = []
        
        if isinstance(polygon, Polygon):
            cls._process_boundary(list(polygon.exterior.coords),
                               lines_to_save, exterior_lines)
            for interior in polygon.interiors:
                cls._process_boundary(list(interior.coords),
                                  lines_to_save, interior_lines)
        else:  # MultiPolygon
            for poly in polygon.geoms:
                cls._process_boundary(list(poly.exterior.coords),
                                  lines_to_save, exterior_lines)
                for interior in poly.interiors:
                    cls._process_boundary(list(interior.coords),
                                     lines_to_save, interior_lines)
        
        with open(filepath, 'w') as f:
            f.write(f"{data_name} = {repr(lines_to_save)}\n")
            f.write(f"{data_name}_exterior = {repr(exterior_lines)}\n")
            f.write(f"{data_name}_interior = {repr(interior_lines)}\n")
        print(f"Successfully saved boundary data to: {filepath}")
