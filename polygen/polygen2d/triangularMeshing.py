import os
import numpy as np
import meshio
from triangle import triangulate
from shapely.geometry import Polygon, LineString
from typing import List, Dict, Union, Optional
from collections.abc import Iterable

class TriangularMesher:
    """A class to handle triangular meshing of polygonal geometries.
    
    This class provides functionality to create high-quality triangular meshes
    from either single polygons (possibly with holes) or multiple polygons
    (e.g., Voronoi cells). It uses the Triangle library for mesh generation
    and ensures quality constraints are met.
    
    Attributes
    ----------
    min_edge_length : float
        Minimum desired edge length for triangles
    min_angle : float
        Minimum angle (in degrees) for triangle quality control
    max_iterations : int
        Maximum number of refinement iterations
    """
    
    def __init__(self, min_edge_length: float, min_angle: float = 30.0, 
                 max_iterations: int = 10):
        """
        Parameters
        ----------
        min_edge_length : float
            Minimum desired edge length for triangles
        min_angle : float, optional
            Minimum angle in degrees, by default 30.0
        max_iterations : int, optional
            Maximum refinement iterations, by default 10
        """
        self.min_edge_length = min_edge_length
        self.min_angle = min_angle
        self.max_iterations = max_iterations
        self.target_area = (min_edge_length ** 2) * np.sqrt(3) / 4
        
    def _subdivide_polygon_edges(self, cell: Polygon) -> Polygon:
        """Subdivide polygon edges to satisfy maximum segment length constraint.

        Parameters
        ----------
        cell : Polygon
            Input polygon with optional holes

        Returns
        -------
        Polygon
            Refined polygon with subdivided edges
        """
        has_interiors = bool(cell.interiors)
        
        def subdivide_ring(coords):
            new_coords = []
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i+1]
                segment = LineString([p1, p2])
                n_subdiv = int(np.ceil(segment.length / self.min_edge_length))
                
                for j in range(n_subdiv):
                    t = j / n_subdiv
                    x = p1[0] + t * (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    new_coords.append((x, y))
            
            new_coords.append(new_coords[0])
            return new_coords

        new_exterior = subdivide_ring(list(cell.exterior.coords))
        
        if has_interiors:
            new_interiors = [subdivide_ring(list(interior.coords)) 
                           for interior in cell.interiors]
            return Polygon(new_exterior, holes=new_interiors)
        return Polygon(new_exterior)

    def _compute_triangle_angles(self, coords: np.ndarray) -> List[float]:
        """Compute angles of a triangle in degrees.

        Parameters
        ----------
        coords : ndarray
            (3,2) array of triangle vertex coordinates

        Returns
        -------
        list
            Three angles in degrees
        """
        def length_sq(a, b):
            return (a[0]-b[0])**2 + (a[1]-b[1])**2

        s0 = length_sq(coords[0], coords[1])
        s1 = length_sq(coords[1], coords[2])
        s2 = length_sq(coords[2], coords[0])
        
        sides_sq = [s0, s1, s2]
        angles = []
        
        for i in range(3):
            a_sq = sides_sq[i]
            b_sq = sides_sq[(i+1)%3]
            c_sq = sides_sq[(i+2)%3]
            numerator = (b_sq + c_sq - a_sq)
            denominator = 2.0 * np.sqrt(b_sq * c_sq)
            angle = 180.0 if denominator == 0 else np.degrees(np.arccos(numerator / denominator))
            angles.append(angle)
        
        return angles

    def _check_mesh_quality(self, mesh: Dict) -> bool:
        """Check if mesh satisfies quality constraints.

        Parameters
        ----------
        mesh : dict
            Triangle mesh dictionary with vertices and triangles

        Returns
        -------
        bool
            True if mesh satisfies constraints
        """
        vertices = mesh['vertices']
        for tri_indices in mesh['triangles']:
            tri_coords = vertices[tri_indices]
            if min(self._compute_triangle_angles(tri_coords)) < self.min_angle:
                return False
            if Polygon(tri_coords).area > self.target_area:
                return False
        return True

    def _triangulate_cell(self, cell: Polygon) -> List[Polygon]:
        """Create quality triangular mesh for a single polygon.

        Parameters
        ----------
        cell : Polygon
            Input polygon with optional holes

        Returns
        -------
        list
            List of triangular polygons
        """
        if not (cell.is_valid and not cell.is_empty):
            return []

        refined_cell = self._subdivide_polygon_edges(cell)
        has_interiors = bool(refined_cell.interiors)
        
        # Prepare vertices and segments
        boundary_coords = []
        segments = []
        vertex_count = 0
        
        # Process exterior
        exterior_coords = list(refined_cell.exterior.coords)[:-1]
        boundary_coords.extend(exterior_coords)
        segments.extend([(i, (i+1) % len(exterior_coords)) 
                        for i in range(len(exterior_coords))])
        vertex_count += len(exterior_coords)
        
        # Process interiors if they exist
        if has_interiors:
            for interior in refined_cell.interiors:
                interior_coords = list(interior.coords)[:-1]
                hole_segments = [(vertex_count + i, 
                                vertex_count + (i+1) % len(interior_coords))
                               for i in range(len(interior_coords))]
                segments.extend(hole_segments)
                boundary_coords.extend(interior_coords)
                vertex_count += len(interior_coords)

        # Prepare PSLG
        cell_dict = {
            'vertices': np.array(boundary_coords),
            'segments': np.array(segments)
        }
        
        if has_interiors:
            holes = [[Polygon(interior).centroid.x, Polygon(interior).centroid.y]
                    for interior in refined_cell.interiors]
            cell_dict['holes'] = np.array(holes)

        # Initial triangulation
        init_cmd = f'pqa{self.target_area:.6f}q{self.min_angle}'
        mesh = triangulate(cell_dict, init_cmd)

        # Iterative refinement
        for _ in range(self.max_iterations):
            if self._check_mesh_quality(mesh):
                break
                
            refine_dict = {
                'vertices': mesh['vertices'],
                'triangles': mesh['triangles'],
                'segments': segments
            }
            if 'holes' in cell_dict:
                refine_dict['holes'] = cell_dict['holes']
            
            mesh = triangulate(refine_dict, f'pra{self.target_area}q{self.min_angle}')

        # Convert to polygons & clip
        cell_triangles = []
        for triangle_indices in mesh['triangles']:
            tri_vertices = mesh['vertices'][triangle_indices]
            tri_poly = Polygon(tri_vertices)
            if refined_cell.contains(tri_poly):
                cell_triangles.append(tri_poly)
            else:
                intersection = tri_poly.intersection(refined_cell)
                if (not intersection.is_empty and 
                    intersection.geom_type == 'Polygon' and
                    intersection.area > 0.95 * tri_poly.area):
                    cell_triangles.append(intersection)

        return cell_triangles

    def create_mesh(self, geometry: Union[Polygon, List[Polygon]]) -> Union[List[Polygon], Dict[int, List[Polygon]]]:
        """Create quality triangular mesh for polygon(s).

        This method creates a high-quality triangular mesh for either a single polygon
        (possibly with holes) or multiple polygons (e.g., Voronoi cells). The mesh
        satisfies constraints on minimum edge length and minimum angle.

        Parameters
        ----------
        geometry : Union[Polygon, List[Polygon]]
            Either a single polygon (possibly with holes) or a list of polygons
            to triangulate. Polygons can have interior holes.

        Returns
        -------
        Union[List[Polygon], Dict[int, List[Polygon]]]
            If input is a single polygon:
                List of triangular polygons forming the mesh
            If input is a list of polygons:
                Dictionary mapping polygon index to list of triangular polygons

        Notes
        -----
        - Uses Triangle library for mesh generation
        - Applies iterative refinement to meet quality constraints
        - Handles both simple polygons and polygons with holes
        - For multiple polygons, indices in output dictionary match input list indices
        
        Examples
        --------
        >>> mesher = TriangularMesher(min_edge_length=0.1)
        >>> # Single polygon
        >>> triangles = mesher.triangulate(polygon)
        >>> # Multiple polygons
        >>> cell_triangulations = mesher.triangulate([poly1, poly2, poly3])
        """
        if isinstance(geometry, Polygon):
            return self._triangulate_cell(geometry)
        
        elif isinstance(geometry, Iterable):
            cell_triangulations = {}
            for i, cell in enumerate(geometry):
                if not isinstance(cell, Polygon):
                    print(f"Warning: Item {i} is not a Polygon. Skipping...")
                    continue
                if not (cell.is_valid and not cell.is_empty):
                    print(f"Warning: Cell {i} is invalid or empty. Skipping...")
                    continue
                try:
                    cell_triangulations[i] = self._triangulate_cell(cell)
                except Exception as e:
                    print(f"Error triangulating cell {i}: {str(e)}")
                    cell_triangulations[i] = []
            return cell_triangulations
        
        else:
            raise TypeError("Input must be either a Polygon or an iterable of Polygons")
        
    def _convert_to_meshio(self, geometry: Union[Polygon, List[Polygon]], 
                       triangulation: Union[List[Polygon], Dict[int, List[Polygon]]]) -> 'meshio.Mesh':
        """Convert triangulated geometry to meshio format.
        
        Parameters
        ----------
        geometry : Union[Polygon, List[Polygon]]
            Original input geometry
        triangulation : Union[List[Polygon], Dict[int, List[Polygon]]]
            Triangulation result from create_mesh method
            
        Returns
        -------
        meshio.Mesh
            Mesh in meshio format with appropriate cell data
        """
        # Initialize data structures
        all_vertices = []
        vertex_to_index = {}
        triangles = []
        cell_ids = []
        index_counter = 0
        
        def clean_coords(coords):
            """Remove duplicate and repeated vertices from coordinate list."""
            rounded_coords = [(round(x, 12), round(y, 12)) for x, y in coords]
            cleaned = []
            for coord in rounded_coords:
                if not cleaned or not np.allclose(coord, cleaned[-1], rtol=1e-12):
                    cleaned.append(coord)
            if len(cleaned) > 1 and np.allclose(cleaned[0], cleaned[-1], rtol=1e-12):
                cleaned.pop()
            return cleaned
        
        def process_triangles(tri_list, cell_id=0):
            nonlocal index_counter
            for triangle in tri_list:
                coords = list(triangle.exterior.coords)
                cleaned_coords = clean_coords(coords)
                
                if len(cleaned_coords) != 3:
                    continue  # Skip invalid triangles silently
                
                # Convert vertices to indices
                triangle_indices = []
                for x, y in cleaned_coords:
                    coord_tuple = (float(x), float(y))
                    if coord_tuple not in vertex_to_index:
                        vertex_to_index[coord_tuple] = index_counter
                        all_vertices.append(coord_tuple)
                        index_counter += 1
                    triangle_indices.append(vertex_to_index[coord_tuple])
                
                # Skip degenerate triangles
                if len(set(triangle_indices)) != 3:
                    continue
                
                triangles.append(triangle_indices)
                cell_ids.append(cell_id)
        
        # Process geometry based on type
        if isinstance(geometry, Polygon):
            process_triangles(triangulation, cell_id=0)
        else:
            for cell_idx, cell_triangles in triangulation.items():
                process_triangles(cell_triangles, cell_id=cell_idx)
        
        # Ensure we have valid triangles
        if not triangles:
            raise ValueError("No valid triangles generated during mesh conversion")
        
        # Convert to numpy arrays
        points = np.array(all_vertices, dtype=np.float64)
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(len(points), dtype=np.float64)])
        
        # Create and return mesh
        return meshio.Mesh(
            points=points,
            cells=[("triangle", np.array(triangles, dtype=np.int32))],
            cell_data={
                "original_cell_id": [np.array(cell_ids, dtype=np.int32)],
                "triangle_id": [np.arange(len(triangles), dtype=np.int32)]
            }
        )

    def save_mesh(self, 
                geometry: Union[Polygon, List[Polygon]],
                triangulation: Union[List[Polygon], Dict[int, List[Polygon]]], 
                filename: str,
                output_dir: Optional[str] = None,
                file_format: Optional[str] = None) -> str:
        """Save triangulated geometry to any meshio-supported format.

        Parameters
        ----------
        geometry : Union[Polygon, List[Polygon]]
            Input geometry to triangulate and save
        triangulation : Union[List[Polygon], Dict[int, List[Polygon]]]
            Triangulation result from create_mesh method
        filename : str
            Output filename with appropriate extension
        output_dir : str, optional
            Directory to save the file. If None, uses 'meshFiles'
        file_format : str, optional
            Output format (e.g., 'vtk', 'abaqus'). If None, inferred from filename

        Returns
        -------
        str
            Full path to the saved file

        Examples
        --------
        >>> mesher = TriangularMesher(min_edge_length=0.1)
        >>> # Save single polygon mesh
        >>> mesher.save_mesh(polygon, "mesh.vtk")
        >>> # Save multiple polygons with explicit format
        >>> mesher.save_mesh(cells, "mesh.inp", output_dir="results", 
        ...                 file_format="abaqus")
        """
        try:
            # Handle output directory
            if output_dir is None:
                output_dir = "meshFiles"
            os.makedirs(output_dir, exist_ok=True)
            
            # Create full file path
            full_path = os.path.join(output_dir, filename)
            print(f"Saving triangular mesh to: {full_path}")
            
            # Get format mapping from meshio
            format_mapping = meshio._helpers.extension_to_filetypes
            supported_extensions = {ext.lstrip('.') for ext in format_mapping.keys()}
            
            # Determine format
            if file_format is None:
                file_extension = os.path.splitext(filename)[1].lower()
                if not file_extension:
                    raise ValueError(
                        f"No file extension provided. Supported: {', '.join(sorted(supported_extensions))}"
                    )
                
                clean_extension = file_extension.lstrip('.')
                if clean_extension not in supported_extensions:
                    raise ValueError(
                        f"Unsupported extension: '{clean_extension}'. Supported: {', '.join(sorted(supported_extensions))}"
                    )
                
                used_format = format_mapping[file_extension][0]
                print(f"Using format '{used_format}' for extension '{clean_extension}'")
            else:
                valid_formats = set()
                for formats in format_mapping.values():
                    valid_formats.update(formats)
                
                if file_format.lower() not in valid_formats:
                    raise ValueError(
                        f"Unsupported format: '{file_format}'. Supported: {', '.join(sorted(valid_formats))}"
                    )
                used_format = file_format.lower()
            
            # Create mesh and save
            # triangulation = self.create_mesh(geometry)
            mesh = self._convert_to_meshio(geometry, triangulation)
            mesh.write(full_path, file_format=used_format)
            print(f"Successfully saved triangular mesh to: {full_path}")
            
            return full_path
            
        except Exception as e:
            print(f"Error saving mesh to {full_path if 'full_path' in locals() else filename}: {str(e)}")
            raise

    @staticmethod
    def triangulateCells_simple(voronoi_cells):
        """
        Triangulate each Voronoi cell separately, including the centroid point in the triangulation,
        and maintaining cell boundaries.
        
        Parameters:
        -----------
        voronoi_cells : list of shapely.geometry.Polygon
            The clipped Voronoi cells from generate_voronoi_cells
            
        Returns:
        --------
        dict : Dictionary mapping cell index to list of triangles for that cell
        """
        from shapely.geometry import MultiPoint
        from shapely import delaunay_triangles
        
        cell_triangulations = {}
        
        for i, cell in enumerate(voronoi_cells):
            # Check if the cell is valid
            if cell.is_valid and not cell.is_empty:
                # Get the exterior coordinates of the cell
                exterior_coords = list(cell.exterior.coords)
                # Get the centroid coordinate
                centroid_coord = (cell.centroid.x, cell.centroid.y)
                # Combine the exterior coordinates with the centroid
                all_points = exterior_coords + [centroid_coord]
                # Create a MultiPoint geometry
                multipoint = MultiPoint(all_points)
                # Perform Delaunay triangulation on the points
                triangles = delaunay_triangles(multipoint)
                # Clip the triangles to the cell polygon
                cell_triangles = []
                for triangle in triangles.geoms:
                    # Intersect the triangle with the cell
                    intersection = triangle.intersection(cell)
                    # Check if the intersection is a Polygon (it might be a LineString or GeometryCollection)
                    if not intersection.is_empty and intersection.geom_type == 'Polygon':
                        cell_triangles.append(intersection)
                cell_triangulations[i] = cell_triangles
        
        return cell_triangulations

def triangulate_geometry(
    geometry: Union[Polygon, List[Polygon]], 
    min_edge_length: float,
    min_angle: float = 30.0,
    max_iterations: int = 10,
    save_mesh: bool = False,
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    file_format: Optional[str] = None
) -> Union[List[Polygon], Dict[int, List[Polygon]]]:
    """Create a high-quality triangular mesh for polygon(s).

    This function provides a simple interface to create triangular meshes for either
    a single polygon or multiple polygons. It handles polygons with or without interior
    holes and ensures the resulting mesh meets quality constraints for minimum edge
    length and minimum angles.

    Parameters
    ----------
    geometry : Union[Polygon, List[Polygon]]
        Input geometry to triangulate. Can be either:
        * A single shapely.geometry.Polygon (possibly with holes)
        * A list of shapely.geometry.Polygon (e.g., Voronoi cells)
    min_edge_length : float
        Minimum desired edge length for triangles in the mesh.
        Controls the mesh density.
    min_angle : float, optional
        Minimum allowed angle in degrees for triangles, by default 30.0.
        Higher values (up to 34) create better-quality triangles but
        may require more iterations.
    max_iterations : int, optional
        Maximum number of refinement iterations, by default 10.
        Increase if mesh quality constraints are not being met.
    save_mesh : bool, optional
        Whether to save the mesh to a file, by default False.
        If True, either filename must be provided or a default will be used.
    filename : str, optional
        Output filename with appropriate extension for mesh saving.
        Required if save_mesh is True and must include a valid extension.
        Common extensions: .vtk, .msh, .inp (Abaqus), .ans (ANSYS)
    output_dir : str, optional
        Directory to save the mesh file. If None, uses 'meshFiles'.
        Directory will be created if it doesn't exist.
    file_format : str, optional
        Output format override (e.g., 'vtk', 'abaqus', 'ansys').
        If None, format is inferred from filename extension.

    Returns
    -------
    Union[List[Polygon], Dict[int, List[Polygon]]]
        If input is a single polygon:
            List of triangular polygons forming the mesh
        If input is a list of polygons:
            Dictionary mapping polygon index to list of triangular polygons

    Raises
    ------
    ValueError
        If save_mesh is True but no valid filename is provided.
        If the specified file format is not supported.
    RuntimeError
        If there are issues during mesh saving.

    Notes
    -----
    - Uses the Triangle library (J.R. Shewchuk) for mesh generation
    - Applies iterative refinement to meet quality constraints
    - Automatically handles both simple polygons and polygons with holes
    - For multiple polygons, indices in output dictionary match input list indices
    - Invalid or empty polygons are skipped with a warning message
    - Failed triangulations return an empty list for that polygon

    Examples
    --------
    >>> # Basic triangulation without saving
    >>> domain = Polygon([(0,0), (1,0), (1,1), (0,1)],
    ...                 holes=[[(0.2,0.2), (0.4,0.2), (0.4,0.4), (0.2,0.4)]])
    >>> triangles = triangulate_geometry(domain, min_edge_length=0.1)
    
    >>> # Triangulate and save mesh
    >>> cells = [poly1, poly2, poly3]  # list of shapely Polygons
    >>> cell_triangulations = triangulate_geometry(
    ...     cells, 
    ...     min_edge_length=0.1,
    ...     min_angle=32,
    ...     save_mesh=True,
    ...     filename="mesh.vtk",
    ...     output_dir="results"
    ... )

    See Also
    --------
    TriangularMesher : The class that implements the triangulation functionality
    """
    mesher = TriangularMesher(min_edge_length, min_angle, max_iterations)
    
    # Create the mesh
    triangulation = mesher.create_mesh(geometry)
    
    # Save if requested
    if save_mesh:
        if filename is None:
            # Create default filename based on input type
            base_name = "single_polygon" if isinstance(geometry, Polygon) else "multi_polygon"
            filename = f"{base_name}_mesh.obj"
            print(f"No filename provided. Using default: {filename}")
        try:
            mesher.save_mesh(geometry, triangulation, filename, output_dir, file_format)
        except Exception as e:
            print(f"Warning: Failed to save mesh: {str(e)}")
            print("Returning triangulation result anyway.")
    
    return triangulation
