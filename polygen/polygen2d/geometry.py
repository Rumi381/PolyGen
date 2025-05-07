import numpy as np
import math
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.affinity import scale, rotate

class Geometry:
    """
    A class for generating various geometric shapes as Shapely polygons.

    Methods
    -------
    circle(center, radius=None, point=None, name='Circle')
        Create a circular region given the center and either a radius or a point on the circle.
    
    annular_region(center_outer, radius_outer=None, point_outer=None, center_inner=None, radius_inner=None, point_inner=None, name='AnnularRegion')
        Create an annular region (a ring-shaped region) with specified outer and inner circles.
    
    annular_region_arc_WithStraightEnds(...)
        Create an annular region arc with straight ends using circular arcs.
    
    annular_region_arc_WithRoundedEnds(...)
        Create an annular region arc with smoothly closed ends using circular arcs.
    
    ellipse(center, semi_major_axis, semi_minor_axis, rotation=0, name='Ellipse')
        Create an elliptical region.
    
    annular_ellipse(...)
        Create an annular elliptical region.
    
    rectangle(width, height, name='Rectangle')
        Create a rectangular region using width and height.
    
    rectangle_corners(point1, point2, name='RectangleCorners')
        Create a rectangular region using two opposite corners.
    
    square(center, side_length, name='Square')
        Create a square region given the center and side length.
    
    regular_polygon(center, radius, num_sides, name='RegularPolygon')
        Create a regular polygon with a given number of sides.
    
    triangle(point1, point2, point3, name='Triangle')
        Create a triangular region using three vertices.
    
    pentagon(center, radius, name='Pentagon')
        Create a pentagonal region.
    
    hexagon(center, radius, name='Hexagon')
        Create a hexagonal region.
    
    octagon(center, radius, name='Octagon')
        Create an octagonal region.
    
    star(center, outer_radius, inner_radius, num_points, name='Star')
        Create a star-shaped polygon.
    
    trapezoid(base1, base2, height, name='Trapezoid')
        Create a trapezoidal region given the lengths of the bases and height.
    
    parallelogram(base, side_length, height, name='Parallelogram')
        Create a parallelogram given the base, side length, and height.

    Examples
    --------
    >>> from voronoi_meshing.geometry import Geometry
    >>> 
    >>> # Create a circle
    >>> circle1 = Geometry.circle((0, 0), radius=5, name='CircleByRadius')
    >>> circle2 = Geometry.circle((0, 0), point=(3, 4), name='CircleByPoint')
    >>> 
    >>> # Create an annular region
    >>> annular1 = Geometry.annular_region((0, 0), radius_outer=10, center_inner=(0, 0),
    ...                                   radius_inner=5, name='AnnularByRadius')
    >>> annular2 = Geometry.annular_region((0, 0), point_outer=(7, 7), center_inner=(0, -1),
    ...                                   point_inner=(3, 4), name='AnnularByPoint')
    >>> 
    >>> # Create an elliptical region
    >>> ellipse = Geometry.ellipse(center=(0, 0), semi_major_axis=7,
    ...                          semi_minor_axis=4, rotation=30, name='Ellipse')
    >>> 
    >>> # Create a rectangular region
    >>> rectangle = Geometry.rectangle(width=10, height=5, name='Rectangle')
    >>> 
    >>> # Create a star-shaped region
    >>> star = Geometry.star(center=(0, 0), outer_radius=7,
    ...                    inner_radius=3, num_points=5, name='Star')

    Returns
    -------
    Each method returns a dictionary containing:
        - 'polygon': The Shapely polygon object
        - 'name': The name of the geometry
    """
    @staticmethod
    def circle(center, radius=None, point=None, name='Circle'):
        """
        Create a circular region.
        :no-index:
        
        Parameters:
        - center: Tuple (x, y) representing the center of the circle.
        - radius: Radius of the circle (optional if point is provided).
        - point: Tuple (x, y) representing a point on the circle (optional if radius is provided).
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - circle1 = Geometry.circle((0, 0), radius=5, name='CircleByRadius') # With center and radius

        - circle2 = Geometry.circle((0, 0), point=(3, 4), name='CircleByPoint') # With center and a point on the perimeter
        """
        if radius is None and point is None:
            raise ValueError("Either 'radius' or 'point' must be provided.")
        if radius is None:
            radius = Point(center).distance(Point(point))
        center_point = Point(center)
        circle = center_point.buffer(radius)
        return {'polygon': circle, 'name': name}

    @staticmethod
    def annular_region(center_outer, radius_outer=None, point_outer=None, center_inner=None, radius_inner=None, point_inner=None, name='AnnularRegion'):
        """
        Create an annular region (a ring-shaped region).
        :no-index:
        
        Parameters:
        - center_outer: Tuple (x, y) representing the center of the outer circle.
        - radius_outer: Radius of the outer circle (optional if point_outer is provided).
        - point_outer: Tuple (x, y) representing a point on the outer circle (optional if radius_outer is provided).
        - center_inner: Tuple (x, y) representing the center of the inner circle.
        - radius_inner: Radius of the inner circle (optional if point_inner is provided).
        - point_inner: Tuple (x, y) representing a point on the inner circle (optional if radius_inner is provided).
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - annular1 = Geometry.annular_region((0, 0), radius_outer=10, center_inner=(0, 0), radius_inner=5, name='AnnularByRadius') # With center and radius

        - annular2 = Geometry.annular_region((0, 0), point_outer=(7, 7), center_inner=(0, -1), point_inner=(3, 4), name='AnnularByPoint') # With center and a point on the perimeter
        """
        if radius_outer is None and point_outer is None:
            raise ValueError("Either 'radius_outer' or 'point_outer' must be provided.")
        if radius_inner is None and point_inner is None:
            raise ValueError("Either 'radius_inner' or 'point_inner' must be provided.")
        if radius_outer is None:
            radius_outer = Point(center_outer).distance(Point(point_outer))
        if radius_inner is None:
            radius_inner = Point(center_inner).distance(Point(point_inner))
        
        outer_circle = Point(center_outer).buffer(radius_outer)
        inner_circle = Point(center_inner).buffer(radius_inner)
        annular_region = outer_circle.difference(inner_circle)
        return {'polygon': annular_region, 'name': name}
    
    @staticmethod
    def annular_region_arc_WithStraightEnds(center_outer, radius_outer=None, point_outer=None, center_inner=None, radius_inner=None, point_inner=None, theta=0, axis='x', alpha=None, name='AnnularRegionArcWithStraightEnds'):
        """
        Create an annular region arc where the ends are closed by straight lines.
        :no-index:
        
        Parameters:
        - center_outer: Tuple (x, y) representing the center of the outer circle.
        - radius_outer: Radius of the outer circle (optional if point_outer is provided).
        - point_outer: Tuple (x, y) representing a point on the outer circle (optional if radius_outer is provided).
        - center_inner: Tuple (x, y) representing the center of the inner circle.
        - radius_inner: Radius of the inner circle (optional if point_inner is provided).
        - point_inner: Tuple (x, y) representing a point on the inner circle (optional if radius_inner is provided).
        - theta: Angle in degrees of the arc.
        - axis: Axis of symmetry ('x' or 'y' or 'alpha').
        - alpha: Angle in degrees of the axis of symmetry with respect to the positive x-axis.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - annular_arc_straight1 = Geometry.annular_region_arc_WithStraightEnds((0, 0), radius_outer=10, center_inner=(0, 0), radius_inner=5, theta=240, axis='y', name='ArcByRadiusY') # Creates a 240 degree annular arc With center and radius and symmetric about Y axis
        
        - annular_arc_straight2 = Geometry.annular_region_arc_WithStraightEnds((0, 0), point_outer=(7, 7), center_inner=(0, 0), point_inner=(3, 4), theta=240, axis='x', name='ArcByPointX') # Creates a 240 degree annular arc With center and a point on the perimeter and symmetric about Y axis

        - annular_arc_straight3 = Geometry.annular_region_arc_WithStraightEnds((0, 0), radius_outer=10, center_inner=(0, -1), radius_inner=5, theta=240, alpha=135, name='ArcByRadiusAlpha') # Creates a 240 degree annular arc With center and a radius and symmetric about a axis which makes 135 degree angle with posivive X axis
        """
        if radius_outer is None and point_outer is None:
            raise ValueError("Either 'radius_outer' or 'point_outer' must be provided.")
        if radius_inner is None and point_inner is None:
            raise ValueError("Either 'radius_inner' or 'point_inner' must be provided.")
        if radius_outer is None:
            radius_outer = Point(center_outer).distance(Point(point_outer))
        if radius_inner is None:
            radius_inner = Point(center_inner).distance(Point(point_inner))

        outer_circle = Point(center_outer).buffer(radius_outer)
        inner_circle = Point(center_inner).buffer(radius_inner)
        annular_region = outer_circle.difference(inner_circle)

        if axis == 'y':
            angle_offset = 90
        elif axis == 'x':
            angle_offset = 0
            if alpha is not None:
                angle_offset = alpha
        else:
            raise ValueError("Axis must be 'x', 'y', or 'alpha' must be provided.")

        theta_rad = np.radians(theta)
        half_theta_rad = theta_rad / 2

        arc_coords = [(0, 0)]
        for angle in np.linspace(-half_theta_rad, half_theta_rad, 100):
            x = radius_outer * np.cos(angle)
            y = radius_outer * np.sin(angle)
            arc_coords.append((x, y))
        arc_coords.append((0, 0))
        sector = Polygon(arc_coords)

        sector = rotate(sector, angle_offset, origin=(0, 0))
        annular_arc = annular_region.intersection(sector)

        return {'polygon': annular_arc, 'name': name}

    @staticmethod
    def annular_region_arc_WithRoundedEnds(center_outer, radius_outer=None, point_outer=None, center_inner=None, radius_inner=None, point_inner=None, theta=0, axis='x', alpha=None, name='AnnularRegionArcWithRoundedEnds'):
        """
        Create an annular region arc with smoothly closed ends using circular arcs.
        :no-index:
        
        Parameters:
        - center_outer, center_inner: Centers of the outer and inner circles.
        - radius_outer, radius_inner: Radii of the outer and inner circles (optional if point_outer and point_inner are provided).
        - point_outer, point_inner: Points on the outer and inner circles (optional if radius_outer and radius_inner are provided).
        - theta: Angular extent of the arc, in degrees.
        - axis: 'x' or 'y', denoting the symmetry axis.
        - alpha: Angle in degrees of the axis of symmetry with respect to the positive x-axis.
        - name: Optional name for the geometry.
        
        Returns:
        - A dictionary containing the polygon and its name.

        Example usage:
        - annular_arc_rounded1 = Geometry.annular_region_arc_WithRoundedEnds((0, 0), radius_outer=10, center_inner=(0, 0), radius_inner=5, theta=240, axis='y', name='RoundedArcByRadiusY') # Creates a 240 degree annular arc With center and radius and symmetric about Y axis
        
        - annular_arc_rounded2 = Geometry.annular_region_arc_WithRoundedEnds((0, 0), point_outer=(7, 7), center_inner=(0, 0), point_inner=(3, 4), theta=240, axis='x', name='RoundedArcByPointX') # Creates a 240 degree annular arc With center and a point on the perimeter and symmetric about Y axis

        - annular_arc_rounded3 = Geometry.annular_region_arc_WithRoundedEnds((0, 0), radius_outer=10, center_inner=(0, -1), radius_inner=5, theta=180, alpha=-45, name='RoundedArcByRadiusAlpha') # Creates a 240 degree annular arc With center and a radius and symmetric about a axis which makes -45 degree angle with posivive X axis
        """
        if radius_outer is None and point_outer is None:
            raise ValueError("Either 'radius_outer' or 'point_outer' must be provided.")
        if radius_inner is None and point_inner is None:
            raise ValueError("Either 'radius_inner' or 'point_inner' must be provided.")
        if radius_outer is None:
            radius_outer = Point(center_outer).distance(Point(point_outer))
        if radius_inner is None:
            radius_inner = Point(center_inner).distance(Point(point_inner))

        # Create the full annular region
        outer_circle = Point(center_outer).buffer(radius_outer)
        inner_circle = Point(center_inner).buffer(radius_inner)
        annular_region = outer_circle.difference(inner_circle)

        # Adjust the rotation based on the axis
        if axis == 'y':
            angle_offset = 90
        elif axis == 'x':
            angle_offset = 0
            if alpha is not None:
                angle_offset = alpha
        else:
            raise ValueError("Axis must be 'x', 'y', or 'alpha' must be provided.")

        # Define the sector mask
        start_angle = -theta / 2 + angle_offset
        end_angle = theta / 2 + angle_offset
        sector_mask = Polygon([
            center_outer,
            *[(center_outer[0] + np.cos(np.radians(ang)) * radius_outer * 2,
               center_outer[1] + np.sin(np.radians(ang)) * radius_outer * 2)
              for ang in np.linspace(start_angle, end_angle, num=100)],
            center_outer
        ])

        # Create the annular arc by intersecting the annular region with the sector mask
        annular_arc = annular_region.intersection(sector_mask)

        # Calculate the endpoints of the cuts
        start_cut_angle = np.radians(start_angle)
        end_cut_angle = np.radians(end_angle)

        outer_start_cut = (center_outer[0] + radius_outer * np.cos(start_cut_angle),
                           center_outer[1] + radius_outer * np.sin(start_cut_angle))
        outer_end_cut = (center_outer[0] + radius_outer * np.cos(end_cut_angle),
                         center_outer[1] + radius_outer * np.sin(end_cut_angle))

        inner_start_cut = (center_inner[0] + radius_inner * np.cos(start_cut_angle),
                           center_inner[1] + radius_inner * np.sin(start_cut_angle))
        inner_end_cut = (center_inner[0] + radius_inner * np.cos(end_cut_angle),
                         center_inner[1] + radius_inner * np.sin(end_cut_angle))

        # Create circles with the endpoints as diameters
        def create_circle_with_diameter(p1, p2):
            center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            radius = np.linalg.norm(np.array(p1) - np.array(p2)) / 2
            return Point(center).buffer(radius)

        outer_tangent_circle = create_circle_with_diameter(outer_start_cut, inner_start_cut)
        inner_tangent_circle = create_circle_with_diameter(outer_end_cut, inner_end_cut)

        # Combine the annular arc with the tangent arcs
        final_shape = unary_union([annular_arc, outer_tangent_circle, inner_tangent_circle])

        return {'polygon': final_shape, 'name': name}
    
    @staticmethod
    def ellipse(center, semi_major_axis, semi_minor_axis, rotation=0, name='Ellipse'):
        """
        Create an elliptical region.
        :no-index:
        
        Parameters:
        - center: Tuple (x, y) for the center of the ellipse.
        - semi_major_axis: Length of the semi-major axis.
        - semi_minor_axis: Length of the semi-minor axis.
        - rotation: Rotation angle in degrees, counterclockwise.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - ellipse = Geometry.ellipse(center=(0, 0), semi_major_axis=7, semi_minor_axis=4, rotation=30, name='Ellipse')
        """
        ellipse = Point(center).buffer(1)
        ellipse = scale(ellipse, xfact=semi_major_axis, yfact=semi_minor_axis, origin=center)
        ellipse = rotate(ellipse, rotation, origin=center)
        return {'polygon': ellipse, 'name': name}

    @staticmethod
    def annular_ellipse(center, semi_major_axis_outer, semi_minor_axis_outer, semi_major_axis_inner, semi_minor_axis_inner, rotation=0, name='AnnularEllipse'):
        """
        Create an annular elliptical region.
        :no-index:
        
        Parameters:
        - center: Tuple (x, y) representing the center of the ellipses.
        - semi_major_axis_outer: Semi-major axis of the outer ellipse.
        - semi_minor_axis_outer: Semi-minor axis of the outer ellipse.
        - semi_major_axis_inner: Semi-major axis of the inner ellipse.
        - semi_minor_axis_inner: Semi-minor axis of the inner ellipse.
        - rotation: Rotation angle in degrees, counterclockwise.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - annular_ellipse = Geometry.annular_ellipse(center=(0, 0), semi_major_axis_outer=10, semi_minor_axis_outer=6, semi_major_axis_inner=7, semi_minor_axis_inner=4, rotation=30, name='AnnularEllipse')
        """
        outer_ellipse = Point(center).buffer(1)
        outer_ellipse = scale(outer_ellipse, xfact=semi_major_axis_outer, yfact=semi_minor_axis_outer, origin=center)
        outer_ellipse = rotate(outer_ellipse, rotation, origin=center)
        
        inner_ellipse = Point(center).buffer(1)
        inner_ellipse = scale(inner_ellipse, xfact=semi_major_axis_inner, yfact=semi_minor_axis_inner, origin=center)
        inner_ellipse = rotate(inner_ellipse, rotation, origin=center)
        
        annular_ellipse = outer_ellipse.difference(inner_ellipse)
        return {'polygon': annular_ellipse, 'name': name}

    @staticmethod
    def rectangle(width, height, name='Rectangle'):
        """
        Create a rectangular region using width and height.
        :no-index:
        
        Parameters:
        - width: Width of the rectangle.
        - height: Height of the rectangle.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - rectangle = Geometry.rectangle(width=10, height=5, name='Rectangle')
        """
        rectangle = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        return {'polygon': rectangle, 'name': name}

    @staticmethod
    def rectangle_corners(point1, point2, name='RectangleCorners'):
        """
        Create a rectangular region using two opposite corners.
        :no-index:
        
        Parameters:
        - point1: Tuple (x, y) for one corner of the rectangle.
        - point2: Tuple (x, y) for the opposite corner of the rectangle.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - rectangle_corners = Geometry.rectangle_corners(point1=(0, 0), point2=(10, 5), name='RectangleCorners')
        """
        x1, y1 = point1
        x2, y2 = point2
        rectangle = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        return {'polygon': rectangle, 'name': name}

    @staticmethod
    def square(center, side_length, name='Square'):
        """
        Create a square region given the center and side length.
        :no-index:
        
        Parameters:
        - center: Tuple (x, y) representing the center of the square.
        - side_length: Length of the sides of the square.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - square = Geometry.square(center=(0, 0), side_length=6, name='Square')
        """
        half_side = side_length / 2
        square = Polygon([
            (center[0] - half_side, center[1] - half_side),
            (center[0] + half_side, center[1] - half_side),
            (center[0] + half_side, center[1] + half_side),
            (center[0] - half_side, center[1] + half_side)
        ])
        return {'polygon': square, 'name': name}

    @staticmethod
    def regular_polygon(center, radius, num_sides, name='RegularPolygon'):
        """
        Create a regular polygon with a given number of sides.
        :no-index:
        
        Parameters:
        - center: Tuple (x, y) representing the center of the polygon.
        - radius: Radius of the circumscribed circle.
        - num_sides: Number of sides of the polygon.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - n = 18
        - n_sided_polygon = Geometry.regular_polygon(center=(0, 0), radius=5, num_sides=n, name='Pentagon')
        """
        cx, cy = center
        vertices = [
            (cx + radius * math.cos(2 * math.pi * i / num_sides), cy + radius * math.sin(2 * math.pi * i / num_sides))
            for i in range(num_sides)
        ]
        regular_polygon = Polygon(vertices)
        return {'polygon': regular_polygon, 'name': name}

    @staticmethod
    def triangle(point1, point2, point3, name='Triangle'):
        """
        Create a triangular region using three vertices.
        :no-index:
        
        Parameters:
        - point1: Tuple (x, y) for the first vertex of the triangle.
        - point2: Tuple (x, y) for the second vertex of the triangle.
        - point3: Tuple (x, y) for the third vertex of the triangle.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - trinangle = Geometry.triangle(point1=(1,2), point2=(5,7,2), point3=(9,8), name='Triangle')
        """
        triangle = Polygon([point1, point2, point3])
        return {'polygon': triangle, 'name': name}

    @staticmethod
    def pentagon(center, radius, name='Pentagon'):
        """
        Create a pentagonal region.
        :no-index:
        
        Parameters:
        - center: Tuple (x, y) representing the center of the pentagon.
        - radius: Radius of the circumscribed circle.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - pentagon = Geometry.pentagon(center=(0, 0), radius=5, name='Hexagon')
        """
        return Geometry.regular_polygon(center, radius, 5, name)

    @staticmethod
    def hexagon(center, radius, name='Hexagon'):
        """
        Create a hexagonal region.
        :no-index:
        
        Parameters:
        - center: Tuple (x, y) representing the center of the hexagon.
        - radius: Radius of the circumscribed circle.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - hexagon = Geometry.hexagon(center=(0, 0), radius=5, name='Hexagon')
        """
        return Geometry.regular_polygon(center, radius, 6, name)

    @staticmethod
    def octagon(center, radius, name='Octagon'):
        """
        Create an octagonal region.
        :no-index:
        
        Parameters:
        - center: Tuple (x, y) representing the center of the octagon.
        - radius: Radius of the circumscribed circle.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - octagon = Geometry.octagon(center=(0, 0), radius=5, name='Octagon')
        """
        return Geometry.regular_polygon(center, radius, 8, name)

    @staticmethod
    def star(center, outer_radius, inner_radius, num_points, name='Star'):
        """
        Create a star-shaped polygon.
        :no-index:
        
        Parameters:
        - center: Tuple (x, y) representing the center of the star.
        - outer_radius: Radius of the outer vertices.
        - inner_radius: Radius of the inner vertices.
        - num_points: Number of points (or tips) of the star.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - star = Geometry.star(center=(0, 0), outer_radius=7, inner_radius=3, num_points=5, name='Star')
        """
        angle = math.pi / num_points
        points = []
        for i in range(2 * num_points):
            r = outer_radius if i % 2 == 0 else inner_radius
            x = center[0] + r * math.cos(i * angle)
            y = center[1] + r * math.sin(i * angle)
            points.append((x, y))
        star = Polygon(points)
        return {'polygon': star, 'name': name}

    @staticmethod
    def trapezoid(base1, base2, height, name='Trapezoid'):
        """
        Create a trapezoidal region given the lengths of the bases and height.
        :no-index:
        
        Parameters:
        - base1: Length of the first base.
        - base2: Length of the second base.
        - height: Height of the trapezoid.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - trapezoid = Geometry.trapezoid(base1=8, base2=5, height=4, name='Trapezoid')
        """
        half_diff = abs(base1 - base2) / 2
        if base1 > base2:
            trapezoid = Polygon([(0, 0), (base1, 0), (base2 + half_diff, height), (half_diff, height)])
        else:
            trapezoid = Polygon([(0, 0), (base1, 0), (base2 - half_diff, height), (-half_diff, height)])
        return {'polygon': trapezoid, 'name': name}

    @staticmethod
    def parallelogram(base, side_length, height, name='Parallelogram'):
        """
        Create a parallelogram given the base, side length, and height.
        :no-index:
        
        Parameters:
        - base: Length of the base.
        - side_length: Length of the side.
        - height: Height of the parallelogram.
        - name: Name of the geometry.
        
        Returns:
        - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.

        Example usage:
        - parallelogram = Geometry.parallelogram(base=10, side_length=6, height=4, name='Parallelogram')
        """
        parallelogram = Polygon([(0, 0), (base, 0), (base + side_length, height), (side_length, height)])
        return {'polygon': parallelogram, 'name': name}
    
    @staticmethod
    def rectangle_with_circular_hole(rect_point1, rect_point2, circle_center, circle_radius=None, circle_point=None, name='RectangleWithCircularHole'):
        """
        Create a rectangular region with a circular hole or cut.
        :no-index:
        
        Creates a geometry where a circular region is subtracted from a rectangular region.
        The result depends on the relative positions of the rectangle and circle:
        
        * If the circle is completely inside the rectangle, it creates a circular hole
        * If the circle intersects the rectangle, it cuts out the intersecting portion
        * If the circle is completely outside the rectangle, it returns the original rectangle
        
        Parameters
        ----------
        rect_point1 : tuple
            (x, y) coordinates for the first corner of the rectangle
        rect_point2 : tuple
            (x, y) coordinates for the opposite corner of the rectangle
        circle_center : tuple
            (x, y) coordinates representing the center of the circle
        circle_radius : float, optional
            Radius of the circle (required if circle_point is not provided)
        circle_point : tuple, optional
            (x, y) coordinates of a point on the circle (required if circle_radius is not provided)
        name : str, optional
            Name of the geometry (default: 'RectangleWithCircularHole')
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'polygon': The Shapely polygon object
            - 'name': The name of the geometry
        
        Examples
        --------
        >>> rect_with_hole = Geometry.rectangle_with_circular_hole(
        ...     rect_point1=(0, 0),
        ...     rect_point2=(10, 5),
        ...     circle_center=(5, 2.5),
        ...     circle_radius=1,
        ...     name='RectWithCentralHole'
        ... )
        """
        # First, create the rectangle using the existing method
        rectangle_geom = Geometry.rectangle_corners(rect_point1, rect_point2)
        rectangle = rectangle_geom['polygon']
        
        # Create the circle using the existing method
        circle_geom = Geometry.circle(circle_center, radius=circle_radius, point=circle_point)
        circle = circle_geom['polygon']
        
        # Check if the circle intersects with the rectangle
        if not circle.intersects(rectangle):
            # If the circle is completely outside the rectangle
            print("Note: The circular region does not intersect with the rectangular region. Returning the original rectangle.")
            return {'polygon': rectangle, 'name': name}
        
        # Get the minimum and maximum coordinates of both shapes
        rect_minx, rect_miny, rect_maxx, rect_maxy = rectangle.bounds
        circ_minx, circ_miny, circ_maxx, circ_maxy = circle.bounds
        
        # Check if the circle is completely inside the rectangle
        circle_inside = (
            circ_minx >= rect_minx and
            circ_miny >= rect_miny and
            circ_maxx <= rect_maxx and
            circ_maxy <= rect_maxy
        )
        
        if circle_inside:
            print("Note: The circular region is completely inside the rectangular region. Creating a hole.")
        else:
            print("Note: The circular region intersects with the rectangular region. Cutting out the intersecting portion.")
        
        # Subtract the circle from the rectangle
        result = rectangle.difference(circle)
        
        return {'polygon': result, 'name': name}
