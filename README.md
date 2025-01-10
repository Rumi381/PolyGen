# VoroMesh: An Efficient Framework for Polycrystal Generation and Cohesive Zone Modeling in Arbitrary Domains

This project aims to generalize polycrystal generation and polygonal meshing in any arbitrary 2D domains through `Constrained Voronoi Tessellation` for Finite Element Method (FEM) and Cohesive Zone Modeling (CZM). `VoroMesh` also provides the functionality to adjust polygonal mesh to insert finite-thickness cohesive zone and offers efficient datastructures to integrate with Abaqus CAE. Additionally, this package also offers an excellent tool for triangular meshing of complex 2D domains.

## Features

- Introduces a comprehensive framework for generating polycrystalline grain structures in arbitrary 2D domains.
- Enables seamless insertion of both zero-thickness and finite-thickness cohesive zones for interface modeling.
- Provides efficient datastructures for integration with Abaqus CAE
- Provides efficient mesh optimization techniques.
- Provides efficient triangular meshing framework for complex domains.
- Enable high-quality visualisation and saving of meshfiles into various formats supported by `meshio`.

## Installation

To install the package, use `pip`:

```bash
pip install voronoi_meshing
```

## Accessing Documentation

To access the documentation for any function, class, or method in the project, please visit the official documentation page of VoroMesh [View the Docs](https://Rumi381.github.io/voromesh/)

## Example Usage

Check out the detailed examples and usages of different functionalities in the [VoroMesh Jupyter Notebook](examples/voromesh.ipynb).

## Running `VoroMesh` from the command line
If you have set up the input file with the desired functionalites, you can run the following command in the terminal (assuming your input file is in the `examples` directory inside your working directory):

```sh
voromesh -d2 ./examples/input.in
```

### Example Input File (`input.in`)

The `input.in` file should contain the necessary inputs for running the desired functions. The file should follow a simple key-value format, with each parameter on a new line. The parameters can be provided in any order. 

#### Parameters

- **boundary** : str or dict
    The path to the `mesio` supported mesh file defining the boundary or predefined geometry object provided by `VoroMesh`
- **N_points** : int
    Number of initial seed points for the Voronoi diagram
- **points_seed** : int, optional
    Random seed for generating initial Poisson disk sampling points (default: 42)
- **margin** : float, optional
    Margin to add around the boundary for generating seed points (default: 0.01)
- **N_iter** : int, optional
    Number of iterations for Lloyd's algorithm to relax the points (default: 100000)
- **use_decay** : bool, optional
    Whether to use decay in Lloyd's algorithm (default: True)
- **edgeLength_threshold** : float, optional
    Length threshold for collapsing short edges in the Voronoi diagram (default: None)
- **cohesiveThickness** : float, optional
    Thickness of the cohesive zone for adjusting the Voronoi cells (default: None)
- **triangularMesh_minLength** : float, optional
    Minimum edge length for triangular meshing (default: None)
- **visualize** : bool, optional
    Whether to show visualization plots (default: False)
- **saveData** : bool, optional
    Whether to save the resulting data structures to files (default: True)

Here is an example of how to organize the `input.in` file:

```ini
# This is a comment
#boundary = Geometry.annular_region((0.0, -0.075), point_outer=(0.0, 1.49), center_inner=(0.0, -0.425), point_inner=(0.0, 0.5), name='Calcite_region360d')
boundary = ./examples/CalciteBoundaryFromPNG.obj
N_points = 1000
points_seed = 7845
margin = 0.01
N_iter = 10000000
use_decay = True
edgeLength_threshold = 6.3
cohesiveThickness = 0.02
triangularMesh_minLength = 15
visualize = True
saveData = True
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Acknowledgements

Special thanks to all contributors and the open-source community for their valuable work and support.

---

Feel free to modify and expand upon this draft to better suit your project's needs and details. This structure should give users a clear understanding of the project and how to use the various scripts.

---