.. polygen documentation master file, created by
   sphinx-quickstart on Thu Jan  9 22:12:03 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PolyGen: An Efficient Framework for Polycrystal Generation and Cohesive Zone Modeling in Arbitrary Domains
===============================================================================================================================

This project aims to generalize polycrystal generation and polygonal meshing in any arbitrary 2D domains through *Constrained Voronoi Tessellation* for Finite Element Method (FEM) and Cohesive Zone Modeling (CZM). **polygen** also provides the functionality to adjust polygonal mesh to insert finite-thickness cohesive zone and offers efficient datastructures to integrate with Abaqus CAE. Additionally, this package also offers an excellent tool for triangular meshing of complex 2D domains.

Features
--------

* Introduces a comprehensive framework for generating polycrystalline grain structures in arbitrary 2D domains
* Enables seamless insertion of both zero-thickness and finite-thickness cohesive zones for interface modeling
* Provides efficient mesh optimization techniques
* Provides efficient triangular meshing framework for complex domains
* Enable high-quality visualisation and saving of meshfiles into various formats supported by `meshio <https://github.com/nschloe/meshio>`_

Installation
------------

You can install polygen using pip:

.. code-block:: bash

   pip install polygen

Documentation
-------------

.. toctree::
   :maxdepth: 2

   modules

Indices and Tables
=========================================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
