.. centered:: Last updated on *February 26th 2018*.

Meshing
=======

For making a mesh we recommend using the `pymesher <https://gitlab.com/Salvus/salvus_mesher/tree/master>`_
which is an easy to use tool to make meshes of varying scales. The mesh we
used in the tutorial was for example created like this:

.. code-block:: bash

    # Ask the mesher to give you a .yaml file to control parameters
    $ python -m pymesher.interface SphericalChunk3D --save_yaml mesh.yaml

What you did there is enter the pymesher package from the command line and
through its interface you told it that you want to make a mesh of a
section of the Earth in 3D. The mesher then outputs a file called *mesh.yaml*.

As our toy study area is in Turkey we can ask the mesher to give us the
coordinates of Turkey just to make things easy. This is of course possible
to do through other methods, but for simplicity you can run:

.. code-block:: bash

    $ python -m pymesher.getcoordinates Turkey

This should output these values: *[38.771730, 34.924965, null]*. Now we can
modify the *mesh.yaml* file so that it makes a mesh that fits our purpose.

The parameters you need to modify are:

* basic.period = 25
* spherical.min_radius = 5400.0
* chunk3D.max_colatitude1 = 10.0
* chunk3D.max_colatitude2 = 10.0
* chunk3D.euler_angles = [38.771730, 34.924965, null]

That should be enough to make a mesh identical to the one we used in the
tutorial. Now you can run:

.. code-block:: bash

    $ python -m pymesher.interface --input_file mesh.yaml

If you have already made a mesh with the same velocity model and period you
will have to add the ``--overwrite`` flag at the end and then your old mesh
will be overwritten with the new one.

For further information and help using the pymesher we refer to its `website <https://gitlab.com/Salvus/salvus_mesher/tree/master>`_.

If you do not want to learn how to make the mesh to use in the tutorial but
you still want to go through the tutorial you can copy the mesh we used and
paste it into your project folder. Then refer to its path in your config file.

.. code-block:: bash

    # Position your self in the root of your lasif project
    $ cp {lasif_code_folder}/lasif/tests/data/ExampleProject/MODELS/ITERATION_1/Turkey.e ./MODELS/.

Then you should have the same mesh ready in your project folder and can
go through the tutorial.
