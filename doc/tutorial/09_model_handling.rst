.. centered:: Last updated on *February 23rd 2018*.

Earth Model Handling
--------------------

**LASIF** has a relatively flexible way of handling Earth Models. It provides
a directory structure for the models to be organized via iterations but this
is optional. It is very important to keep in mind that **LASIF** will always
look for the Earth Model referenced in the *lasif_config.toml* file. So before
every generate_input_files command (explain later) you need to be sure that
the configuration file references the model that you plan to use for the
simulations of this iteration.
**LASIF** expects that your Earth Model mesh files are stored in a format
called exodus. These files can be queried and modified via the pyexodus
python package. We recommend using the pymesher which we demonstrate how works
earlier in this tutorial.

For visualization of your Earth model we recommend using `Paraview <https://www.paraview.org/>`_.
You can then visualize your model and the parameters it contains in a
relatively easy way. Paraview is a powerful tool that can take a bit of
practise to learn how to properly use. But it is fully worth it.

Again we want to make clear that when running simulations the user is
responsible for referencing the correct Earth Model.
