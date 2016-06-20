.. centered:: Last updated on *June 19th 2016*.

Model Update
------------

**LASIF** per se does not aid you with actually updating the model. This is a
concious design decision as a large part of the research or black art aspect in
full seismic waveform inversions actually comes from the model update,
potential preconditioning, and the actually employed numerical optimization
scheme.

This parts illustrates a simple update. When running a real inversion, please
consider spending a couple of days to scripts model updates and everything else
required around a given LASIF project.


Actual model update
^^^^^^^^^^^^^^^^^^^

Because LASIF does not (yet) aid you with updating the model, you will have
to do this manually. Keep in mind that the kernels SES3D produces initially
are the raw gradients, and unsmoothed. You will probably need to smooth them
before updating the model to get reasonable results without excessive
focusing near sources and/or receivers.
