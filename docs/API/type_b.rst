.. _type_b:

=============================
Evaluating type-B uncertainty
=============================

The shorter name ``tb`` has been defined as an alias for :mod:`type_b`, to resolve the
names of objects in this module.

.. automodule:: type_b

.. autofunction:: type_b.mean

.. autofunction:: type_b.line_fit
.. autofunction:: type_b.line_fit_wls
.. autofunction:: type_b.line_fit_wtls

.. autoclass:: type_b.LineFitOLS
    :members: N, a_b, intercept, slope, ssr
    :inherited-members: x_from_y, y_from_x

.. autoclass:: type_b.LineFitWLS
    :members: N, a_b, intercept, slope, ssr
    :inherited-members: x_from_y,y_from_x

.. autoclass:: type_b.LineFitWTLS
    :members: N, a_b, intercept, slope, ssr
    :inherited-members: x_from_y,y_from_x

.. autofunction:: type_b.uniform
.. autofunction:: type_b.triangular
.. autofunction:: type_b.u_shaped
.. autofunction:: type_b.arcsine
.. autofunction:: type_b.uniform_ring
.. autofunction:: type_b.uniform_disk
.. autofunction:: type_b.unknown_phase_product
    
.. autodata:: type_b.distribution
    :annotation:
