Align
=====

Calibrating the grazing incidence angle is a critical step for
GIXS experiments :cite:`mythesis, align`. The align module has tools for
assisting in this calibration process based on :cite:`align`.

Importing the align module:

>>> import gixstools.align

Functions
---------

Macros for *spec* :cite:`spec` can be generated with
``gixstools.align.create_z_macro`` and ``gixstools.align.create_om_macro``.
These can then be called in *spec* to perform a spatially
resolved scan :cite:`align`.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   gixstools.align.create_z_macro
   gixstools.align.create_om_macro

   
.. currentmodule:: gixstools.align

.. autofunction:: create_z_macro
.. autofunction:: create_om_macro

.. bibliography::
    :style: plain