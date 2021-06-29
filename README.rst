===
GTC
===

|docs| |github tests| |pypi| |zenodo|

The GUM Tree Calculator is a Python package for processing data with measurement uncertainty.

Python objects, called uncertain numbers, are used to encapsulate information about measured
quantities. Calculations of derived quantities that involve uncertain numbers will propagate this
information automatically. So, data processing results are always accompanied by uncertainties. 

GTC follows international guidelines on the evaluation of measurement data and measurement
uncertainty (the so-called `GUM <https://www.bipm.org/utils/common/documents/jcgm/JCGM_100_2008_E.pdf>`_).
It has been developed for use in the context of metrology, test and calibration work.

Example: an electrical circuit
==============================

Suppose the DC electrical current flowing in a circuit and the voltage across a circuit
element have both been measured. 

The values obtained were 0.1 V, for the voltage, and 15 mA for the current. These values have
the associated standard uncertainties 0.001 V and 0.5 mA, respectively. 

Uncertain numbers for voltage and current can be defined using the function `ureal()` 

.. code-block:: pycon

   >>> V = ureal(0.1,1E-3)
   >>> I = ureal(15E-3,0.5E-3)

The resistance of the circuit element can then be calculated directly using Ohm's law

.. code-block:: pycon

   >>> R = V/I
   >>> print(R)
   6.67(23)
    
The uncertain number `R` represents the resistance of the circuit element. The value 6.67 ohm
is an estimate (or approximation) of the actual resistance. The standard uncertainty associated
with this value, is 0.23 ohm.

Installation
============

**GTC** is available as a `PyPI package <https://pypi.org/project/GTC/>`_. It can be installed
using pip

.. code-block:: console

   pip install gtc

Dependencies
------------
* Python 2.7, 3.5+
* `scipy <https://www.scipy.org/>`_

Documentation
=============

The documentation for **GTC** can be found `here <https://gtc.readthedocs.io/en/stable/>`_.

.. |docs| image:: https://readthedocs.org/projects/gtc/badge/?version=latest
    :target: https://gtc.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |github tests| image:: https://github.com/MSLNZ/GTC/actions/workflows/run-tests.yml/badge.svg
   :target: https://github.com/MSLNZ/GTC/actions/workflows/run-tests.yml

.. |pypi| image:: https://badge.fury.io/py/GTC.svg
    :target: https://badge.fury.io/py/GTC

.. |zenodo| image:: https://zenodo.org/badge/147150740.svg
   :target: https://zenodo.org/badge/latestdoi/147150740
