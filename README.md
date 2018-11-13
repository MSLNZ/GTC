# GTC

[![rtd badge][]](https://gtc.readthedocs.io/en/latest/)
[![travis shield][]](https://travis-ci.org/MSLNZ/GTC)
[![appveyor shield][]](https://ci.appveyor.com/project/jborbely/gtc/branch/develop)

The GUM Tree Calculator is a Python package for processing data with measurement uncertainty.

Python objects, called uncertain numbers, are used to encapsulate information about measured quantities. Calculations of derived quantities that involve uncertain numbers will propagate this information automatically. So, data processing results are always accompanied by uncertainties. 

GTC follows international guidelines on the evaluation of measurement data and measurement uncertainty (the GUM `<https://www.bipm.org/utils/common/documents/jcgm/JCGM_100_2008_E.pdf>`_). It has been developed for use in the context of metrology, test and calibration work.

Example: an electrical circuit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose the DC electrical current :math:`I` flowing in an circuit and the voltage :math:`V` across a circuit element have both been measured. 

The values obtained were :math:`x_V = 0.1\, \mathrm{V}` and :math:`x_I = 15\,\mathrm{mA}`> These values have the associated standard uncertainties :math:`u(x_V) = 1\, \mathrm{mV}` and :math:`u(x_I) = 0.5\,\mathrm{mA}`, respectively. 

Uncertain numbers for :math:`V` and :math:`I` can be defined using the function :func:`ureal()` ::

	>>> V = ureal(0.1,1E-3)
	>>> I = ureal(15E-3,0.5E-3)

The resistance of the circuit element :math:`R` can then be calculated directly using Ohm's law ::

    >>> R = V/I
    >>> print(R)
    6.67(23)
    
The uncertain number ``R`` represents the quantity :math:`R`. The value :math:`6.67 \,\Omega` is an estimate (or approximation) of :math:`R`. The standard uncertainty associated with this value, is :math:`0.23 \,\Omega`.

Installation
============


Documentation
=============

[rtd badge]: https://readthedocs.org/projects/gtc/badge/
[travis shield]: https://img.shields.io/travis/MSLNZ/GTC/develop.svg?label=Travis-CI
[appveyor shield]: https://img.shields.io/appveyor/ci/jborbely/gtc/develop.svg?label=AppVeyor
