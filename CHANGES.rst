=============
Release Notes
=============

Version 1.3.4 (in development)
==============================

    * :func:`reporting.budget` now expects explicit keyword arguments for all options, instead of positional arguments (the names of the previous positional arguments are now the keywords).
    * :func:`reporting.budget` takes a new key word ``intermediate``

Version 1.3.3 (2021.02.16)
==========================

    * Fixed an issue with merging uncertain numbers. The function :func:`type_a.merge` now has a tolerance parameter, which is used to determine whether the arguments ``a`` and ``b`` have equivalent values.

Version 1.3.2 (2020.09.16)
==========================

    * Fixed an issue with restoration of archived uncertain numbers. A `RuntimeError` arose if two uncertain numbers, originally created in the same context, were restored to different archive objects in a new common context.
    
    * An attempt to create a file or string representation of an empty archive raises a `RuntimeError`

    * Docstrings for :meth:`~.Archive.add` and :meth:`~.Archive.extract` now mention the option of using the name as a look-up key (like a mapping) 
    
Version 1.3.1 (2020.08.21)
==========================

    * Fixed an issue with the `r` attribute of uncertain complex numbers, which returns the correlation coefficient between real and imaginary components: the calculation was incorrect (however, :func:`core.get_correlation` gave the correct result).
    
    * Fixed an issue with the calculation of the variance-covariance matrix for an uncertain complex number with finite degrees of freedom: the matrix element for the variance of the real component was sometimes incorrectly returned for the variance of the imaginary component as well.

Version 1.3.0 (2020.07.28)
==========================

    * Added support to :mod:`persistence` for archive storage in a JSON format. The new functions are: :func:`persistence.dump_json`, :func:`persistence.dumps_json`, :func:`persistence.load_json` and :func:`persistence.loads_json`
    
Version 1.2.1 (2020.04.01)
==========================

    * Fixed issue `#18 <https://github.com/MSLNZ/GTC/issues/18>`_ - calculate the inverse of a matrix with uncertain elements 
    
    * Revised the documentation for the :mod:`persistence` module 

Version 1.2.0 (2019.10.16)
==========================

    * Functions to perform straight-line regressions are included in modules :mod:`type_a` and :mod:`type_b`. 
    
    * The regression functions in :mod:`type_a` act on sequences of numerical data in the conventional sense (i.e., only the values of data are used; if the data include uncertain number objects, the associated uncertainty is ignored). The residuals are evaluated and may contribute to the uncertainty of the results obtained, depending on the regression method. 
    
    * The regression functions in :mod:`type_b` act on sequences of uncertain-numbers, propagating uncertainty into the results obtained. In most cases, the regression functions in this module are paired with a function of the same name in :mod:`type_a`. For example, :func:`type_a.line_fit` and :func:`type_b.line_fit` both perform an ordinary least-squares regression. The uncertain-numbers for the intercept and slope obtained from :func:`type_a.line_fit` are correlated and have uncertainties that depend on the fitting residuals. On the other hand, the intercept and slope obtained by :func:`type_b.line_fit` depend on the uncertain-number data supplied, and does not take account of the residuals.
    
    * The function :func:`type_a.merge` may be used to combine results obtained from type-A and type-B regressions performed on the same data. 
    
    * A number of example calculations are included from Appendix H of the *Guide to the expression of uncertainty in measurement* (`GUM <https://www.iso.org/sites/JCGM/GUM/JCGM100/C045315e-html/C045315e.html?csnumber=50461>`_).
    
    * A number of example calculations are included from the 3rd Edition (2012) of the EURACHEM/CITAC Guide: *Quantifying Uncertainty in Analytical Measurement* (`CG4 <http://www.citac.cc/QUAM2012_P1.pdf>`_). 
    
    * There are several examples of applying GTC to linear calibration problems, including the use of regression functions in :mod:`type_a` and :mod:`type_b`.

Version 1.1.0 (2019.05.30)
==========================

    * Mathematical functions in the :mod:`core` module (``sin``, ``sqrt``, etc) can be applied to Python numbers as well as uncertain numbers (previously these functions raised an exception when applied to Python numbers).
    
    * There is a new array-like class to hold collections of uncertain numbers. :class:`~uncertain_array.UncertainArray` is based on :class:`numpy.ndarray`, which provides excellent support for manipulating stored data. Standard mathematical operations in the :mod:`core` module can be applied to :class:`~uncertain_array.UncertainArray` objects. 
    
    * A function :func:`reporting.sensitivity` calculates partial derivatives (sensitivity coefficients).

Version 1.0.0 (2018.11.16)
==========================

    The initial release of the Python code version of the GUM Tree Calculator.
    
    The source code was derived from the stand-alone GUM Tree Calculator version 0.9.11, which is available from the MSL `web site <https://www.measurement.govt.nz/resources>`_ . The new version has made some significant changes to the data structures used, with accompanying changes to the underlying algorithms. 
    
    The application programmer interface in GTC 1.0.0 remains very close to that provided in GTC 0.9.11, although not all functions in GTC 0.9.11 are available yet. It is our intention to provide the remainder in forthcoming releases.  
    
    The most significant change has been to the method of storing uncertain numbers. The ``archive`` module in GTC 0.9.11 was replaced in GTC 1.0.0 by the ``persistence`` module. So, archives created using GTC 0.9.11 are not interchangeable with GTC 1.0.0. 
    
    
    
    
    
    

