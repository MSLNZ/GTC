=============
Release Notes
=============

Version 1.1.0 (????.??.??)
==========================

    Inclusion of an array-like class to hold collections of uncertain numbers. The new ``UncertainArray`` class is based on :class:`numpy.ndarray`, which provides excellent support 
    for manipulating stored data. Standard mathematical operations can be applied to ``UncertainArray`` objects. This vectorisation offers a succinct way of expressing repetitive operations 
    but it does not provide a significant speed advantage over Python iteration.
    
    The standard mathematical functions (``sin``, ``sqrt``, etc) defined by **GTC** now accept Python numbers as arguments, as well as uncertain number types and ``UncertainArray``.

Version 1.0.0 (2018.11.16)
==========================

    The initial release of the Python code version of the GUM Tree Calculator.
    
    The source code was derived from the stand-alone GUM Tree Calculator version 0.9.11, which is available from the MSL `web site <https://www.measurement.govt.nz/resources>`_ . The new version has made some significant changes to the data structures used, with accompanying changes to the underlying algorithms. 
    
    The application programmer interface in GTC 1.0.0 remains very close to that provided in GTC 0.9.11, although not all functions in GTC 0.9.11 are available yet. It is our intention to provide the remainder in forthcoming releases.  
    
    The most significant change has been to the method of storing uncertain numbers. The ``archive`` module in GTC 0.9.11 was replaced in GTC 1.0.0 by the ``persistence`` module. So, archives created using GTC 0.9.11 are not interchangeable with GTC 1.0.0. 
    
    
    
    
    
    

