.. _numpy-uarray:

================================
Collections of Uncertain Numbers
================================

The :class:`.UncertainArray` array object is a convenient container for uncertain-number objects based on :class:`numpy.ndarray`, which provides excellent support 
for manipulating stored data (see :ref:`arrays.indexing` ). An :class:`.UncertainArray` can contain a mixture of :class:`~.lib.UncertainReal`, 
:class:`~.lib.UncertainComplex` or Python numbers (:class:`int`, :class:`float` and :class:`complex`).  
The usual mathematical operations can be applied to :class:`.UncertainArray` objects, producing an :class:`.UncertainArray` 
containing the results of the operation. For instance, if :code:`A` and :code:`B` are arrays of the same size, they can be added :code:`A + B`, subtracted :code:`A - B`, etc; or a function like :code:`sqrt(A)` applied. This vectorisation provides a succinct expression for repetitive operations 
but it does not offer a significant speed advantage over explicit Python iteration. 

The function :func:`~.core.uarray` is used to create an  :class:`.UncertainArray` array object.

.. note::

    When we need the matrix product of a pair of two-dimensional arrays, the 
    function :func:`~core.matmul` should be used (for Python 3.5 and above the 
    binary ``@`` operator is an alternative). For example::
    
        >>> a = uarray([[ureal(1,1),ureal(2,1)],[ureal(3,1),ureal(4,1)]])
    
        
    
.. _uarray-example-1:

Example 1. Creating an UncertainArray
-------------------------------------

The following example illustrates how to create an :class:`.UncertainArray` and how
to use **GTC** functions for calculation.

Import the necessary **GTC** functions and modules

.. code-block:: pycon

   >>> from GTC import ureal, uarray, cos, type_a

Next, define the uncertain arrays

.. code-block:: pycon

   >>> voltages = uarray([ureal(4.937, 0.012), ureal(5.013, 0.008), ureal(4.986, 0.014)])
   >>> currents = uarray([ureal(0.023, 0.003), ureal(0.019, 0.006), ureal(0.020, 0.004)])
   >>> phases = uarray([ureal(1.0442, 2e-4), ureal(1.0438, 5e-4), ureal(1.0441, 3e-4)])

We can use the :func:`~.core.cos` function to calculate the AC resistances

.. code-block:: pycon

   >>> resistances = (voltages / currents) * cos(phases)
   >>> resistances
   UncertainArray([ureal(107.88283143147648,14.07416562378944,inf),
                   ureal(132.69660967977737,41.90488273081293,inf),
                   ureal(125.3181626494936,25.06618583901181,inf)],
                  dtype=object)

Now, to calculate the average AC resistance we could use :func:`.type_a.mean`, which evaluates the mean of the uncertain number values 

.. code-block:: pycon

   >>> type_a.mean(resistances)
   121.96586792024915

However, that is a real number, not an uncertain number. We have discarded all information about the uncertainty of each resistance!

A better calculation in this case uses :func:`.function.mean`, which will propagate uncertainties 

.. code-block:: pycon

   >>> fn.mean(resistances)
   ureal(121.96586792024915,16.939155846751817,inf)

This obtains an uncertain number with a standard uncertainty of 16.939155846751817, which is the combined uncertainty of the mean of AC resistance values according to the GUM calculation

.. code-block:: pycon

   >>> math.sqrt(resistances[0].u**2 + resistances[1].u**2 + resistances[2].u**2)/3.0
   16.939155846751817

.. note::

    A Type-A evaluation of the standard uncertainty of the mean of the three resistance values is a different calculation  

    .. code-block:: pycon

           >>> type_a.standard_uncertainty(resistances)
           7.356613978879885

    The standard uncertainty evaluated here by :func:`.type_a.standard_uncertainty`
    is a sample statistic calculated from the values alone. On the other hand,
    the standard uncertainty obtained by :func:`.function.mean` is evaluated by propagating 
    the input uncertainties through the calculation of the mean value. There is no reason to expect 
    these two different calculations to yield the same result.    


.. _uarray-example-2:

Example 2. Creating a Structured UncertainArray
-----------------------------------------------

One can also make use of the :ref:`structured_arrays` feature of numpy to access
columns in the array by *name* instead of by *index*.

.. note::

   numpy arrays use a zero-based indexing scheme, so the first column corresponds
   to index 0

Suppose that we have the following :class:`list` of data

.. code-block:: pycon

   >>> data = [[ureal(1, 1), ureal(2, 2), ureal(3, 3)],
   ...         [ureal(4, 4), ureal(5, 5), ureal(6, 6)],
   ...         [ureal(7, 7), ureal(8, 8), ureal(9, 9)]]

We can create an :class:`.UncertainArray` from this :class:`list`

.. code-block:: pycon

   >>> ua = uarray(data)

When ``ua`` is created it is a *view* into ``data`` (i.e., no elements in ``data``
are copied)

.. code-block:: pycon

   >>> ua[0,0] is data[0][0]
   True

However, if an element in ``ua`` is redefined to point to a new object then the
corresponding element is ``data`` does not change

.. code-block:: pycon

   >>> ua[0,0] = ureal(99, 99)
   >>> ua[0,0]
   ureal(99.0,99.0,inf)
   >>> data[0][0]
   ureal(1.0,1.0,inf)
   >>> ua[1,1] is data[1][1]
   True

If we wanted to access the data in column 1 we would use the following

.. code-block:: pycon

   >>> ua[:,1]
   UncertainArray([ureal(2.0,2.0,inf), ureal(5.0,5.0,inf),
                   ureal(8.0,8.0,inf)], dtype=object)

Alternatively, we can assign a *name* to each column so that we can access columns
by *name* rather than by an *index* number *(note that we must cast each row*
*in data to be a* :class:`tuple` *data type)*

.. code-block:: pycon

   >>> ua = uarray([tuple(row) for row in data], names=['a', 'b', 'c'])

Since we chose column 1 to have the name ``'b'`` we can now access column 1
by its *name*

.. code-block:: pycon

   >>> ua['b']
   UncertainArray([ureal(2.0,2.0,inf), ureal(5.0,5.0,inf),
                   ureal(8.0,8.0,inf)], dtype=object)

and then perform a calculation by using the *names* that were chosen

.. code-block:: pycon

   >>> ua['a'] * ua['b'] + ua['c']
   UncertainArray([ureal(5.0,4.123105625617661,inf),
                   ureal(26.0,28.91366458960192,inf),
                   ureal(65.0,79.7057087039567,inf)], dtype=object)

.. _uarray-example-3:

Example 3. Calibrating a Photodiode
-----------------------------------

Suppose that we have the task of calibrating the spectral response of a
photodiode. We perform the following steps to acquire the data and then perform
the calculation to determine the spectral response of the photodiode (PD)
relative to a calibrated reference detector (REF). The experimental procedure
is as follows:

1) Select a wavelength from the light source.
2) Move REF to be in the beam path of the light source.
3) Block the light and measure the background signal of REF.
4) Unblock the light and measure the signal of REF.
5) Move PD to be in the beam path of the light source.
6) Block the light and measure the background signal of PD.
7) Unblock the light and measure the signal of PD.
8) Repeat step (1).

10 readings were acquired in steps 3, 4, 6 and 7 and they were used determine
the average and standard deviation for each measurement. The standard deviation
is shown in brackets in the table below. The uncertainty of the wavelength is
negligible.

+------------+-----------+---------------+------------+----------------+
| Wavelength | PD Signal | PD Background | REF Signal | REF Background |
|    [nm]    |  [Volts]  |    [Volts]    |   [Volts]  |    [Volts]     |
+============+===========+===============+============+================+
|     400    |  1.273(4) |   0.0004(3)   |  3.721(2)  |   0.00002(2)   |
+------------+-----------+---------------+------------+----------------+
|     500    |  2.741(7) |   0.0006(2)   |  5.825(4)  |   0.00004(3)   |
+------------+-----------+---------------+------------+----------------+
|     600    |  2.916(3) |   0.0002(1)   |  6.015(3)  |   0.00003(1)   |
+------------+-----------+---------------+------------+----------------+
|     700    |  1.741(5) |   0.0003(4)   |  4.813(4)  |   0.00005(4)   |
+------------+-----------+---------------+------------+----------------+
|     800    |  0.442(9) |   0.0004(3)   |  1.421(2)  |   0.00003(1)   |
+------------+-----------+---------------+------------+----------------+

We can create a :class:`list` from the information in the table. It is okay to mix
built-in data types (e.g., :class:`int`, :class:`float` or
:class:`complex`) with uncertain numbers. The degrees of freedom = 10 - 1 = 9.

.. code-block:: pycon

   >>> data = [
   ...  (400., ureal(1.273, 4e-3, 9), ureal(4e-4, 3e-4, 9), ureal(3.721, 2e-3, 9), ureal(2e-5, 2e-5, 9)),
   ...  (500., ureal(2.741, 7e-3, 9), ureal(6e-4, 2e-4, 9), ureal(5.825, 4e-3, 9), ureal(4e-5, 3e-5, 9)),
   ...  (600., ureal(2.916, 3e-3, 9), ureal(2e-4, 1e-4, 9), ureal(6.015, 3e-3, 9), ureal(3e-5, 1e-5, 9)),
   ...  (700., ureal(1.741, 5e-3, 9), ureal(3e-4, 4e-4, 9), ureal(4.813, 4e-3, 9), ureal(5e-5, 4e-5, 9)),
   ...  (800., ureal(0.442, 9e-3, 9), ureal(4e-4, 3e-4, 9), ureal(1.421, 2e-3, 9), ureal(3e-5, 1e-5, 9))
   ... ]

Next, we create a *named* :class:`.UncertainArray` from ``data`` and calculate the
relative spectral response by using the *names* that were specified

.. code-block:: pycon

   >>> ua = uarray(data, names=['nm', 'pd-sig', 'pd-bg', 'ref-sig', 'ref-bg'])
   >>> res = (ua['pd-sig'] - ua['pd-bg']) / (ua['ref-sig'] - ua['ref-bg'])
   >>> res
   UncertainArray([ureal(0.342006675660713,0.0010935674325269068,9.630065079733788),
                   ureal(0.4704581662363347,0.0012448685947602906,10.30987538377716),
                   ureal(0.4847571974590064,0.0005545173836499742,13.031921586772652),
                   ureal(0.36167007760313324,0.0010846673083513545,10.620461706054874),
                   ureal(0.31077362646642787,0.006352297390618683,9.105944114389143)],
                  dtype=object)

Since ``ua`` and ``res`` are numpy arrays we can use numpy syntax to filter information. To select
the data where the PD signal is > 2 volts, we can use

.. code-block:: pycon

   >>> gt2 = ua[ ua['pd-sig'] > 2 ]
   >>> gt2
   UncertainArray([(500., ureal(2.741,0.007,9.0), ureal(0.0006,0.0002,9.0), ureal(5.825,0.004,9.0), ureal(4e-05,3e-05,9.0)),
                   (600., ureal(2.916,0.003,9.0), ureal(0.0002,0.0001,9.0), ureal(6.015,0.003,9.0), ureal(3e-05,1e-05,9.0))],
                  dtype=[('nm', '<f8'), ('pd-sig', 'O'), ('pd-bg', 'O'), ('ref-sig', 'O'), ('ref-bg', 'O')])

We can also use the *name* feature on ``gt2`` to then get the REF signal for the filtered data

.. code-block:: pycon

   >>> gt2['ref-sig']
   UncertainArray([ureal(5.825,0.004,9.0), ureal(6.015,0.003,9.0)],
                  dtype=object)

To select the relative spectral response where the wavelengths are < 700 nm

.. code-block:: pycon

   >>> res[ ua['nm'] < 700 ]
   UncertainArray([ureal(0.342006675660713,0.0010935674325269068,9.630065079733788),
                   ureal(0.4704581662363347,0.0012448685947602906,10.30987538377716),
                   ureal(0.4847571974590064,0.0005545173836499742,13.031921586772652)],
                  dtype=object)

This is a very simplified analysis. In practise one should use a
:ref:`Measurement Model <measurement_models>`.

.. _uarray-example-4:

Example 4. N-Dimensional UncertainArrays
----------------------------------------

The multi-dimensional aspect of numpy arrays is also supported.

Suppose that we want to multiply two matrices that are composed of uncertain numbers

.. math::

    C=AB\;

The :math:`A` and :math:`B` matrices are defined as

.. code-block:: pycon

   >>> A = uarray([[ureal(3.6, 0.1), ureal(1.3, 0.2), ureal(-2.5, 0.4)],
   ...             [ureal(-0.2, 0.5), ureal(3.1, 0.05), ureal(4.4, 0.1)],
   ...             [ureal(8.3, 1.5), ureal(4.2, 0.6), ureal(3.3, 0.9)]])
   >>> B = uarray([ureal(1.8, 0.3), ureal(-3.5, 0.9), ureal(0.8, 0.03)])

Using the ``@`` operator for matrix multiplication, which was introduced in
Python 3.5 (:pep:`465`), we can determine :math:`C`

.. parsed-literal::

   >>> C = A @ B  # doctest: +SKIP
   >>> C  # doctest: +SKIP
   UncertainArray([ureal(-0.0699999999999994,1.7792484368406793,inf),
                   ureal(-7.689999999999999,2.9414535522424963,inf),
                   ureal(2.8800000000000003,5.719851484085929,inf)],
                  dtype=object)

Alternatively, we could use the :func:`numpy.dot` function, which is automatically imported by **GTC**

.. code-block:: pycon

   >>> C = dot(A, B)
   >>> C
   UncertainArray([ureal(-0.0699999999999994,1.7792484368406793,inf),
                   ureal(-7.689999999999999,2.9414535522424963,inf),
                   ureal(2.8800000000000003,5.719851484085929,inf)],
                  dtype=object)
