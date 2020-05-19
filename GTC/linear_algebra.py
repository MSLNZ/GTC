"""
Classes
-------
    * :class:`.UncertainArray`

Arithmetic operations
---------------------
    Arithmetic operations are defined for arrays
    (unary ``+`` and ``-``, and binary ``+``, ``-`` and ``*``).
    The multiplication  operator ``*`` is implemented element-wise.
    For two-dimensional arrays, matrix multiplication is performed
    by :func:`.matmul` (since Python 3.5, the ``@`` operator can be used).
    Also, :func:`.dot` evaluates the array dot product, which for
    two-dimensional arrays is equivalent to matrix multiplication.

    When one argument is a scalar, it is applied to each element
    of the array in turn.

Mathematical operations
-----------------------

    The standard mathematical operations defined in :mod:`.core`
    can be applied directly to an :class:`.UncertainArray`. An
    :class:`.UncertainArray` is returned, containing the result
    of the function applied to each element.

Functions
---------
    The functions :func:`inv`, :func:`transpose`, :func:`solve`
    and :func:`det` implement the usual linear algebra operations.

    The functions :func:`identity`, :func:`empty`, :func:`zeros`
    :func:`full` and :func:`ones` create simple arrays.

Reporting functions
-------------------

    Reporting functions :func:`~.reporting.u_component` and
    :func:`~.reporting.sensitivity` can be applied directly to
    a pair of arrays. An :class:`.UncertainArray` containing
    the result of applying the function to pairs of elements
    will be returned.

    The core `GTC` function :func:`~.core.result` can be used to
    define elements of an array as intermediate uncertain numbers.

Array broadcasting
------------------
    When binary arithmetic operations are applied to arrays, the shape
    of the array may be changed for the purposes of the
    calculation. The rules are as follows:

    *   If arrays do not have the same number of dimensions, then
        dimensions of size `1` are prepended to the smaller array's
        shape

    Following this, the size of array dimensions are compared and
    checked for compatibility. Array dimensions are compatible when

    *   dimension sizes are equal, or
    *   one of the dimension sizes is `1`

    Finally, if either of the compared dimension sizes is `1`, the
    size of the larger dimension is used. For example::

        >>> x = la.uarray([1,2])
        >>> y = la.uarray([[1],[2]])
        >>> print(x.shape,y.shape)  # doctest: +SKIP
        (2,) (2, 1)
        >>> x + y
        uarray([[2, 3],
                [3, 4]])

Module contents
---------------

"""
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip

import numpy as np

from GTC import (
    LU,
    is_sequence,
)

__all__ = (
    'uarray',
    'dot',
    'matmul',
    'solve',
    'inv',
    'det',
    'identity', 'zeros', 'ones', 'empty', 'full',
    'transpose'
)


def uarray(array, label=None, names=None):
    """Create an array of uncertain numbers.

    For an overview on how to use an :class:`.UncertainArray` see :ref:`numpy-uarray`.

    .. versionadded:: 1.1

    .. attention::

       Requires numpy :math:`\geq` v1.13.0 to be installed.

    :param array: An array-like object containing :class:`int`, :class:`float`, :class:`complex`
                  :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex` elements.
    :param label: A label to assign to the `array`. This `label` does not
                  change labels previously assigned to array elements.
    :type label: str
    :param names: The field `names` to use to create a
                  :ref:`structured array <structured_arrays>`.
    :type names: list[str]

    :return: An :class:`.UncertainArray`.

    **Examples**:

        Create an `amps` and a `volts` array and then calculate the `resistances`

        >>> amps = la.uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)])
        >>> volts = la.uarray([ureal(10.3, 1.3), ureal(9.5, 0.8), ureal(12.6, 1.9)])
        >>> resistances = volts / amps
        >>> resistances
        uarray([ureal(18.070175438596493,6.145264246839438,inf),
                ureal(21.11111111111111,5.903661880050747,inf),
                ureal(18.52941176470588,5.883187720636909,inf)])

        Create a :ref:`Structured array <structured_arrays>`, with the names ``'amps'`` and ``'volts'``,
        and then calculate the `resistances`.

        >>> data = la.uarray([(ureal(0.57, 0.18), ureal(10.3, 1.3)),
        ...                (ureal(0.45, 0.12), ureal(9.5, 0.8)),
        ...                (ureal(0.68, 0.19), ureal(12.6, 1.9))], names=['amps', 'volts'])
        >>> resistances = data['volts'] / data['amps']
        >>> resistances
        uarray([ureal(18.070175438596493,6.145264246839438,inf),
                ureal(21.11111111111111,5.903661880050747,inf),
                ureal(18.52941176470588,5.883187720636909,inf)])

    """
    if np.__version__ < '1.13.0':
        # the __array_ufunc__ method was not introduced until version 1.13.0
        raise ValueError('creating an UncertainArray requires numpy >= 1.13.0')

    # don't allow a scalar UncertainArray
    if not (is_sequence(array) or (isinstance(array, np.ndarray) and array.ndim > 0)):
        raise ValueError('cannot create an UncertainArray from a scalar')

    dtype = None

    if names is not None:
        try:
            a_len = len(array[0])
            values = array[0]
        except:
            try:
                a_len = len(array)
                values = array
            except:
                a_len = None
                values = None

        if a_len is None or not isinstance(values, tuple):
            raise TypeError('The elements in the uarray must be a tuple if specifying field names')

        if a_len != len(names):
            raise ValueError('len(array[0]) != len(names) -> {} != {}'.format(a_len, len(names)))

        dtype = [(name, type(val)) for name, val in izip(names, array[0])]

    return UncertainArray(array, dtype=dtype, label=label)


def matmul(lhs, rhs):
    """Matrix product of a pair of two-dimensional arrays.

    For more details see :data:`numpy.matmul`.

    .. versionadded:: 1.1

    :param lhs: 2D array-like object.
    :param rhs: 2D array-like object.
    :return: The matrix product.
    :rtype: :class:`.UncertainArray`

    """
    # Must implement matrix multiplication because np.matmul does not
    # support dtype=object arrays in versions <= 1.15.4. Support for
    # np.matmul was added in numpy 1.16.0 as a ufunc.

    if not isinstance(lhs, np.ndarray):
        lhs = np.asarray(lhs)
    if not isinstance(rhs, np.ndarray):
        rhs = np.asarray(rhs)

    nd1, nd2 = lhs.ndim, rhs.ndim
    if nd1 == 0 or nd2 == 0:
        raise ValueError("Scalar operands are not allowed, use '*' instead")

    if nd1 <= 2 and nd2 <= 2:
        return UncertainArray(lhs.dot(rhs))

    broadcast = np.broadcast(np.empty(lhs.shape[:-2]), np.empty(rhs.shape[:-2]))
    ranges = [np.arange(s) for s in broadcast.shape]
    grid = np.meshgrid(*ranges, sparse=False, indexing='ij')
    indices = np.array([item.ravel() for item in grid]).transpose()

    i1 = indices.copy()
    i2 = indices.copy()
    for i in range(len(indices[0])):
        i1[:, i] = indices[:, i].clip(max=lhs.shape[i] - 1)
        i2[:, i] = indices[:, i].clip(max=rhs.shape[i] - 1)

    slices = np.array([[slice(None), slice(None)]]).repeat(len(indices), axis=0)
    i1 = np.hstack((i1, slices))
    i2 = np.hstack((i2, slices))
    out = np.array([matmul(lhs[tuple(a)], rhs[tuple(b)]) for a, b in izip(i1, i2)])
    return UncertainArray(out.reshape(list(broadcast.shape) + [lhs.shape[-2], rhs.shape[-1]]))


def dot(lhs, rhs):
    """Dot product of two arrays.

    For more details see :func:`numpy.dot`.

    .. versionadded:: 1.1

    :param lhs: The array-like object on the left-hand side.
    :param rhs: The array-like object on the right-hand side.
    :return: The dot product.
    :rtype: :class:`.UncertainArray`
    """
    return UncertainArray(np.dot(lhs, rhs))


def transpose(a, axes=None):
    """Array transpose

    For more details see :func:`numpy.transpose`.

    .. versionadded:: 1.1

    :param a: The array-like object
    :return: The transpose
    :rtype: :class:`.UncertainArray`

    """
    return UncertainArray(np.transpose(a, axes))


# ---------------------------------------------------------------------------
def solve(a, b):
    """Return :math:`x`, the solution of :math:`a \cdot x = b`

    .. versionadded:: 1.1

    :arg a: 2D :class:`~uncertain_array.UncertainArray`
    :arg b: :class:`~uncertain_array.UncertainArray`

    :rtype: :class:`~uncertain_array.UncertainArray`


    **Example**::

        >>> a = la.uarray([[-2,3],[-4,1]])
        >>> b = la.uarray([4,-2])
        >>> la.solve(a,b)
        uarray([1.0, 2.0])

    """
    return LU.solve(a, b)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def psolve(a, b):
    """
    Funstion solves overdetermined (more equations than variables) equation system with using Moore–Penrose pseudoinverse matrix
    Return :math:`x`, the solution of :math:`a \cdot x = b`

    .. versionadded:: 1.1

    :arg a: 2D :class:`~uncertain_array.UncertainArray`
    :arg b: :class:`~uncertain_array.UncertainArray`

    :rtype: :class:`~uncertain_array.UncertainArray`


    **Example**::

        >>> a = la.uarray([[2,1],[-3,1],[-1,1]])
        >>> b = la.uarray([[4], [-1], [0.98]])
        >>> la.psolve(a,b)
        uarray([1.0005263157894735, 1.9936842105263157])
    """
    return matmul(pinv(a),b)


# ---------------------------------------------------------------------------
def inv(a):
    """Return the (multiplicative) matrix inverse

    .. versionadded:: 1.1

    **Example**::

        >>> x = la.uarray( [[2,1],[3,4]])
        >>> x_inv =la.inv(x)
        >>> la.matmul(x,x_inv)
        uarray([[1.0, 0.0],
                [4.440892098500626e-16, 1.0]])

    """
    b = np.identity(a.shape[0], a.dtype)
    return LU.invab(a, b)


# ---------------------------------------------------------------------------
def pinv(a):
    """Return the Moore–Penrose pseudoinverse matrix

    .. versionadded:: x

    **Example**::

        >>> x = la.uarray( [[2,1],[3,4],[1,2]])
        >>> x_pinv =la.pinv(x)
    """
    # a+=(atxa)^1 x at
    at = transpose(a)
    ata = matmul(at, a)
    inv_ata = inv(ata)
    b = matmul(inv_ata, at)
    return b
# ---------------------------------------------------------------------------
def det(a):
    """Return the matrix determinant

    .. versionadded:: 1.1

    **Example**::

        >>> x = la.uarray( range(4) )
        >>> x.shape = 2,2
        >>> print(x)
        [[0 1]
        [2 3]]
        >>> la.det(x)
        -2.0

    """
    a_lu, i, p = LU.ludcmp(a.copy())
    return LU.ludet(a_lu, p)


# ---------------------------------------------------------------------------
def identity(n):
    """Return an identity array with ``n`` dimensions

    .. versionadded:: 1.1

    **Example**::

        >>> la.identity(3)
        uarray([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

    """
    return uarray(np.identity(n, dtype=object))


def empty(shape):
    """Return an array of shape ``shape`` containing ``None`` elements

    .. versionadded:: 1.1

    **Example**::

        >>> la.empty( (2,3) )
        uarray([[None, None, None],
                [None, None, None]])

    """
    return uarray(np.empty(shape, dtype=object))


def zeros(shape):
    """Return an array of shape ``shape`` containing ``0`` elements

    .. versionadded:: 1.1

    **Example**::

        >>> la.zeros( (2,3) )
        uarray([[0, 0, 0],
                [0, 0, 0]])

    """
    return uarray(np.zeros(shape, dtype=object))


def ones(shape):
    """Return an array of shape ``shape`` containing ``1`` elements

    .. versionadded:: 1.1

    **Example**::

        >>> la.ones( (2,3) )
        uarray([[1, 1, 1], [1, 1, 1]])

    """
    return uarray(np.ones(shape, dtype=object))


def full(shape, fill_value):
    """Return an array of shape ``shape`` containing ``fill_value`` elements

    .. versionadded:: 1.1

    **Example**::

        >>> la.full( (1,3),ureal(2,1) )
        uarray([[ureal(2.0,1.0,inf), ureal(2.0,1.0,inf),
                 ureal(2.0,1.0,inf)]])

    """
    return uarray(np.full(shape, fill_value, dtype=object))


# import here to avoid circular imports when importing the matmul function in uncertain_array.py
from GTC.uncertain_array import UncertainArray

# ============================================================================
if __name__ == "__main__":
    import doctest
    from GTC import *

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
