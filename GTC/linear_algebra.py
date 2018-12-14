"""

"""
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip

import numpy as np

from GTC.uncertain_array import UncertainArray
from GTC import LU

__all__ = (
    'uarray',
    'dot',
    'matmul',
    'solve',
    'inv',
    'det',
    'identity','zeros','ones','empty','full'
)

def uarray(array, label=None, dtype=None, names=None):
    """Create an array of uncertain numbers.

    For an overview on how to use an :class:`.UncertainArray` see :ref:`numpy-uarray`.

    .. attention::

       This function requires that numpy :math:`\geq` v1.13.0 is installed.

    :param array: An array-like object containing :class:`int`, :class:`float`, :class:`complex`
                  :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex` elements.
    :param label: A label to assign to the `array`. This `label` does not
                  change the labels that were previously assigned to each array
                  element when they were created using :func:`~core.ureal` or
                  :func:`~core.ucomplex`.
    :type label: str
    :param dtype: The data type to use to create the array.
    :type dtype: :class:`numpy.dtype`
    :param names: The field `names` to use to create a
                  :ref:`Structured array <structured_arrays>`. If `dtype` is
                  specified then it gets precedence.
    :type names: list[str]

    :return: An :class:`.UncertainArray`.

    **Examples**:

        Create an `amps` and a `volts` array and then calculate the `resistances`

        >>> amps = uarray([ureal(0.57, 0.18), ureal(0.45, 0.12), ureal(0.68, 0.19)])
        >>> volts = uarray([ureal(10.3, 1.3), ureal(9.5, 0.8), ureal(12.6, 1.9)])
        >>> resistances = volts / amps
        >>> resistances
        UncertainArray([ureal(18.070175438596493,6.145264246839438,inf),
                        ureal(21.11111111111111,5.903661880050747,inf),
                        ureal(18.52941176470588,5.883187720636909,inf)], dtype=object)

        Create a :ref:`Structured array <structured_arrays>`, with the names ``'amps'`` and ``'volts'``,
        and then calculate the `resistances`

        >>> data = uarray([(ureal(0.57, 0.18), ureal(10.3, 1.3)),
        ...                (ureal(0.45, 0.12), ureal(9.5, 0.8)),
        ...                (ureal(0.68, 0.19), ureal(12.6, 1.9))], names=['amps', 'volts'])
        >>> resistances = data['volts'] / data['amps']
        >>> resistances
        UncertainArray([ureal(18.070175438596493,6.145264246839438,inf),
                        ureal(21.11111111111111,5.903661880050747,inf),
                        ureal(18.52941176470588,5.883187720636909,inf)], dtype=object)

    """
    if UncertainArray is None:
        raise ImportError('Requires numpy >= v1.13.0 to be installed')

    if (dtype is None) and (names is not None):
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
    """Matrix product of two arrays.

    For more details see :func:`numpy.matmul`.

    :param lhs: The array-like object on the left-hand side.
    :param rhs: The array-like object on the right-hand side.
    :return: The matrix product.
    :rtype: :class:`.UncertainArray`
    """
    try:
        # first, see if support for dtype=object was added
        return UncertainArray(np.matmul(lhs, rhs))
    except TypeError:
        # Must re-implement matrix multiplication because np.matmul
        # does not currently (as of v1.15.3) support dtype=object arrays.
        # A fix is planned for v1.16.0

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

    :param lhs: The array-like object on the left-hand side.
    :param rhs: The array-like object on the right-hand side.
    :return: The dot product.
    :rtype: :class:`.UncertainArray`
    """
    return UncertainArray(np.dot(lhs, rhs))

#---------------------------------------------------------------------------
def solve(a,b):
    """Return :math:`x`, the solution of :math:`a \cdot x = b`

    :arg a: 2D :class:`~uncertain_array.UncertainArray`
    :arg b: :class:`~uncertain_array.UncertainArray`
    
    :rtype: :class:`~uncertain_array.UncertainArray`

    
    **Example**::
    
        >>> a = la.uarray([[-2,3],[-4,1]])
        >>> b = la.uarray([4,-2])
        >>> la.solve(a,b)
        UncertainArray([1.0, 2.0], dtype=object)
    
    """
    return LU.solve(a,b)
    

#---------------------------------------------------------------------------
def inv(a):
    """Return the (multiplicative) matrix inverse 

    **Example**::
    
        >>> x = la.uarray( [[2,1],[3,4]])
        >>> x_inv =la.inv(x)
        >>> la.matmul(x,x_inv)
        UncertainArray([[1.0, 0.0],
                [4.440892098500626e-16, 1.0]], dtype=object)
    
    """
    return LU.invab(a,np.identity(a.shape[0]))

#---------------------------------------------------------------------------
def det(a):
    """Return the matrix determinant

    **Example**::
    
        >>> x = la.uarray( range(4) )
        >>> x.shape = 2,2
        >>> print x
        [[0 1]
        [2 3]]
        >>> la.det(x)
        -2.0

    """    
    a_lu, i, p = LU.ludcmp(a.copy())
    return LU.ludet(a_lu,p)

#---------------------------------------------------------------------------
def identity(n,dtype=None):
    """Return an identity array with ``n`` dimensions
    
    **Example**::
    
        >>> la.identity(3)
        UncertainArray([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]], dtype=object)
        
    """    
    if dtype is None: dtype=object
    return uarray( np.identity(n,dtype=dtype) )

def empty(shape,dtype=None):
    """Return an array of shape ``shape`` containing ``None`` elements
    
    **Example**::
    
        >>> la.empty( (2,3) )
        UncertainArray([[None, None, None],
                    [None, None, None]], dtype=object)
        
    """
    if dtype is None: dtype=object
    return uarray( np.empty(shape,dtype=dtype) )


def zeros(shape,dtype=None):
    """Return an array of shape ``shape`` containing ``0``'s
    
    **Example**::
    
        >>> la.zeros( (2,3) )
        UncertainArray([[0, 0, 0],
                [0, 0, 0]], dtype=object)
        
    """
    if dtype is None: dtype=object
    return uarray( np.zeros(shape,dtype=dtype) )

def ones(shape,dtype=None):
    """Return an array of shape ``shape`` containing ``1``'s

    **Example**::
    
        >>> la.ones( (2,3) )
        UncertainArray([[1, 1, 1],
                    [1, 1, 1]], dtype=object)
    
    """
    if dtype is None: dtype=object
    return uarray( np.ones(shape,dtype=dtype) )
    
def full(shape,fill_value,dtype=None):
    """Return an array of shape ``shape`` containing ``fill_value``'s

    **Example**::
    
        >>> la.full( (1,3),ureal(2,1) )
        UncertainArray([[ureal(2.0,1.0,inf), ureal(2.0,1.0,inf),
                 ureal(2.0,1.0,inf)]], dtype=object)
    
    """
    if dtype is None: dtype=object
    return uarray( np.full(shape,fill_value,dtype=dtype) )

#============================================================================    
if __name__ == "__main__":
    import doctest       
    from GTC import *
    doctest.testmod(  optionflags=doctest.NORMALIZE_WHITESPACE )