# This script is imported by test_uncertain_array.py
#
# From Python 3.5+ the @ symbol can be used for an ndarray with dtype=object
#
# If the test_uncertain_array.py module is executed with Python < 3.5 and it contains
# the @ symbol then a SyntaxError is raised. Therefore, this script is only imported
# and tested if Python >= 3.5
#
# This script is included in the --ignore option in setup.cfg [tool:pytest].
#
# Since the name of this file doesn't start with or end with "test"
# the `unittest discover` command will not attempt to import it.
import numpy as np

from GTC import ureal
from GTC.linear_algebra import uarray

from testing_tools import equivalent


def run():
    m = [[ureal(5, 1), ureal(-1, 0.3), ureal(3, 1.3)],
         [ureal(1, 0.1), ureal(2, 0.8), ureal(-3, 1)],
         [ureal(-1, 0.5), ureal(2, 1.1), ureal(4, 0.3)]]
    b = [ureal(1, 0.2), ureal(2, 1.1), ureal(3, 0.4)]

    ma = uarray(m)
    ba = uarray(b)

    # vector * vector

    z = b[0] * 1 + b[1] * 2 + b[2] * 3
    za = ba @ [1, 2, 3]
    assert equivalent(z.x, za.value())
    assert equivalent(z.u, za.uncertainty())

    try:
        ba @ [1, 2]
    except ValueError:  # Expect this error -> shapes (3,) and (2,) not aligned: 3 (dim 0) != 2 (dim 0)
        pass
    else:
        raise ValueError('this should not work -> ba @ [1, 2]')

    # vector * matrix

    z = [1 * m[0][0] + 2 * m[1][0] + 3 * m[2][0],
         1 * m[0][1] + 2 * m[1][1] + 3 * m[2][1],
         1 * m[0][2] + 2 * m[1][2] + 3 * m[2][2]]
    za = [1, 2, 3] @ ma
    for i in range(3):
        assert equivalent(z[i].x, za[i].x)
        assert equivalent(z[i].u, za[i].u)

    try:
        [1, 2] @ ma
    except ValueError:  # Expect this error -> shapes (2,) and (3,3) not aligned: 2 (dim 0) != 3 (dim 0)
        pass
    else:
        raise ValueError('this should not work -> [1, 2] @ ma')

    # matrix * vector

    z = [m[0][0] * b[0] + m[0][1] * b[1] + m[0][2] * b[2],
         m[1][0] * b[0] + m[1][1] * b[1] + m[1][2] * b[2],
         m[2][0] * b[0] + m[2][1] * b[1] + m[2][2] * b[2]]

    za = ma @ ba
    for i in range(3):
        assert equivalent(z[i].x, za[i].x)
        assert equivalent(z[i].u, za[i].u)

    try:
        ma @ np.arange(4)
    except ValueError:  # Expect this error -> shapes (3,3) and (4,) not aligned: 3 (dim 1) != 4 (dim 0)
        pass
    else:
        raise ValueError('this should not work -> ma @ np.arange(4)')

    # matrix * matrix

    na = np.arange(10*10).reshape(10, 10) * -3.1
    nb = np.arange(10*10).reshape(10, 10) * 2.3
    nc = na @ nb

    ua = uarray(na.copy() * ureal(1, 0))
    ub = uarray(nb.copy() * ureal(1, 0))
    uc = ua @ ub
    assert nc.shape == uc.shape

    i, j = nc.shape
    for ii in range(i):
        for jj in range(j):
            assert equivalent(na[ii, jj], ua[ii, jj].x)
            assert equivalent(nb[ii, jj], ub[ii, jj].x)
            assert equivalent(nc[ii, jj], uc[ii, jj].x, tol=1e-10)

    try:
        ma @ np.arange(4*4).reshape(4, 4)
    except ValueError:  # Expect this error -> shapes (3,3) and (4,4) not aligned: 3 (dim 1) != 4 (dim 0)
        pass
    else:
        raise ValueError('this should not work -> ma @ np.arange(4*4).reshape(4,4)')

    # test a bunch of different dimensions
    test_dims = [
        [(), ()],
        [(0,), (1, 3)],
        [(1,), (1, 3)],
        [(4,), (4, 3)],
        [(2, 4), (4,)],
        [(2, 4), (3,)],
        [(2, 4), (3, 2)],
        [(2, 4), (4, 2)],
        [(1, 2, 4), (1, 4, 2)],
        [(2, 2, 4), (1, 4, 2)],
        [(1, 2, 4), (2, 4, 2)],
        [(2, 2, 4), (2, 4, 2)],
        [(3, 2, 4), (3, 4, 2)],
        [(6, 2, 4), (3, 2, 2)],
        [(6, 2, 4), (3, 4, 8)],
        [(6, 2, 4), (6, 4, 8)],
        [(5, 3, 2, 4), (5, 3, 4, 2)],
        [(3, 2, 2, 4), (3, 9, 4, 2)],
        [(8, 3, 1, 2, 4), (8, 3, 9, 4, 2)],
    ]

    for s1, s2 in test_dims:
        na = np.arange(int(np.prod(np.array(s1)))).reshape(s1)
        nb = np.arange(int(np.prod(np.array(s2)))).reshape(s2)
        try:
            nc = na @ nb
        except:
            nc = None

        ua = uarray(na.copy() * ureal(1, 0))
        ub = uarray(nb.copy() * ureal(1, 0))
        try:
            uc = ua @ ub
        except:
            if nc is not None:
                raise AssertionError('The regular @ PASSED, the custom-written @ FAILED')
        else:
            if nc is None:
                raise AssertionError('The regular @ FAILED, the custom-written @ PASSED')
            assert np.array_equal(nc, uc), 'The arrays are not equal\n{}\n{}'.format(nc, uc)
