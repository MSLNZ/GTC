"""
"""
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip
import math

__all__ = (
    'equivalent',
    'equivalent_complex',
    'equivalent_matt',
    'equivalent_matrix',
    'equivalent_sequence',
    'show_complex_difference',
    'TOL',
)

TOL = 1E-13
#-----------------------------------------------------
def equivalent_sequence(u,m,tol=TOL):
    """
    Compare each element of a pair of sequences
    for equivalence
    
    """
    assert len(u) == len(m), "different lengths: %s and %s" % (u,m)
    OK = True
    for u_i,m_i in izip(u,m):
        OK &=  abs(u_i - m_i) < tol       

    if OK:
        return True
    else:
        print("Tolerance = %G" % tol)
        print("Differences: ")
        for u_i,m_i in izip(u,m):
            print(abs(u_i - m_i))
            
        raise AssertionError("'%s' <> '%s' " % (u,m))

#-----------------------------------------------------
def equivalent_matt(m,n,tol=TOL):
    """
    `m` and `n` can be mapping objects or namedtuples
    
    """
    # convert namedtuples to mappings
    if hasattr(m,'_asdict'):
        m = m._hasdict()
    if hasattr(n,'_asdict'):
        n = n._hasdict()
        
    keys = m.keys()
    if (keys == n.keys()):
        for k in keys:
            if abs(m[k] - n[k]) >= tol:
                print("Values for '%s' differ: abs(%.15G - %.15G) = %.15G" % (
                        k,
                        m[k],
                        n[k],
                        abs(m[k] - n[k])
                    ))
                return False
        return True
    else:
        return False
    
#-----------------------------------------------------
def equivalent_matrix(u,m,tol=TOL):
    if(
        abs(u[0,0] - m[0,0]) < tol
    and abs(u[1,0] - m[1,0]) < tol
    and abs(u[0,1] - m[0,1]) < tol
    and abs(u[1,1] - m[1,1]) < tol
    ):
        return True
    else:
        print("Differences and tolerance: {} {} {} {} {}".format(
            abs(u[0,0] - m[0,0])
        ,   abs(u[1,0] - m[1,0])
        ,   abs(u[0,1] - m[0,1])
        ,   abs(u[1,1] - m[1,1])
        ,   tol))
        raise AssertionError("'%s' <> '%s' " % (u,m))
#-----------------------------------------------------
def show_complex_difference(x,y):
    """-> a string showing the difference in each component
    """
    z = complex(x)-complex(y)
    return "Differences: (%.15G,%.15G)" %(z.real, z.imag)
    
#-----------------------------------------------------
def equivalent_complex(x,y,tol=TOL):
    "Test the numerical equivalence of the complex arguments"
    xc = complex(x)
    yc = complex(y)
    if _equivalent(xc.real,yc.real,tol) and _equivalent(xc.imag,yc.imag,tol):
        return True
    else:
        print("Differences and tolerance: {} {} {}".format(abs(xc.real-yc.real), abs(xc.imag-yc.imag), tol))
        raise AssertionError("'(%.15G,%.15G)' <> '(%.15G,%.15G)' " % (x.real,x.imag,y.real,y.imag))

#-----------------------------------------------------
def _equivalent(x,y,tol):
    if( math.isinf(x) and math.isinf(y) and x == y ):
        return True
    elif( abs(x-y) < tol  ):
        return True
    else:
        return False
    
def equivalent(x,y,tol=TOL):
    """Test the numerical equivalence of the arguments
    """
    if _equivalent(x,y,tol):
        return True
    else:
        msg = "Values are not numerically equivalent: abs(%.16G-%.16G) = %.16G with tol= %.16G" % (
            x,y,abs(x-y),tol)
        raise AssertionError(msg)
