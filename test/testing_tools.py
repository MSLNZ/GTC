"""
"""
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
    assert len(u) == len(m), f"different lengths: {u} and {m}"
    OK = True
    for u_i,m_i in zip(u,m):
        OK &=  abs(u_i - m_i) < tol       

    if OK:
        return True
    else:
        print(f"Tolerance = {tol:G}")
        print("Differences: ")
        for u_i,m_i in zip(u,m):
            print(abs(u_i - m_i))
            
        raise AssertionError(f"'{u}' <> '{m}'")

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
                print(f"Values for '{k}' differ: abs({m[k]:.15G} - {n[k]:.15G}) = {abs(m[k] - n[k]):.15G}")
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
        print(f"Differences and tolerance: "
              f"{abs(u[0,0] - m[0,0])} "
              f"{abs(u[1,0] - m[1,0])} "
              f"{abs(u[0,1] - m[0,1])} "
              f"{abs(u[1,1] - m[1,1])} "
              f"{tol}")
        raise AssertionError(f"'{u}' <> '{m}'")
#-----------------------------------------------------
def show_complex_difference(x,y):
    """-> a string showing the difference in each component
    """
    z = complex(x)-complex(y)
    return f"Differences: ({z.real:.15G},{z.imag:.15G)}"
    
#-----------------------------------------------------
def equivalent_complex(x,y,tol=TOL):
    "Test the numerical equivalence of the complex arguments"
    xc = complex(x)
    yc = complex(y)
    if _equivalent(xc.real,yc.real,tol) and _equivalent(xc.imag,yc.imag,tol):
        return True
    else:
        print(f"Differences and tolerance: {abs(xc.real-yc.real)} {abs(xc.imag-yc.imag)} {tol}")
        raise AssertionError(f"'({x.real:.15G},{x.imag:.15G})' <> '({y.real:.15G},{y.imag:.15G})'")

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
        raise AssertionError(f"Values are not numerically equivalent: "
                             f"abs({x:.16G}-{y:.16G}) = {abs(x-y):.16G} with tol= {tol:.16G}")
