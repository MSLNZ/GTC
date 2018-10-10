from __future__ import division

import math
import cmath
import numbers
import collections
import itertools

from GTC.lib_real import *
from GTC.lib_complex import *
from GTC.context import *

from GTC import reporting
from GTC import type_b
from GTC import type_a

from GTC.named_tuples import (
    CorrelationMatrix, 
    CovarianceMatrix
)

def is_sequence(obj):
    if isinstance(obj, basestring):
        return False
    return isinstance(obj, collections.Sequence)
    
# aliases 
rp = reporting
tb = type_b
ta = type_a

__all__ = (
        'ureal'
    ,   'multiple_ureal'
    ,   'multiple_ucomplex'
    ,   'ucomplex'
    ,   'constant'
    ,   'value'
    ,   'uncertainty'
    ,   'variance'
    ,   'dof'
    ,   'label'
    ,   'component'
    ,   'inf'
    ,   'nan'
    ,   'get_covariance'
    ,   'get_correlation'
    ,   'set_correlation'
    ,   'result'
    ,   'cos'
    ,   'sin'
    ,   'tan'
    ,   'acos'
    ,   'asin'
    ,   'atan'
    ,   'atan2'
    ,   'exp'
    ,   'pow'
    ,   'log'
    ,   'log10'
    ,   'sqrt'
    ,   'sinh'
    ,   'cosh'
    ,   'tanh'
    ,   'acosh'
    ,   'asinh'
    ,   'atanh'
    ,   'mag_squared'
    ,   'magnitude'
    ,   'phase'
    ,   'reporting', 'rp'
    ,   'type_b', 'tb'
    ,   'type_a', 'ta'
    ,   'math'
    ,   'cmath'
    ,   'is_infinity'
    ,   'is_undefined'
)

#----------------------------------------------------------------------------
inf = float('inf')
nan = float('nan')

is_infinity = math.isinf
is_undefined = math.isnan

#----------------------------------------------------------------------------
def value(x):
    """Return the value 
    
    Returns a complex number if ``x`` is an uncertain complex number
    
    Returns a real number if ``x`` is an uncertain real number
    
    Returns ``x`` otherwise.

    **Example**::

        >>> un = ureal(3,1)
        >>> value(un)
        3.0
        >>> un.x
        3.0                  

    """
    try:
        return x.x
    except AttributeError:
        return x
    
#----------------------------------------------------------------------------
def uncertainty(x):
    """
    Return the standard uncertainty

    If ``x`` is an uncertain complex number,
    return a 2-element sequence containing the standard
    uncertainties of the real and imaginary components. 
    
    If ``x`` is an uncertain real number,
    return the standard uncertainty.
    
    Otherwise, return 0.    

    **Examples**::

        >>> ur = ureal(2.5,0.5,3,label='x')
        >>> uncertainty(ur)
        0.5
        >>> ur.u
        0.5
        
        >>> uc = ucomplex(1+2j,(.5,.5),3,label='x')
        >>> uncertainty(uc)
        standard_uncertainty(real=0.5, imag=0.5)
    
    """
    try:
        return x.u
    except AttributeError:
        return 0.0
    
#----------------------------------------------------------------------------
def variance(x):
    """
    Return the standard variance 

    If ``x`` is an uncertain real number, return the
    standard variance.
    
    If ``x`` is an uncertain complex number, return
    a 4-element sequence containing elements of the
    variance-covariance matrix.
    
    Otherwise, return 0.    
    
    **Examples**::

        >>> ur = ureal(2.5,0.5,3,label='x')
        >>> variance(ur)
        0.25
        >>> ur.v
        0.25
        
        >>> uc = ucomplex(1+2j,(.5,.5),3,label='x')
        >>> variance(uc)
        variance_covariance(rr=0.25, ri=0.0, ir=0.0, ii=0.25)
    
    """
    try:
        return x.v
    except AttributeError:
        return 0.0
    
#----------------------------------------------------------------------------
def dof(x):
    """
    Return the degrees-of-freedom 

    Returns ``inf`` when the degrees of freedom is greater than 1E6
    
    Returns ``nan`` when the calculation is invalid   
    
    **Examples**::

        >>> ur = ureal(2.5,0.5,3,label='x')
        >>> dof(ur)
        3.0
        >>> ur.df
        3.0
        
        >>> uc = ucomplex(1+2j,(.3,.2),3,label='x')
        >>> dof(uc)
        3.0
    
    """
    try:
        return x.df
    except AttributeError:
        return inf

#---------------------------------------------------------------------------
def component(y,x):
    """
    Return the magnitude of the component of uncertainty in ``y`` due to ``x``.

    :arg y: an uncertain number
    :type y: :class:`UncertainReal` or :class:`UncertainComplex`

    :arg x: an uncertain number
    :type x: :class:`UncertainReal` or :class:`UncertainComplex`

    :rtype: float
    
    If ``x`` and ``y`` are uncertain real, the magnitude of 
    :func:`reporting.u_component` is returned. 
    
    If either ``x`` or ``y`` is uncertain complex,
    the returned value represents the magnitude 
    of the component of uncertainty matrix (see also:  
    :func:`reporting.u_component` and :func:`reporting.u_bar`).

    If either ``x`` or ``y`` is a number, zero is returned.
    
    ``component`` can be used in conjunction with :func:`~core.result` 
    to evaluate a component of uncertainty with respect to an 
    intermediate uncertain number. 
    
    **Examples**::
    
        >>> x1 = ureal(2,1)
        >>> x2 = ureal(5,1)
        >>> y = x1/x2
        >>> reporting.u_component(y,x2)
        -0.08
        >>> component(y,x2)
        0.08    
        
        >>> z1 = ucomplex(1+2j,1)
        >>> z2 = ucomplex(3-2j,1)
        >>> y = z1 - z2
        >>> reporting.u_component(y,z2)
        u_components(rr=-1.0, ri=0.0, ir=0.0, ii=-1.0)        
        >>> component(y,z2)
        1.0

        >>> I = ureal(1E-3,1E-5)
        >>> R = ureal(1E3,1)
        >>> V = result( I*R )
        >>> P = V**2/R  
        >>> component(P,V)   
        2.0099751242241783E-05
        
    """
    return reporting.u_bar( reporting.u_component(y,x) )
            
#----------------------------------------------------------------------------
def label(x):
    """
    Return the label
    
    """
    return x.label 
    
#----------------------------------------------------------------------------
def ureal(x,u,df=inf,label=None,independent=True):
    """
    Create an elementary uncertain real number

    :arg x: the value (estimate)
    :type x: float
    
    :arg u: the standard uncertainty
    :type u: float
    
    :arg df: the degrees-of-freedom
    :type df: float
    
    :arg label: a string label 
    :type label: string 
    
    :arg independent: not correlated with other UNs
    :type independent: Boolean
    
    :rtype: :class:`UncertainReal`
    
    **Example**::
    
        >>> ur = ureal(2.5,0.5,3,label='x')
        >>> ur
        ureal(2.5, 0.5, 3.0, label='x')
    
    """    
    if not isinstance(x,numbers.Real):
        raise RuntimeError("invalid value: '{!r}'".format(x) )
    if not isinstance(u,numbers.Real) or u < 0:
        raise RuntimeError("invalid uncertainty: '{!r}'".format(u) )
    if not isinstance(df,numbers.Real) or df < 1:
        raise RuntimeError("invalid dof: '{!r}'".format(df) )
    
    if u == 0:
        return context.constant_real(float(x),label)
    else:
        return context.elementary_real(
            float(x),
            float(u),
            float(df),
            label,
            independent
        )
        
#---------------------------------------------------------------------------
def multiple_ureal(x_seq,u_seq,df,label_seq=None):
    """Return a sequence of related elementary uncertain real numbers

    :arg x_seq: a sequence of values (estimates)
    :arg u_seq: a sequence of standard uncertainties
    :arg df: the degrees-of-freedom 
    :arg label_seq: a sequence of labels
    
    :rtype: a sequence of :class:`UncertainReal`

    Defines an set of related uncertain real
    numbers with the same number of degrees-of-freedom.
    
    Correlation between any pairs of the uncertain  
    numbers defined does not invalidate degrees-of-freedom 
    calculations.
    (see: R Willink, *Metrologia* 44 (2007) 340-349, Sec. 4.1)
    
    **Example**::
    
        # Example from GUM-H2
        >>> x = [4.999,19.661E-3,1.04446]
        >>> u = [3.2E-3,9.5E-6,7.5E-4]
        >>> labels = ['V','I','phi']
        >>> v,i,phi = multiple_ureal(x,u,4,labels)
        
        >>> 
        correlation(-0.36,v,i)
        >>> set_correlation(0.86,v,phi)
        >>> set_correlation(-0.65,i,phi)

        >>> r = v/i*cos(phi)
        >>> summary(r)
        '127.732, u=0.070, df=4'
        >>> reporting.uncertainty_interval(r)
        expanded_uncertainty(lower=127.53788813535775, upper=127.92645172084642)

    """
    if len(x_seq) != len(u_seq):
        raise RuntimeError(
            "unequal length sequences: x={!r} u={!r}".format(x_seq,u_seq)
        )

    if label_seq is None:
        label_seq = [None]*len(x_seq)
    elif is_sequence(label_seq):
        if len(x_seq) != len(label_seq):
            raise RuntimeError(
                "unequal length sequences: x={!r} label_seq={!r}".format(x_seq,u_seq)
            )
    else:
        raise RuntimeError(
            "invalid `label_seq`: {!r}".format(label_seq)
        )
        
    rtn = [
        context.elementary_real(x_i,u_i,df,label=l_i,independent=False)
            for x_i,u_i,l_i in itertools.izip(
                x_seq,u_seq,label_seq
            )
    ]

    # Only non-constant UNs can be collected ina n ensemble
    context.real_ensemble( [un_i for un_i in rtn if un_i.u != 0], df )

    # All uncertain numbers are returned, including the constants
    return rtn

#----------------------------------------------------------------------------
def constant(x,label=None):
    """
    Create a constant uncertain number (with no uncertainty)

    :arg x: a number
    :type x: float or complex

    :rtype: :class:`UncertainReal` or :class:`UncertainComplex`

    If ``x`` is complex, return an uncertain complex number.
    
    If ``x`` is real return an uncertain real number.
        
    **Example**::
    
        >>> e = constant(math.e,label='Euler')
        >>> e
        ureal(2.718281828459045, 0, inf, label='Euler')
    
    """
    if isinstance(x,numbers.Real):
        return context.constant_real(
            x,label
        )
    elif isinstance(x,numbers.Complex):
        return context.constant_complex(
            x,label
        )
    else:
        raise RuntimeError(
            "Cannot make a constant: {!r}".format( x )
        )
  
#----------------------------------------------------------------------------
def result(un,label=None):
    """
    Declare ``un`` as an intermediate uncertain-number 'result'
    
    `un` - an uncertain number
    `label` - a label can be assigned
    
    The dependence of other uncertain numbers on a
    result can be stored in an archive.

    This function must be called before other
    uncertain numbers have been derived from the
    uncertain result.
    
    **Example**::

        # The dependence of `P` on `V`
        # can be archived 

        V = result( I*R )
        P = V**2/R  
        
    """
    un = +un 
    
    context = un._context
    if isinstance(un,UncertainReal):
        context.real_intermediate(un,label)
        
    elif isinstance(un,UncertainComplex):
        context.complex_intermediate(un,label)
        
    else:
        raise RuntimeError(
            "expected an uncertain number '{!r}'".format(un)
        )
          
    return un 

#----------------------------------------------------------------------------
def ucomplex(z,u,df=inf,label=None,independent=True):
    """
    Create an elementary uncertain complex number

    :arg z: the value (estimate)
    :type z: complex

    :arg u: the standard uncertainty or variance
    :type u: a float, 2-element or 4-element sequence

    :arg df: the degrees-of-freedom
    :type df: float

    :arg label: a string label 
    :type label: string 
    
    :rtype: :class:`UncertainComplex`

    ``u`` can be a float, a 2-element or 4-element sequence.

    If ``u`` is a float, the standard uncertainty in both
    the real and imaginary components is equal to ``u``.

    If ``u`` is a 2-element sequence, the first element is
    the standard uncertainty in the real component and the second
    element is the standard uncertainty in the imaginary.

    If ``u`` is a 4-element sequence, it is 
    associated with a variance-covariance matrix.

    A `RuntimeError` is raised if ``df`` or
    ``u`` have illegal values.

    **Examples**::

        >>> uc = ucomplex(1+2j,(.5,.5),3,label='x')
        >>> uc
        ucomplex( (1+2j), [0.25,0.0,0.0,0.25], 3.0, label=x )
   
    >>> cv = (1.2,0.7,0.7,2.2)
    >>> uc = ucomplex(0.2-.5j, cv)
    >>> variance(uc)
    variance_covariance(rr=1.1999999999999997, ri=0.7, ir=0.7, ii=2.2)   
    
    """
    if not isinstance(z,numbers.Complex):
        raise RuntimeError("invalid value: '{!r}'".format(z) )
        
    if not isinstance(df,numbers.Real) or df < 1:
        raise RuntimeError("invalid dof: '{!r}'".format(df) )
        
    if is_sequence(u):
    
        case = len(u)
        
        if case == 2:
            u_r = float(u[0])
            u_i = float(u[1])
            r = None
            
        elif case == 4:
            u_r,cv1,cv2,u_i = u
            if not isinstance(cv1,numbers.Real) or cv1 != cv2:
                raise RuntimeError(
                    "covariance elements not equal: {!r} and {!r}".format(cv1,cv2) 
                )
            
            u_r = math.sqrt(u_r)
            u_i = math.sqrt(u_i)
            r = cv1 / (u_r*u_i) if cv1 != 0 else None
            if r is not None:
                # This overrides an initial assignment
                independent = False
                
        else:
            raise RuntimeError(
                "invalid uncertainty sequence: '{!r}'".format(u)
            )
        
        if not isinstance(u_r,numbers.Real) or u_r < 0:
            raise RuntimeError("invalid real uncertainty: '{!r}'".format(u_r) )
            
        if not isinstance(u_i,numbers.Real) or u_i < 0:
            raise RuntimeError("invalid imag uncertainty: '{!r}'".format(u_i) )
        
        # Allow a little tolerance for numerical imprecision
        if r is not None and abs(r) > 1 + 1E-10:
            raise RuntimeError(
                "invalid correlation: {!r}, cv={}".format(r,u)
            )

    elif isinstance(u,numbers.Real):
        u_r = u_i = float(u)
        r = None
    else:
        raise RuntimeError("invalid uncertainty: '{!r}'".format(u) )
        

    if u_r == 0 and u_i == 0:
        return context.constant_complex(complex(z),label)
    else:
        return context.elementary_complex(
            complex(z),
            u_r,u_i,r,
            float(df),
            label,
            independent
        )
        
#---------------------------------------------------------------------------
def multiple_ucomplex(x_seq,u_seq,df,label_seq=None):
    """Return a sequence of uncertain complex numbers

    :arg x_seq: a sequence of complex values (estimates)
    :arg u_seq: a sequence of standard uncertainties or covariances
    :arg df: the degrees-of-freedom
    :arg label_seq: a sequence of labels for the uncertain numbers

    :rtype: a sequence of :class:`UncertainComplex`
    
    This function defines an set of related uncertain complex
    numbers with the same number of degrees-of-freedom.
    
    Correlation between pairs of these uncertain  
    numbers does not invalidate degrees-of-freedom calculations.
    (see: R Willink, *Metrologia* 44 (2007) 340-349, Sec. 4.1)
    
    **Example**::
    
        # GUM Appendix H2
        >>> values = [4.999+0j,0.019661+0j,1.04446j]
        >>> uncert = [(0.0032,0.0),(0.0000095,0.0),(0.0,0.00075)]
        >>> v,i,phi = multiple_ucomplex(values,uncert,5)
        
        >>> set_correlation(-0.36,v.real,i.real)
        >>> set_correlation(0.86,v.real,phi.imag)
        >>> set_correlation(-0.65,i.real,phi.imag)
        
        >>> z = v * exp(phi)/ i
        >>> z
        ucomplex(
            (127.73216992810208+219.84651191263839j), 
            u=[0.069978727988371722,0.29571682684612355], 
            r=-0.59099999999999997, 
            df=5
        )
        >>> reporting.uncertainty_interval(z.real)
        expanded_uncertainty(lower=127.55228662093492, upper=127.91205323526924)

    """
    if len(x_seq) != len(u_seq):
        raise RuntimeError(
            "unequal length sequences: x={!r} u={!r}".format(x_seq,u_seq)
        )
 
    if label_seq is None:
        label_seq = [None]*len(x_seq)
    elif is_sequence(label_seq):
        if len(x_seq) != len(label_seq):
            raise RuntimeError(
                "unequal length sequences: x={!r} label_seq={!r}".format(x_seq,u_seq)
            )
    else:
        raise RuntimeError(
            "invalid `label_seq`: {!r}".format(label_seq)
        )
 
    rtn = [
        ucomplex(x_i,u_i,df,label=l_i,independent=False)
            for x_i,u_i,l_i in itertools.izip(
                x_seq,u_seq,label_seq
            )
    ]

    # Only non-constant UNs can be collected in an ensemble
    context.complex_ensemble(
        [ 
            un_i for un_i in rtn
                if un_i.u[0] != 0.0 or un_i.u[1] != 0.0 
        ], 
        df
    )

    # All uncertain numbers are returned, including the constants
    return rtn

#----------------------------------------------------------------------------
def set_correlation(r,arg1,arg2=None):
    """Set the correlation between elementary uncertain numbers

    The input arguments may be a pair of uncertain numbers
    (the same type, real or complex), or a single
    uncertain complex number.
    
    Arguments must be elementary uncertain numbers.
    
    If the uncertain number arguments have finite degrees of 
    freedom they must have been declared using either 
    :func:`~core.multiple_ureal` or :func:`~multiple_ucomplex`.
    
    If the uncertain number arguments have infinite degrees of 
    freedom they may also be declared by setting `independent=False`
    when calling :func:`~ureal` or :func:`~ucomplex`.
        
    :class:`RuntimeError` is raised when illegal arguments are used

    When a pair of uncertain real numbers is used,
    ``r`` is the correlation between them. 
    
    When a pair of uncertain complex number arguments is used,
    ``r`` must be a 4-element sequence of correlation
    coefficients between the components of the complex quantities.
    
    **Examples**::

        >>> x1 = ureal(2,1,independent=False)
        >>> x2 = ureal(5,1,independent=False)
        >>> set_correlation(.3,x1,x2)
        >>> get_correlation(x1,x2)
        0.3

        >>> z = ucomplex(1+0j,(1,1),independent=False)
        >>> z
        ucomplex((1+0j), u=[1,1], r=0, df=inf)
        >>> set_correlation(0.5,z)
        >>> z
        ucomplex((1+0j), u=[1,1], r=0.5, df=inf)

        >>> x1 = ucomplex(1,(1,1),independent=False)
        >>> x2 = ucomplex(1,(1,1),independent=False)
        >>> correlation_mat = (0.25,0.5,0.75,0.5)
        >>> set_correlation(correlation_mat,x1,x2)
        >>> get_correlation(x1,x2)
        correlation_matrix(rr=0.25, ri=0.5, ir=0.75, ii=0.5)
    
    """
    # This requires no action, because no correlation 
    # is the same as zero.
    # NB, we don't check that a correlation coefficient 
    # is being re-defined, ever! If someone were doing that 
    # and tried to set to zero, they'd be disappointed :-)
    if r == 0.0: return 
    
    # This relies on checking done in the calling functions 
    # to ensure that arguments are elementary, declared dependent,
    # and that the correlation coefficient value is OK.
    if isinstance(arg1,UncertainReal):
        if isinstance(arg2,UncertainReal):
            if (
                is_infinity( arg1._node.df ) and
                is_infinity( arg2._node.df )
            ):
                set_correlation_real(arg1,arg2,r)
            else:
                if arg2._node in context._ensemble.get(arg1._node,[]):
                    set_correlation_real(arg1,arg2,r)
                else:
                    raise RuntimeError( 
                        "arguments must be in the same ensemble:" +\
                        "{!r}, {!r}".format(arg2._node,arg1._node)
                    )
        elif isinstance(arg2,UncertainComplex):
            raise RuntimeError(
                "`arg1` and `arg2` not the same type, "+\
                "got: {!r} and {!r}".format(arg1,arg2)
            )
            # r_rr = set_correlation_real(arg1,arg2.real,r[0])
            # r_ri = set_correlation_real(arg1,arg2.imag,r[1])
        else:
            raise RuntimeError(
                "second argument must be ureal, got: {!r}".format(arg2) 
            )
        
    elif isinstance(arg1,UncertainComplex):
        # A single complex number may have correlation set 
        # provided it was declared ``dependent``. The additional
        # requirement of infinite dof does not apply.
        if arg2 is None:
            set_correlation_real(arg1.real,arg1.imag,r)
            
        elif isinstance(arg2,UncertainReal):
            raise RuntimeError(
                "`arg1` and `arg2` not the same type, "+\
                "got: {!r} and {!r}".format(arg1,arg2)
            )
            # r_rr = set_correlation_real(arg1.real,arg2,r[0])
            # r_ir = set_correlation_real(arg1.imag,arg2,r[2])
        elif isinstance(arg2,UncertainComplex):
            if not( is_sequence(r) and len(r)==4 ):
                raise RuntimeError(
                    "needs a sequence of 4 correlation coefficients,"
                    + "got '{!r}'".format(r)
                )
            else:
                # Trivial case
                if all( r_i == 0.0 for r_i in r ): return 
                
                if (
                    is_infinity( arg1.real._node.df ) and
                    # ucomplex prevents these two cases
                    # is_infinity( arg2.real._node.df ) and
                    # is_infinity( arg1.imag._node.df ) and
                    is_infinity( arg2.imag._node.df )
                ):
                    set_correlation_real(arg1.real,arg2.real,r[0])
                    set_correlation_real(arg1.real,arg2.imag,r[1])
                    set_correlation_real(arg1.imag,arg2.real,r[2])
                    set_correlation_real(arg1.imag,arg2.imag,r[3])
                else:
                    # They have to be in the same ensemble. 
                    # Just need to cross-check one of the component 
                    # pairs to verify this
                    if arg2.real._node in context._ensemble.get(arg1.real._node,[]):
                        set_correlation_real(arg1.real,arg2.real,r[0])
                        set_correlation_real(arg1.real,arg2.imag,r[1])
                        set_correlation_real(arg1.imag,arg2.real,r[2])
                        set_correlation_real(arg1.imag,arg2.imag,r[3])
                    else:
                        raise RuntimeError( 
                            "arguments must be in the same ensemble"
                        )
        else:
            raise RuntimeError(
                "Illegal type for second argument: {!r}".format(arg2)
            )
    else:
        raise RuntimeError(
            "illegal arguments: {}, {}, {}".format( 
                repr(r), repr(arg1), repr(arg2) 
            )
        )

#---------------------------------------------------------------------------
def get_correlation(arg1,arg2=None):
    """Return the correlation 
    
    The input arguments may be a pair of uncertain numbers, 
    or a single uncertain complex number.
    
    When a pair of uncertain real numbers is used,
    the a real number is returned that is the correlation between 
    the two arguments. 
    
    One or both arguments are uncertain complex numbers,
    a ``CorrelationMatrix`` namedtuple is returned, representing
    a 2-by-2 matrix of correlation coefficients.
    
    """
    # If numerical arguments are given, then return zero. 
    
    # NB if the arg is any number type, it matches `numbers.Complex`
    if isinstance(arg1,numbers.Complex): arg1 = constant(arg1)
    
    if isinstance(arg1,UncertainReal):
        if isinstance(arg2,UncertainReal):
            return get_correlation_real(arg1,arg2)
        elif isinstance(arg2,UncertainComplex):
            r_rr = get_correlation_real(arg1,arg2.real)
            r_ri = get_correlation_real(arg1,arg2.imag)
            r_ir = 0.0
            r_ii = 0.0
            return CorrelationMatrix(r_rr,r_ri,r_ir,r_ii)
        elif isinstance(arg2,numbers.Real) or arg2 is None:
            # When second argument is a number, there is no correlation.
            # Arg2 is None when a real number is found, like 0,
            # and gets converted above to an UncertainReal constant,
            # when really it represented 0+0j. By implication
            # we return the zero correlation between real and 
            # imaginary components            
            return 0
        elif isinstance(arg2,numbers.Complex):
            # If second argument is a number, 
            # there is no correlation
            return CorrelationMatrix(0.0,0.0,0.0,0.0)
        else:
            raise RuntimeError(
                "illegal second argument '%r'" % arg2
            )  
            
    elif isinstance(arg1,UncertainComplex):
        if arg2 is None:
            return get_correlation_real(arg1.real,arg1.imag)
        elif isinstance(arg2,numbers.Complex): 
            # If second argument is a number, 
            # there is no correlation
            return CorrelationMatrix(0.0,0.0,0.0,0.0)
        elif isinstance(arg2,UncertainReal):
            r_rr = get_correlation_real(arg1.real,arg2)
            r_ri = 0.0
            r_ir = get_correlation_real(arg1.imag,arg2)
            r_ii = 0.0
            return CorrelationMatrix(r_rr,r_ri,r_ir,r_ii)
        elif isinstance(arg2,UncertainComplex):
            r_rr = get_correlation_real(arg1.real,arg2.real)
            r_ri = get_correlation_real(arg1.real,arg2.imag)
            r_ir = get_correlation_real(arg1.imag,arg2.real)
            r_ii = get_correlation_real(arg1.imag,arg2.imag)
            return CorrelationMatrix(r_rr,r_ri,r_ir,r_ii)
        else:
            raise RuntimeError(
                "illegal second argument '%r'" % arg2
            )
        
    else:
        raise RuntimeError(
            "illegal first argument '%r'" % arg1
        )
        
#---------------------------------------------------------------------------
def get_covariance(arg1,arg2=None):
    """Return the covariance 
    
    The input arguments may be a pair of uncertain numbers, 
    or a single uncertain complex number.
    
    When a pair of uncertain real numbers is used,
    the a real number is returned that is the correlation between 
    the two arguments. 
    
    One or both arguments are uncertain complex numbers,
    a ``CovarianceMatrix`` namedtuple is returned, representing
    a 2-by-2 variance-covariance matrix.
    
    """
    # If numerical arguments are given then return zero. 
    
    # NB if the arg is any number type, it matches `numbers.Complex`
    if isinstance(arg1,numbers.Complex): arg1 = constant(arg1)
    
    # Different possibilities for the second argument lead
    # to different types of return type:
    # 
    if isinstance(arg1,UncertainReal):
        if isinstance(arg2,UncertainReal):
            return get_covariance_real(arg1,arg2)
        elif isinstance(arg2,UncertainComplex):
            cv_rr = get_covariance_real(arg1,arg2.real)
            cv_ri = get_covariance_real(arg1,arg2.imag)
            cv_ir = 0.0
            cv_ii = 0.0
            return CovarianceMatrix(cv_rr,cv_ri,cv_ir,cv_ii)
        elif isinstance(arg2,numbers.Real) or arg2 is None:
            # Second argument can be a number, but
            # there is no correlation.
            # Arg2 is None when a real number is found, like 0,
            # and gets converted above to an UncertainReal constant,
            # when really it represented 0+0j. By implication
            # we return the zero correlation between real and imaginary
            # components            
            return 0.0
        elif isinstance(arg2,numbers.Complex):
            # Second argument can be a complex number, but
            # there is no correlation
            return CovarianceMatrix(0.0,0.0,0.0,0.0)
        else:
            raise RuntimeError(
                "illegal second argument '%r'" % arg2
            )  
            
    elif isinstance(arg1,UncertainComplex):
        if arg2 is None:
            return get_covariance_real(arg1.real,arg1.imag)
        elif isinstance(arg2,numbers.Complex): 
            # Second argument can be a number, but
            # there is no correlation
            return CovarianceMatrix(0.0,0.0,0.0,0.0)
        elif isinstance(arg2,UncertainReal):
            cv_rr = get_covariance_real(arg1.real,arg2)
            cv_ri = 0.0
            cv_ir = get_covariance_real(arg1.imag,arg2)
            cv_ii = 0.0
            return CovarianceMatrix(cv_rr,cv_ri,cv_ir,cv_ii)
        elif isinstance(arg2,UncertainComplex):
            cv_rr = get_covariance_real(arg1.real,arg2.real)
            cv_ri = get_covariance_real(arg1.real,arg2.imag)
            cv_ir = get_covariance_real(arg1.imag,arg2.real)
            cv_ii = get_covariance_real(arg1.imag,arg2.imag)
            return CovarianceMatrix(cv_rr,cv_ri,cv_ir,cv_ii)
        else:
            raise RuntimeError(
                "illegal second argument '%r'" % arg2
            )
        
    else:
        raise RuntimeError(
            "illegal first argument '%r'" % arg1
        )
        
#---------------------------------------------------------------------------
def log(x):
    """
    Uncertain number natural logarithm

    .. note::
        In the complex case there is one branch cut,
        from 0 along the negative real axis to :math:`-\infty`,
        continuous from above.
        
    """
    return x._log()
    
#---------------------------------------------------------------------------
def log10(x):
    """
    Uncertain number common logarithm (base-10)

    .. note::
        In the complex case there is one branch cut,
        from 0 along the negative real
        axis to :math:`-\infty`, continuous from above.
        
    """
    return x._log10()
    
#---------------------------------------------------------------------------
def exp(x):
    """
    Uncertain number exponential function

    """
    return x._exp()
    
#---------------------------------------------------------------------------
def pow(x,y):
    """
    Uncertain number power function
    
    Raises ``x`` to the power of ``y``
    
    """
    return x**y
        
#---------------------------------------------------------------------------
def sqrt(x):
    """
    Uncertain number square root function

    .. note::
        In the complex case there is one branch cut,
        from 0 along the negative real
        axis to :math:`-\infty`, continuous from above.
        
    """
    return x._sqrt()

#----------------------------------------------------------------------------
def sin(x):
    """
    Uncertain number sine function

    """
    return x._sin()

#----------------------------------------------------------------------------
def cos(x):
    """
    Uncertain number cosine function

    """
    return x._cos()
#----------------------------------------------------------------------------
def tan(x):
    """
    Uncertain number tangent function

    """
    return x._tan()

#---------------------------------------------------------------------------
def asin(x):
    """
    Uncertain number arcsine function

    .. note::
        In the complex case there are two branch cuts: one extends
        right, from 1 along the real axis to :math:`\infty`,
        continuous from below; the other extends left, from -1 along
        the real axis to :math:`-\infty`, continuous from above.
        
    """
    return x._asin()
#---------------------------------------------------------------------------
def acos(x):
    """
    Uncertain number arc-cosine function

    .. note::
        In the complex case there are two branch cuts: one extends
        right, from 1 along the real axis to :math:`\infty`, continuous
        from below; the other extends left, from -1 along the real axis
        to :math:`-\infty`, continuous from above.
        
    """
    return x._acos()
            
#---------------------------------------------------------------------------
def atan(x):
    """
    Uncertain number arctangent function

    .. note::
    
        In the complex case there are two branch cuts:
        One extends from :math:`\mathrm{j}` along the imaginary axis to
        :math:`\mathrm{j}\infty`, continuous from the right.
        The other extends from :math:`-\mathrm{j}` along the imaginary
        axis to :math:`-\mathrm{j}\infty`, continuous from the left.
        
    """
    return x._atan()
#----------------------------------------------------------------------------
def atan2(y,x):
    """
    Two-argument uncertain number arctangent function    

    :arg x: abscissa
    :type x: :class:`UncertainReal`
    :arg y: ordinate
    :type y: :class:`UncertainReal`
    
    .. note::   this function is not defined for uncertain complex numbers
                (use :func:`phase`)

    **Example**::

        >>> x = ureal(math.sqrt(3)/2,1)
        >>> y = ureal(0.5,1)
        >>> theta = atan2(y,x)
        >>> theta
        ureal(0.523598775598299,1,inf)
        >>> math.degrees( theta.x )
        30.000000000000004
    
    """
    try:
        return y._atan2(x)
    except AttributeError:
        return x._ratan2(y)

#---------------------------------------------------------------------------
def sinh(x):
    """
    Uncertain number hyperbolic sine function

    """
    return x._sinh()
#---------------------------------------------------------------------------
def cosh(x):
    """
    Uncertain number hyperbolic cosine function

    """
    return x._cosh()
    
#---------------------------------------------------------------------------
def tanh(x):
    """
    Uncertain number hyperbolic tangent function

    """
    return x._tanh()
            
#---------------------------------------------------------------------------
def asinh(x):
    """
    Uncertain number hyperbolic arcsine function

    .. note::
    
        In the complex case there are two branch cuts: one extends
        from :math:`\mathrm{j}` along the imaginary axis to
        :math:`\mathrm{j}\infty`, continuous from the right;
        the other extends from :math:`-\mathrm{j}` along the
        imaginary axis to :math:`-\mathrm{j}\infty`, continuous
        from the left.
        
    """
    return x._asinh()

#---------------------------------------------------------------------------
def acosh(x):
    """
    Uncertain number hyperbolic arc-cosine function

    .. note::
        In the complex case there is one branch cut,
        extending left from 1 along the
        real axis to :math:`-\infty`, continuous from above.
        
    """
    return x._acosh()

#---------------------------------------------------------------------------
def atanh(x):
    """
    Uncertain number hyperbolic arctangent function

    .. note::
        In the complex case there are two branch cuts:
        one extends from 1 along the real axis to :math:`\infty`,
        continuous from below; the other extends from -1
        along the real axis to :math:`-\infty`, continuous
        from above.
        
    """
    return x._atanh()
            
#----------------------------------------------------------------------------
def magnitude(x):
    """
    Return the magnitude of ``x``

    .. note::

        If ``x`` is not an uncertain number type,
        returns :func:`abs(x)`.    
    
    """
    try:
        return x._magnitude()
    except AttributeError:
        return abs(x)

#---------------------------------------------------------------------------
def phase(z):
    """
    :arg z: an uncertain complex number    
    :type z: :class:`UncertainComplex`

    :returns:   the phase in radians
    :rtype:     :class:`UncertainReal`
    
    """
    return z._phase()

#---------------------------------------------------------------------------
def mag_squared(x):
    """
    Return the squared magnitude of ``x``.

    .. note::
    
        If ``x`` is an uncertain number, the magnitude
        squared is returned as an uncertain real number, 
        otherwise :func:``abs(x)**2`` is returned.
    
    """
    try:
        return x._mag_squared()
    except AttributeError:
        return abs(x)**2
        
#============================================================================    
if __name__ == "__main__":
    import doctest
    from GTC import *
    
    doctest.testmod( optionflags=doctest.NORMALIZE_WHITESPACE )
