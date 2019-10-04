from __future__ import division

import math
import cmath
import numbers
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip

from GTC import lib
from GTC import reporting
from GTC import type_b
from GTC import type_a
from GTC import persistence
from GTC import function

from GTC import (   
    inf,
    nan,
    is_sequence,
    copyright,
    version
)

# aliases 
rp = reporting
tb = type_b
ta = type_a
fn = function
pr = persistence

value = type_b.value

UncertainReal = lib.UncertainReal
UncertainComplex = lib.UncertainComplex

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
    ,   'copyright'
    ,   'version'
    ,   'reporting',    'rp'
    ,   'function',     'fn'
    ,   'type_b',       'tb'
    ,   'type_a',       'ta'
    ,   'persistence',  'pr'
    ,   'linear_algebra', 'la'
    ,   'math'
    ,   'cmath'
)


# #----------------------------------------------------------------------------
# def value(x):
    # """Return the value 
    
    # Returns a complex number if ``x`` is an uncertain complex number
    
    # Returns a real number if ``x`` is an uncertain real number
    
    # Returns ``x`` otherwise.

    # **Example**::

        # >>> un = ureal(3,1)
        # >>> value(un)
        # 3.0
        # >>> un.x
        # 3.0

    # """
    # try:
        # return x.x
    # except AttributeError:
        # return x
    
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
        StandardUncertainty(real=0.5, imag=0.5)
    
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
        VarianceCovariance(rr=0.25, ri=0.0, ir=0.0, ii=0.25)
    
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
    Return the magnitude of the component of uncertainty 
    in ``y`` due to ``x``.

    :arg y: an uncertain number
    :type y: :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex`

    :arg x: an uncertain number
    :type x: :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex`

    :rtype: float
    
    If ``x`` and ``y`` are uncertain real, the function calls 
    :func:`reporting.u_component` and returns the magnitude 
    of the result. 
    
    If either ``x`` or ``y`` is uncertain complex,
    the returned value represents the magnitude 
    of the component of uncertainty matrix (this is 
    obtained by applying :func:`reporting.u_bar`    
    to the result obtained from :func:`reporting.u_component`).

    If either ``x`` or ``y`` is a number, zero is returned.
    
    ``component`` can also e used in conjunction with :func:`~core.result` 
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
        ComponentOfUncertainty(rr=-1.0, ri=0.0, ir=0.0, ii=-1.0)
        >>> component(y,z2)
        1.0

        >>> I = ureal(1E-3,1E-5)
        >>> R = ureal(1E3,1)
        >>> V = result( I*R )
        >>> P = V**2/R  
        >>> component(P,V)   
        2.0099751242241783e-05
        
    """
    return reporting.u_bar( y.u_component(x) )
            
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
    :type label: str 
    
    :arg independent: not correlated with other UNs
    :type independent: bool
    
    :rtype: :class:`~lib.UncertainReal`
    
    **Example**::
    
        >>> ur = ureal(2.5,0.5,3,label='x')
        >>> ur
        ureal(2.5,0.5,3.0, label='x')
    
    """    
    # Arguments to these math functions must be compatible with float
    if math.isnan(x) or math.isinf(x):
        raise ValueError("invalid: '{!r}'".format(x) )

    if u < 0 or math.isinf(u) or math.isnan(u):
        raise ValueError("invalid uncertainty: '{!r}'".format(u) )

    # inf is allowed, but not nan
    if df < 1 or math.isnan(df):
        raise ValueError("invalid dof: '{!r}'".format(df) )
    
    if u == 0:
        # Is this what we want? Perhaps not.
        return UncertainReal._constant(float(x),label)

    else:
        return UncertainReal._elementary(
            float(x),
            float(u),
            float(df),
            label,
            independent
        )
        
#---------------------------------------------------------------------------
# TODO: think of a better name! Perhaps `ureal_ensemble`
def multiple_ureal(x_seq,u_seq,df,label_seq=None):
    """Return a sequence of related elementary uncertain real numbers

    :arg x_seq: a sequence of values (estimates)
    :arg u_seq: a sequence of standard uncertainties
    :arg df: the degrees-of-freedom 
    :arg label_seq: a sequence of labels
    
    :rtype: a sequence of :class:`~lib.UncertainReal`

    Defines an set of uncertain real numbers with 
    the same number of degrees-of-freedom.
    
    Correlation between any pairs of this set of uncertain  
    numbers defined will not invalidate degrees-of-freedom 
    calculations.
    (see: R Willink, *Metrologia* 44 (2007) 340-349, Sec. 4.1)
    
    **Example**::
    
        # Example from GUM-H2
        >>> x = [4.999,19.661E-3,1.04446]
        >>> u = [3.2E-3,9.5E-6,7.5E-4]
        >>> labels = ['V','I','phi']
        >>> v,i,phi = multiple_ureal(x,u,4,labels)
   
        >>> set_correlation(-0.36,v,i)
        >>> set_correlation(0.86,v,phi)
        >>> set_correlation(-0.65,i,phi)

        >>> r = v/i*cos(phi)
        >>> r
        ureal(127.732169928102...,0.0699787279883717...,4.0)

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
        # NB `ureal` creates constant objects when u == 0
        ureal(x_i,u_i,df,label=l_i,independent=False)
            for x_i,u_i,l_i in izip(
                x_seq,u_seq,label_seq
            )
    ]

    # Only non-constant UNs can be collected in an ensemble
    lib.real_ensemble( 
        [ un_i 
            for un_i in rtn 
                if not lib._is_uncertain_real_constant(un_i) 
        ], df 
    )

    # All uncertain numbers are returned, including the constants
    return rtn

#----------------------------------------------------------------------------
def constant(x,label=None):
    """
    Create a constant uncertain number (with no uncertainty)

    :arg x: a number
    :type x: float or complex

    :rtype: :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex`

    If ``x`` is complex, return an uncertain complex number.
    
    If ``x`` is real return an uncertain real number.
        
    **Example**::
    
        >>> e = constant(math.e,label='Euler')
        >>> e
        ureal(2.718281828459045,0.0,inf, label='Euler')
    
    """
    if isinstance(x,numbers.Real):
        return UncertainReal._constant(x,label)
    elif isinstance(x,numbers.Complex):
        return UncertainComplex._constant(x,label)
    else:
        raise TypeError(
            "Cannot make a constant: {!r}".format( x )
        )
  
#----------------------------------------------------------------------------
def result(un,label=None):
    """
    Declare an uncertain number to be an intermediate result

    :arg un: an uncertain number or :class:`.UncertainArray`
    :arg label: a string or sequence of strings
    
    When ``un`` is an array, an :class:`.UncertainArray` is returned  
    containing the intermediate uncertain number objects.
    
    .. note::
    
        This function does not affect the argument ``un``.
        Rather, a new intermediate result object is created.
        So, this function will usually be applied to a temporary object.
    
    The component of uncertainty, or the sensitivity, of an uncertain number 
    with respect to an intermediate result can be evaluated. 
    
    Declaring intermediate results also enables the dependencies of uncertain 
    numbers to be stored in an archive.

    :arg un: :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex` or :class:`.UncertainArray`
    :arg label: str or a sequence of str
    :rtype: :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex` or :class:`.UncertainArray`
    
    **Example**::

        >>> I = ureal(1.3E-3,0.01E-3)
        >>> R = ureal(995,7)
        >>> V = result( I*R )
        >>> P = V**2/R
        >>> component(P,V)
        3.505784505642068e-05  
    
    """
    if hasattr(un,'_intermediate'):
        return un._intermediate(label)
    elif isinstance(un,numbers.Complex):
        return un
    else:
        raise TypeError(
            "undefined for {!r}'".format(un)
        )
          

#----------------------------------------------------------------------------
def ucomplex(z,u,df=inf,label=None,independent=True):
    """
    Create an elementary uncertain complex number

    :arg z: the value (estimate)
    :type z: complex

    :arg u: the standard uncertainty or variance
    :type u: float, 2-element or 4-element sequence

    :arg df: the degrees-of-freedom
    :type df: float

    :type label: str 
    
    :rtype: :class:`~lib.UncertainComplex`
    :raises: :exc:`ValueError` if ``df`` or ``u`` have illegal values.

    ``u`` can be a float, a 2-element or 4-element sequence.

    If ``u`` is a float, the standard uncertainty in both
    the real and imaginary components is taken to be ``u``.

    If ``u`` is a 2-element sequence, the first element is
    taken to be the standard uncertainty in the real component 
    and the second element is taken to be the standard 
    uncertainty in the imaginary component.

    If ``u`` is a 4-element sequence, the sequence is 
    interpreted as a variance-covariance matrix.

    **Examples**::

        >>> uc = ucomplex(1+2j,(.5,.5),3,label='x')
        >>> uc
        ucomplex((1+2j), u=[0.5,0.5], r=0.0, df=3.0, label=x)
   
    >>> cv = (1.2,0.7,0.7,2.2)
    >>> uc = ucomplex(0.2-.5j, cv)
    >>> variance(uc)
    VarianceCovariance(rr=1.1999999999999997, ri=0.7, ir=0.7, ii=2.2)
    
    """
    # Arguments to these math functions must be compatible with float
    # otherwise a TypeError is raised by Python
    if cmath.isnan(z) or cmath.isinf(z):
        raise ValueError("invalid: '{!r}'".format(z) )
        
    if df < 1 or math.isnan(df):
        raise ValueError("invalid dof: '{!r}'".format(df) )
        
    if is_sequence(u):
    
        case = len(u)
        
        if case == 2:
            u_r = float(u[0])
            u_i = float(u[1])
            r = None

        elif case == 4:
            u_r,cv1,cv2,u_i = u

            # nan != nan is True
            if math.isinf(cv1) or cv1 != cv2:
                raise ValueError(
                    "covariance elements not equal: {!r} and {!r}".format(cv1,cv2) 
                )
            u_r = math.sqrt(u_r)
            u_i = math.sqrt(u_i)
            r = cv1 / (u_r*u_i) if cv1 != 0 else None

            # Allow a little tolerance for numerical imprecision
            if r is not None and abs(r) > 1 + 1E-10:
                raise ValueError(
                    "invalid correlation: {!r}, cv={}".format(r,u)
                )
            if r is not None:
                # This overrides an initial assignment
                independent = False
                
        else:
            raise ValueError(
                "invalid uncertainty sequence: '{!r}'".format(u)
            )
        
    elif not math.isinf(u) and not math.isnan(u):
        u_r = u_i = float(u)
        r = None
    else:
        raise TypeError("invalid uncertainty: '{!r}'".format(u) )

    # Checking of valid uncertainty values
    # Note, comparisons with nan are always false
    if not( 0 <= u_r and u_r < inf ):
        raise ValueError("invalid real uncertainty: '{!r}'".format(u_r) )

    if not ( 0 <= u_i and u_i < inf ):
        raise ValueError("invalid imag uncertainty: '{!r}'".format(u_i) )
        
    # TODO: is this what we want? Perhaps not!
    if u_r == 0 and u_i == 0:
        return UncertainComplex._constant(complex(z),label)
    else:
        return UncertainComplex._elementary(
            complex(z),
            u_r,u_i,r,
            float(df),
            label,
            independent
        )
        
#---------------------------------------------------------------------------
# TODO: think of a better name! Perhaps `ucomplex_ensemble`
def multiple_ucomplex(x_seq,u_seq,df,label_seq=None):
    """Return a sequence of uncertain complex numbers

    :arg x_seq: a sequence of complex values
    :arg u_seq: a sequence of standard uncertainties or covariances
    :arg df: the degrees-of-freedom
    :arg label_seq: a sequence of labels for the uncertain numbers

    :rtype: a sequence of :class:`~lib.UncertainComplex`
    
    This function defines an set of uncertain complex
    numbers with the same number of degrees-of-freedom.
    
    Correlation between any pairs of these uncertain  
    numbers will not invalidate degrees-of-freedom calculations.
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
        >>> print(z)
        (127.732(70)+219.847(296)j)
        >>> z.r
        -28.5825760885182...

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
        # When u_i == 0 constant objects are created
        ucomplex(x_i,u_i,df,label=l_i,independent=False)
            for x_i,u_i,l_i in izip(
                x_seq,u_seq,label_seq
            )
    ]

    # Only non-constant UNs can be collected in an ensemble
    lib.complex_ensemble(
        [ 
            un_i for un_i in rtn
                if not lib._is_uncertain_complex_constant(un_i) 
        ], 
        df
    )

    # All uncertain numbers are returned, including the constants
    return rtn

#----------------------------------------------------------------------------
def set_correlation(r,arg1,arg2=None):
    """Set correlation between elementary uncertain numbers

    The input arguments can be a pair of uncertain numbers
    (the same type, real or complex), or a single
    uncertain complex number.
    
    The uncertain number arguments must be elementary 
    uncertain numbers.
    
    If the arguments have finite degrees of 
    freedom, they must be declared together using either 
    :func:`~core.multiple_ureal` or :func:`~multiple_ucomplex`.
    
    If the uncertain number arguments have infinite degrees of 
    freedom they can, alternatively, be declared by setting the 
    argument `independent=False` when calling 
    :func:`~ureal` or :func:`~ucomplex`.
        
    A :exc:`ValueError` is raised when illegal arguments are used

    When a pair of uncertain real numbers is provided,
    ``r`` is the correlation coefficient between them. 
    
    When a pair of uncertain complex number arguments is provided,
    ``r`` must be a 4-element sequence containing correlation
    coefficients between the components of the complex quantities.
    
    **Examples**::

        >>> x1 = ureal(2,1,independent=False)
        >>> x2 = ureal(5,1,independent=False)
        >>> set_correlation(.3,x1,x2)
        >>> get_correlation(x1,x2)
        0.3

        >>> z = ucomplex(1+0j,(1,1),independent=False)
        >>> z
        ucomplex((1+0j), u=[1.0,1.0], r=0.0, df=inf)
        >>> set_correlation(0.5,z)
        >>> z
        ucomplex((1+0j), u=[1.0,1.0], r=0.0, df=inf)

        >>> x1 = ucomplex(1,(1,1),independent=False)
        >>> x2 = ucomplex(1,(1,1),independent=False)
        >>> correlation_mat = (0.25,0.5,0.75,0.5)
        >>> set_correlation(correlation_mat,x1,x2)
        >>> get_correlation(x1,x2)
        CorrelationMatrix(rr=0.25, ri=0.5, ir=0.75, ii=0.5)
    
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
    if isinstance( arg1,(UncertainReal,UncertainComplex) ):
        arg1.set_correlation(r,arg2)
    else:
        raise TypeError(
            "illegal arguments: {}, {}, {}".format( 
                repr(r), repr(arg1), repr(arg2) 
            )
        )

#---------------------------------------------------------------------------
def get_correlation(arg1,arg2=None):
    """Return correlation 
    
    The input arguments may be a pair of uncertain numbers, 
    or a single uncertain complex number.
    
    When a pair of uncertain real numbers is provided,
    the correlation between the arguments is returned as 
    a real number. 
    
    When one, or both, arguments are uncertain complex numbers,
    a :obj:`~named_tuples.CorrelationMatrix` is returned, 
    representing a 2-by-2 matrix of correlation coefficients.
    
    """
    # Return zero if numerical arguments are given
    
    # If the arg is any number type, it matches `numbers.Complex`
    if isinstance(arg1,numbers.Complex): 
        arg1 = constant(arg1)
    
    if isinstance(arg1,(UncertainReal,UncertainComplex)):
        return arg1.get_correlation(arg2)
        
    else:
        raise TypeError(
            "illegal first argument {!r}".format(arg1)
        )
        
#---------------------------------------------------------------------------
def get_covariance(arg1,arg2=None):
    """Evaluate covariance.
    
    The input arguments can be a pair of uncertain numbers, 
    or a single uncertain complex number.
    
    When a pair of uncertain real numbers is supplied,
    the correlation between the two arguments is returned 
    as a real number. 
    
    When one, or both, arguments are uncertain complex numbers,
    a :class:`~named_tuples.CovarianceMatrix` is returned, 
    representing a 2-by-2 variance-covariance matrix.
    
    """
    # If numerical arguments are given then return zero. 
    # NB if the arg is any number type, it matches `numbers.Complex`
    if isinstance(arg1,numbers.Complex): 
        arg1 = constant(arg1)
    
    if isinstance( arg1,(UncertainReal,UncertainComplex) ):
        return arg1.get_covariance(arg2)
        
    else:
        raise TypeError(
            "illegal first argument {!r}".format(arg1)
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
    try:
        return x._log()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.log(x)
        elif isinstance(x,numbers.Complex):
            return cmath.log(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )
    
#---------------------------------------------------------------------------
def log10(x):
    """
    Uncertain number common logarithm (base-10)

    .. note::
        In the complex case there is one branch cut,
        from 0 along the negative real
        axis to :math:`-\infty`, continuous from above.
        
    """
    try:
        return x._log10()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.log10(x)
        elif isinstance(x,numbers.Complex):
            return cmath.log10(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )
    
#---------------------------------------------------------------------------
def exp(x):
    """
    Uncertain number exponential function

    """
    try:
        return x._exp()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.exp(x)
        elif isinstance(x,numbers.Complex):
            return cmath.exp(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )
    
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
    try:
        return x._sqrt()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.sqrt(x)
        elif isinstance(x,numbers.Complex):
            return cmath.sqrt(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )

#----------------------------------------------------------------------------
def sin(x):
    """
    Uncertain number sine function

    """
    try:
        return x._sin()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.sin(x)
        elif isinstance(x,numbers.Complex):
            return cmath.sin(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )

#----------------------------------------------------------------------------
def cos(x):
    """
    Uncertain number cosine function

    """
    try:
        return x._cos()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.cos(x)
        elif isinstance(x,numbers.Complex):
            return cmath.cos(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )
#----------------------------------------------------------------------------
def tan(x):
    """
    Uncertain number tangent function

    """
    try:
        return x._tan()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.tan(x)
        elif isinstance(x,numbers.Complex):
            return cmath.tan(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )

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
    try:
        return x._asin()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.asin(x)
        elif isinstance(x,numbers.Complex):
            return cmath.asin(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )
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
    try:
        return x._acos()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.acos(x)
        elif isinstance(x,numbers.Complex):
            return cmath.acos(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )
            
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
    try:
        return x._atan()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.atan(x)
        elif isinstance(x,numbers.Complex):
            return cmath.atan(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )
            
#----------------------------------------------------------------------------
def atan2(y,x):
    """
    Two-argument uncertain number arctangent function    

    :arg x: abscissa
    :type x: :class:`~lib.UncertainReal`
    :arg y: ordinate
    :type y: :class:`~lib.UncertainReal`
    
    .. note::   this function is not defined for uncertain complex numbers
                (use :func:`phase`)

    **Example**::

        >>> x = ureal(math.sqrt(3)/2,1)
        >>> y = ureal(0.5,1)
        >>> theta = atan2(y,x)
        >>> theta
        ureal(0.5235987755982989,1.0,inf)
        >>> math.degrees( theta.x )
        30.000000000000004
    
    """
    try:
        return y._atan2(x)
    except AttributeError:
        try:
            return x._ratan2(y)
        except AttributeError:
            if isinstance(x,numbers.Real) and isinstance(y,numbers.Real):
                return math.atan2(y,x)
            else:
                raise TypeError(
                    "illegal arguments: x={!r} y={!r}".format(x,y)
                )

#---------------------------------------------------------------------------
def sinh(x):
    """
    Uncertain number hyperbolic sine function

    """
    try:
        return x._sinh()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.sinh(x)
        elif isinstance(x,numbers.Complex):
            return cmath.sinh(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )
#---------------------------------------------------------------------------
def cosh(x):
    """
    Uncertain number hyperbolic cosine function

    """
    try:
        return x._cosh()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.cosh(x)
        elif isinstance(x,numbers.Complex):
            return cmath.cosh(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )
    
#---------------------------------------------------------------------------
def tanh(x):
    """
    Uncertain number hyperbolic tangent function

    """
    try:
        return x._tanh()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.tanh(x)
        elif isinstance(x,numbers.Complex):
            return cmath.tanh(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )
            
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
    try:
        return x._asinh()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.asinh(x)
        elif isinstance(x,numbers.Complex):
            return cmath.asinh(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )

#---------------------------------------------------------------------------
def acosh(x):
    """
    Uncertain number hyperbolic arc-cosine function

    .. note::
        In the complex case there is one branch cut,
        extending left from 1 along the
        real axis to :math:`-\infty`, continuous from above.
        
    """
    try:
        return x._acosh()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.acosh(x)
        elif isinstance(x,numbers.Complex):
            return cmath.acosh(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )

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
    try:
        return x._atanh()
    except AttributeError:
        if isinstance(x,numbers.Real):
            return math.atanh(x)
        elif isinstance(x,numbers.Complex):
            return cmath.atanh(x)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(x)
            )
            
#----------------------------------------------------------------------------
def magnitude(x):
    """
    Return the magnitude of ``x``

    .. note::

        If ``x`` is not an uncertain number type,
        returns :func:`abs(x)<abs>`.    
    
    """
    try:
        return x._magnitude()
    except AttributeError:
        return abs(x)

#---------------------------------------------------------------------------
def phase(z):
    """
    :arg z: an uncertain complex number    
    :type z: :class:`~lib.UncertainComplex`

    :returns:   the phase in radians
    :rtype:     :class:`~lib.UncertainReal`
    
    """
    try:
        return z._phase()
    except AttributeError:
        if isinstance(z,numbers.Complex):
            return cmath.phase(z)
        else:
            raise TypeError(
                "illegal argument: {!r}".format(z)
            )

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

# import here to avoid circular imports
from GTC import linear_algebra
la = linear_algebra

#============================================================================    
if __name__ == "__main__":
    import doctest       
    # from GTC import *
    doctest.testmod(  optionflags=doctest.NORMALIZE_WHITESPACE )
