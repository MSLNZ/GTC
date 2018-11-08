"""
Reporting functions
-------------------

    *   The function :func:`budget` produces an uncertainty budget.
    *   The function :func:`k_factor` returns the coverage factor 
        used for real-valued problems (based on the Student-t distribution). 
    *   The function :func:`k_to_dof` returns the degrees of freedom 
        corresponding to a given coverage factor and coverage probability.
    *   The function :func:`k2_factor_sq` returns   
        coverage factor squared for the complex-valued problem. 
    *   The function :func:`k2_to_dof` returns the degrees of freedom 
        corresponding to a given coverage factor and coverage probability
        in complex-valued problems.
    *   Functions :func:`u_bar` and :func:`v_bar` return summary values 
        for matrix results associated with 2-D uncertainty.

Uncertainty functions
---------------------

    *   The function :func:`u_component` returns the signed 
        component of uncertainty in one uncertain number 
        due to uncertainty in another.

Type functions
--------------

    *   The function :func:`is_ureal` can be used to
        identify uncertain real numbers. 
    *   The function :func:`is_ucomplex` can be used to
        identify uncertain complex numbers. 

Module contents
---------------

"""
from __future__ import division     # True division

import math
import numbers

from operator import attrgetter as getter
from functools import reduce

from scipy import special, optimize

try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip
    long = int

from GTC.lib import (
    std_variance_real,
    std_variance_covariance_complex,
    UncertainReal,
    UncertainComplex,
    welch_satterthwaite,
    willink_hall,
    _is_uncertain_real_constant,
    _is_uncertain_complex_constant
)
from GTC.named_tuples import (
    ComponentOfUncertainty, 
    Influence
)
from GTC.vector import extend_vector

from GTC import (
    is_sequence,
    inf,
    inf_dof,
)

__all__ = (
    'budget',
    'k_factor',
    'k_to_dof',
    'k2_factor_sq',
    'k2_to_dof',
    'u_component',
    'is_ureal',
    'is_ucomplex',
    'v_bar',
    'u_bar',
)

#--------------------------------------------------------------------------
uid_str = lambda id: "{0[0]:d}_{0[1]:d}".format(id)

#--------------------------------------------------------------------------
def is_ureal(x):
    """Return ``True`` if ``x`` is an uncertain real number
    
    **Example**::

        >>> x = ureal(1,1)
        >>> reporting.is_ureal(x)
        True
        
    """
    return isinstance(x,UncertainReal)
    
#--------------------------------------------------------------------------
def is_ucomplex(z):
    """Return ``True`` if ``z`` is an uncertain complex number
    
    **Example**::

        >>> z = ucomplex(1+2j,(0.1,0.2))
        >>> reporting.is_ucomplex(z)
        True

    """
    return isinstance(z,UncertainComplex)

#------------------------------------------------------------
def _df_k2(k2,p,nu1,TOL):
    """
    Return `nu2` such that the integral of 
    F(nu1,nu2) from -infty to `x` is `p`
    
    `x` is k2**2 * nu2/ ( nu1*(nu2+1) )
    
    """
    # We have `k2` the integral limit, so `pf` gives us `p`
    # we must vary the `nu2` argument until the
    # returned value equals `p`.
    # `fdtr` returns the integral of F probability density from -infty to `x`
    def fn(nu2):
        x = k2**2 * nu2/ ( nu1*(nu2+1) )  
        # return pf(x,nu1,nu2) - p 
        return special.fdtr(nu1,nu2,x) - p 
    
    # dof here is nu2-1 and cannot be less than 2
    # This setting of `lo` is not a strict bound, because
    # the calculation will succeed, we just don't want to 
    # go there.
    
    lo = 1 - 1E-3   
    fn_lo = fn(lo)
    if fn_lo > 0.0:
        raise RuntimeError(
            "dof < 2 cannot be calculated"
        )
        
    upper_limit = (20,50,1E2,1E3,inf_dof)
    for hi in upper_limit:
        if fn(hi) > 0.0: 
            return optimize.ridder(fn,lo,hi)
        else:    
            lo = hi
        
    return inf       
#----------------------------------------------------------------------------
def k2_to_dof(k2,p=95):
    """Return the dof corresponding to a bivariate coverage factor `k2`  
    
    :arg k2: coverage factor (>0)
    :arg p: coverage probability (%)
    :type k2: float
    :type p: int or float

    Evaluates a number of degrees-of-freedom given a coverage 
    factor for an elliptical uncertainty region with coverage 
    probability ``p`` based on the F-distribution.
    
    **Example**::

        >>> reporting.k2_to_dof(2.6,95)
        34.35788424389927
        
    """
    if k2 <= 0:
        raise RuntimeError( "invalid k:  {}".format(k2) ) 
    if p <= 0 or p >= 100:
        raise RuntimeError( "invalid p: {}".format(p) )
    else:
        p = p / 100.0     

    return _df_k2(k2,p,2,1E-7) + 1

#----------------------------------------------------------------------------
def k2_factor_sq(df=inf,p=95):
    """Return a squared coverage factor for an elliptical uncertainty region

    :arg df: the degrees-of-freedom (>=2)
    :arg p: the coverage probability (%)
    :type df: float
    :type p: int or float

    Evaluates the square of the coverage factor for an elliptical uncertainty 
    region with coverage probability ``p``  and ``df`` degrees of freedom
    based on the F-distribution.
    
    **Example**::

        >>> reporting.k2_factor_sq(3)
            56.99999999999994
    
    """
    p = p / 100.0
    
    if df > inf_dof:
        return -2.0 * math.log(1-p)
        
    elif(df>1):   
        # norm = l * (n-1) / (n - l) in the general
        # 'l'-dimensional case for 'n' observations
        # here l = 2, df = n-1
        norm = 2*df / (df-1)
        
        # `fdtri` is the inverse of the cumulative F distribution
        # returning `x` such that `fdtr(dfn, dfd, x) = p`
        return norm*special.fdtri(2.0,df-1.0,p)
        
    else:
        raise RuntimeError("invalid df={!r}".format( df ) )
 
#----------------------------------------------------------------------------
def k_factor(df=inf,p=95):
    """Return the a coverage factor for an uncertainty interval

    :arg df: the degrees-of-freedom (>1)
    :arg p: the coverage probability (%)
    :type df: float
    :type p: int or float

    Evaluates the coverage factor for an uncertainty interval
    with coverage probability ``p`` and degrees-of-freedom ``df``
    based on the Student t-distribution. 
    
    **Example**::
    
        >>> reporting.k_factor(3)
        3.182446305284263

    """
    if p <= 0.0 or p >= 100.0:
        raise RuntimeError( "invalid p: {}".format( p ) )
    
    p = (1.0 + p/100.0)/2.0
    
    if df > inf_dof:
        # inverse cumulative Gaussian distribution
        return special.ndtri(p)
    elif df >= 1:
        # inverse cumulative Student-t distribution
        return special.stdtrit(df,p)
    else:
        raise RuntimeError( "invalid df: {}".format( df ) )
   
#----------------------------------------------------------------------------
def k_to_dof(k,p=95):
    """Return the dof corresponding to a univariate coverage factor `k` 
    
    :arg k: coverage factor (>0)
    :arg p: coverage probability (%)
    :type k: float
    :type p: int or float

    Evaluates the degrees-of-freedom given a coverage factor for 
    an uncertainty interval with coverage probability ``p``
    based on the Student t-distribution.
    
    **Example**::

        >>> reporting.k_to_dof(2.0,95)
        60.43756442698591
        
    """
    if k <= 0:
        raise RuntimeError( "invalid k:  {}".format( k ) )  
    if p <= 0 or p >= 100:
        raise RuntimeError( "invalid p: {}".format( p ) )
    else:
        p = (1.0 + p/100.0)/2.0         
        df = special.stdtridf(p,k) 
        
        return df if df < inf_dof else inf 

#----------------------------------------------------------------------------
def u_component(y,x):
    """Return the component of uncertainty in ``y`` due to ``x``
    
    If ``x`` and ``y`` are uncertain real numbers, return a float. 

    If ``y`` or ``x`` is an uncertain complex number, return 
    a 4-element sequence of float, containing the components of 
    the uncertainty matrix.

    Otherwise, return 0.

    **Example**::

        >>> x = ureal(3,1)
        >>> y = 3 * x
        >>> reporting.u_component(y,x)
        3.0

        >>> q = ucomplex(2,1)
        >>> r = ucomplex(3,1)
        >>> z = q * r
        >>> reporting.u_component(z,q)
        ComponentOfUncertainty(rr=3.0, ri=-0.0, ir=0.0, ii=3.0)
        
        >>> q = ucomplex(2,1)
        >>> z = magnitude(q)    # uncertain real numbers
        >>> reporting.u_component(z,q)
        ComponentOfUncertainty(rr=1.0, ri=0.0, ir=0.0, ii=0.0)
        
    """
    if isinstance(y,UncertainReal):
        if isinstance(x,UncertainReal):
            if x.is_elementary:
                if x._node.independent:
                    return y._u_components.get(x._node,0.0)
                else:
                    return y._d_components.get(x._node,0.0)
                    
            elif x.is_intermediate:
                # Because `x` is an intermediate, if `y` depends on it at all
                # there will be an entry in `_i_components` 
                return y._i_components.get(x._node,0.0)
                
            elif _is_uncertain_real_constant(x):
                return 0
                
            else:
                raise RuntimeError(
                    "`x` is not an elementary or intermediate uncertain number"
                )
            
        elif isinstance(x,UncertainComplex):
            result = [0.0,0.0,0.0,0.0]
            for i,x_i in enumerate( (x.real, x.imag) ):
                if x_i.is_elementary:
                    if x_i._node.independent:
                        u_i = y._u_components.get(x_i._node,0.0)
                    else:
                        u_i = y._d_components.get(x_i._node,0.0)
                        
                elif x_i.is_intermediate:
                    u_i = y._i_components.get(x_i._node,0.0)

                elif _is_uncertain_complex_constant(x):
                    u_i = 0 
                    
                else:
                    raise RuntimeError(
                        "An elementary or intermediate " 
                        + "uncertain number is expected: {!r}".format(x)
                    )
                result[i] = u_i
            
            return ComponentOfUncertainty(*result)
                    
        elif isinstance(x,numbers.Real):
            return 0.0
            
        elif isinstance(x,numbers.Complex):
            return ComponentOfUncertainty(0.0,0.0,0.0,0.0)
            
        else:
            assert False, 'unexpected'
        
    elif isinstance(y,UncertainComplex):
        if isinstance(x,UncertainComplex):
            x_re, x_im  = x.real, x.imag
            y_re, y_im = y.real, y.imag
            
            # require 4 partial derivatives:
            #   dy_re_dx_re, dy_re_dx_im, dy_im_dx_re, dy_im_dx_im
            if x.real.is_elementary and x.imag.is_elementary:
                if x.real._node.independent:
                    dy_re_dx_re = y_re._u_components.get(x_re._node,0.0)
                    dy_re_dx_im = y_re._u_components.get(x_im._node,0.0)
                    dy_im_dx_re = y_im._u_components.get(x_re._node,0.0)
                    dy_im_dx_im = y_im._u_components.get(x_im._node,0.0)
                else:
                    dy_re_dx_re = y_re._d_components.get(x_re._node,0.0)
                    dy_re_dx_im = y_re._d_components.get(x_im._node,0.0)
                    dy_im_dx_re = y_im._d_components.get(x_re._node,0.0)
                    dy_im_dx_im = y_im._d_components.get(x_im._node,0.0)
                
                return ComponentOfUncertainty(dy_re_dx_re, dy_re_dx_im, dy_im_dx_re, dy_im_dx_im)
                
            elif x.real.is_intermediate and x.imag.is_intermediate:
                dy_re_dx_re = y_re._i_components.get(x_re._node,0.0)
                dy_re_dx_im = y_re._i_components.get(x_im._node,0.0)
                dy_im_dx_re = y_im._i_components.get(x_re._node,0.0)
                dy_im_dx_im = y_im._i_components.get(x_im._node,0.0) 
            
                return ComponentOfUncertainty(dy_re_dx_re, dy_re_dx_im, dy_im_dx_re, dy_im_dx_im)
                
            elif _is_uncertain_complex_constant(x):
                return ComponentOfUncertainty(0.0,0.0,0.0,0.0)
                
            else:
                raise RuntimeError(
                    "An elementary or intermediate " 
                    + "uncertain number was expected: {!r}".format(x)
                )
                
        elif isinstance(x,UncertainReal):
            y_re, y_im = y.real, y.imag

            if x.is_elementary:
                if x._node.independent:
                    dy_re_dx_re = y_re._u_components.get(x._node,0.0)
                    dy_im_dx_re = y_im._u_components.get(x._node,0.0)
                else:
                    dy_re_dx_re = y_re._d_components.get(x._node,0.0)
                    dy_im_dx_re = y_im._d_components.get(x._node,0.0)
                
                return ComponentOfUncertainty(dy_re_dx_re, 0.0, dy_im_dx_re, 0.0)
                
            elif x.is_intermediate:
                dy_re_dx_re = y_re._i_components.get(x._node,0.0)
                dy_im_dx_re = y_im._i_components.get(x._node,0.0)

                return ComponentOfUncertainty(dy_re_dx_re, 0.0, dy_im_dx_re, 0.0)
 
            elif _is_uncertain_real_constant(x):
                return ComponentOfUncertainty(0.0,0.0,0.0,0.0)
                
            else:
                raise RuntimeError(
                    "An elementary or intermediate " 
                    + "uncertain number is expected: {!r}".format(x)
                )
            
        elif isinstance(x,numbers.Complex):
            return ComponentOfUncertainty(0.0,0.0,0.0,0.0)
    else:
        raise RuntimeError(
            "An uncertain number is expected: {!r}".format(y)
        )

#----------------------------------------------------------------------------
def u_bar(ucpt):
    """Return the magnitude of a component of uncertainty
    
    :arg ucpt: a component of uncertainty
    :type ucpt: float or 4-element sequence of float

    If ``ucpt`` is a sequence, return the root-sum-square 
    of the elements divided by :math:`\sqrt{2}`

    If ``ucpt`` is a number, return the absolute value.

    **Example**::

        >>> x1 = 1-.5j
        >>> x2 = .2+7.1j
        >>> z1 = ucomplex(x1,1)
        >>> z2 = ucomplex(x2,1)
        >>> y = z1 * z2
        >>> dy_dz1 = reporting.u_component(y,z1)
        >>> dy_dz1
        ComponentOfUncertainty(rr=0.2, ri=-7.1, ir=7.1, ii=0.2)
        >>> reporting.u_bar(dy_dz1)
        7.102816342831905
    
    """
    if is_sequence(ucpt):
        if len(ucpt) != 4:
            raise RuntimeError(
                "need a 4-element sequence, got: {!r}".format(ucpt)
            )
           
        return math.sqrt( reduce(lambda y,x: y + x*x,ucpt,0) / 2 )
        
    elif isinstance(ucpt,numbers.Real):
        return abs(ucpt)
        
    else:
        raise RuntimeError(
            "need a 4-element sequence or float, got: {!r}".format(ucpt)
        )
        
#----------------------------------------------------------------------------
def v_bar(cv):
    """Return the trace of ``cv`` divided by 2 
    
    :arg cv: a variance-covariance matrix
    :type cv: 4-element sequence of float
    
    :returns: float

    **Example**::
    
        >>> x1 = 1-.5j
        >>> x2 = .2+7.1j
        >>> z1 = ucomplex(x1,(1,.2))
        >>> z2 = ucomplex(x2,(.2,1))
        >>> y = z1 * z2
        >>> y.v
        VarianceCovariance(rr=2.3464, ri=1.8432, ir=1.8432, ii=51.4216)
        >>> reporting.v_bar(y.v)
        26.884

    """
    assert len(cv) == 4,\
           "'%s' a 4-element sequence is needed" % type(cv)

    return (cv[0] + cv[3]) / 2.0

#----------------------------------------------------------------------------
def budget(y,influences=None,key='u',reverse=True,trim=0.01,max_number=None):
    """Return a sequence of label-component of uncertainty pairs

    :arg y:  an uncertain number
    :type y: :class:`~lib.UncertainReal` or :class:`~lib.UncertainComplex`

    :arg influences:  a sequence of uncertain numbers

    :arg key: the list sorting key

    :arg reverse:  determines sorting order (forward or reverse)
    :type reverse: bool

    :arg trim:  remove components of uncertainty that are
                less than ``trim`` times the largest component
    
    :arg max_number: return no more than ``max_number`` components
    
    A sequence of :obj:`~named_tuples.Influence` namedtuples is 
    returned, each with the attributes ``label`` and ``u`` for a 
    component of uncertainty (see :func:`~core.component`). 

    The argument ``influences`` can be used to select the influences
    are that reported.

    The argument ``key`` can be used to order the sequence
    by the component of uncertainty or the label (``u`` or ``label``).

    The argument ``reverse`` controls the sense of ordering.
    
    The argument ``trim`` can be used to set a minimum relative 
    magnitude of components returned. Set ``trim=0`` for a 
    complete list.

    The argument ``max_number`` can be used to restrict the 
    number of components returned.  

    **Example**::

        >>> x1 = ureal(1,1,label='x1')
        >>> x2 = ureal(2,0.5,label='x2')
        >>> x3 = ureal(3,0.1,label='x3')
        >>> y = (x1 - x2) / x3
        >>> for l,u in reporting.budget(y):
        ... 	print("{0}: {1:G}".format(l,u))
        ... 	
        x1: 0.333333
        x2: 0.166667
        x3: 0.0111111
        
        >>> for l,u in reporting.budget(y,reverse=False):
        ... 	print("{0}: {1:G}".format(l,u))
        ... 	
        x3: 0.0111111
        x2: 0.166667
        x1: 0.333333
        
    """    
    if isinstance(y,UncertainReal):
        if influences is None:
            nodes = y._u_components.keys()
            labels = [ n_i.label 
                        if n_i.label is not None else "{}".format(n_i.uid) 
                           for n_i in nodes ]
            values = [ math.fabs( u ) for u in y._u_components.itervalues() ]
        else:
            labels = []
            values = []
            for i in influences:
                if isinstance(i,UncertainReal):
                    labels.append( i.label )
                    values.append( math.fabs(u_component(y,i)) ) 
                elif isinstance(i,UncertainComplex):
                    labels.append( i.real.label )
                    values.append( math.fabs(u_component(y,i.real)) ) 
                    labels.append( i.imag.label )
                    values.append( math.fabs(u_component(y,i.imag)) ) 
                else:
                    assert False,\
                           "unexpected type: '{}'".format( type(i) )

        if len(values):
            cut_off = max(values) * float(trim)
            this_budget = [ Influence( label=n, u=u )
                            for (u,n) in izip(values,labels) if u >= cut_off ]
        else:
            this_budget = [ ]
        
    elif isinstance(y,UncertainComplex):        
        if influences is None:
            
            # Ensure that the influence vectors have the same keys
            re = extend_vector(y.real._u_components,y.imag._u_components)    
            im = extend_vector(y.imag._u_components,y.real._u_components)

            try:
                labels = []
                values = []
                it_re = re.iteritems()
                it_im = im.iteritems()
                
                while True:
                    ir_0,ur_0 = next(it_re)
                    ii_0,ui_0 = next(it_im)

                    if hasattr(ir_0,'complex'):
                        ir_1,ur_1 = next(it_re)
                        ii_1,ui_1 = next(it_im)
                        
                        if ir_0.label is None:
                            # No label assigned, report uids
                            label = "uid({},{})".format(
                                uid_str(ir_0.uid),uid_str(ii_0.uid)
                            )
                        else:
                            # take the trailing _re off the real label
                            # to label the complex influence
                            label = ir_0.label[:-3]

                        u=u_bar([ur_0,ur_1,ui_0,ui_1])
                        labels.append(label)
                        values.append(u)
                        
                    else:
                        # Not wrt a complex influence
                        if ir_0.label is None:
                            label = "uid({})".format( uid_str(ir_0.uid) )
                        else:
                            label = ir_0.label 
                            
                        u=u_bar([ur_0,0,ui_0,0])
                        labels.append(label)
                        values.append(u)
                        
            except StopIteration:
                pass
        else:
            labels = [ i.label for i in influences ]
            values = [ u_bar( u_component(y,i) ) for i in influences ]
            
        if len(values):
            cut_off = max(values) * float(trim)
            this_budget = [ 
                Influence( label=n, u=u ) 
                    for (u,n) in izip(values,labels) 
                        if u >= cut_off  
            ]
            
        else:   
            this_budget = []
    else:
        this_budget = []

    if key is not None:
        this_budget.sort( key=getter(key),reverse=reverse )
    
    if max_number is not None and len(this_budget) > max_number:
        this_budget = this_budget[:max_number]
        
    return this_budget
        
#============================================================================
if __name__ == "__main__":
    import doctest
    from GTC import *
    doctest.testmod(  optionflags=doctest.NORMALIZE_WHITESPACE )