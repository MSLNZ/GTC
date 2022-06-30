.. _formatting-uncertain-numbers:

Formatting Uncertain Numbers
============================

Formatting an uncertain number is useful for display purposes. 
The number of significant digits for the value and the uncertainty
can be controlled, based on the uncertainty.
The number of digits displayed after the decimal point for the degrees of freedom 
and the correlation coefficient can also be specified. 

Three functions help to format and display uncertain numbers:

* :func:`~GTC.formatting.create_format`
* :func:`~GTC.formatting.apply_format`
* :func:`~GTC.formatting.to_string`


.. invisible-code-block: pycon

    >>> SKIP_IF_PYTHON_27()

Generating strings
------------------

As an example, consider the following uncertain number

    >>> un = ureal(3.14159e-3, 2.71828e-4, 9.876)

We create an object to control the format using the
:func:`~GTC.formatting.create_format` function

    >>> fmt = create_format(un)

The *fmt* object contains formatting specifications

    >>> fmt
    Format(format_spec='.2f', df_precision=0, r_precision=3)

Fixed-point notation, ``f``, is used. The ``.2`` indicates that 2 significant digits will be retained in the uncertainty 
and the value will be displayed using the same number of significant digits. 
The degrees of freedom will be truncated to a precision of 0 digits after the decimal point,
and the correlation coefficient will be rounded to a precision of 3 digits
after the decimal point.

Note that the meaning of the ``.2f`` field in the `format_spec` is different
from the usual Python interpretation. 
In GTC, this field specifies the number of *significant digits* of the uncertainty 
that will be displayed, which also affects the number of digits displayed for the value.

There are two ways to use a format specification to display the string representation of an uncertain number. 
The first, is to use the builtin :func:`format` function

    >>> '{:.2f}'.format(un)
    '0.00314(27)'

The second is to use the :func:`~GTC.formatting.to_string` function

    >>> to_string(un, fmt)
    '0.00314(27)'

Scientific notation is also supported. For example,

    >>> '{:e}'.format(un)
    '3.14(27)e-03'

Alternatively, the :func:`~GTC.formatting.create_format` and
:func:`~GTC.formatting.to_string` functions can be used

    >>> fmt = create_format(un, type='e')
    >>> to_string(un, fmt)
    '3.14(27)e-03'

The order of characters in the `format_spec` is important. Python supports
a specific grammar when using the :func:`format` function (see :ref:`formatspec`).
The GTC-specific field -- *style* -- must occur after the builtin
fields. The equivalent format-specification grammar for GTC is:

.. centered:: [[fill]align][sign][#][0][width][grouping][.digits][type][style]

Note the use of *digits* (not *precision*) and *style* (there are two styles:
``L`` (:math:`\LaTeX`) and ``U`` (Unicode)).

The following examples illustrate for an uncertain number.


Use 3 significant digits and scientific notation

    >>> fmt = create_format(un, digits=3, type='e')
    >>> to_string(un, fmt)
    '3.142(272)e-03'
    >>> '{:.3e}'.format(un)
    '3.142(272)e-03'


Use 1 significant digit, scientific notation, and unicode style

    >>> fmt = create_format(un, digits=1, type='e', style='U')
    >>> to_string(un, fmt)
    '3.1(3)×10⁻³'
    >>> '{:.1eU}'.format(un)
    '3.1(3)×10⁻³'

Use 4 significant digits, scientific notation, and :math:`\LaTeX` style

    >>> fmt = create_format(un, digits=4, type='e', style='L')
    >>> to_string(un, fmt)
    '3.1416\\left(2718\\right)\\times10^{-3}'
    >>> '{:.4eL}'.format(un)
    '3.1416\\left(2718\\right)\\times10^{-3}'

In this case, the text output may not be easy to interpret in Python, but 
it becomes :math:`3.1416\left(2718\right)\times10^{-3}` when
rendered in a :math:`\LaTeX` document.

Fill with ``*``, align right ``>``, show a ``+`` sign with positive values, a width of 20 characters and 1 significant digit

    >>> fmt = create_format(un, fill='*', align='>', sign='+', width=20, digits=1)
    >>> to_string(un, fmt)
    '**********+0.0031(3)'
    >>> '{:*>+20.1}'.format(un)
    '**********+0.0031(3)'

Generating formatted attributes
-------------------------------

The :func:`~GTC.formatting.apply_format` function formats an uncertain number.
The returned object is a :obj:`~collections.namedtuple` with numerical attributes that have been manipulated to have the specified number of digits for: value, uncertainty, degrees of freedom, and correlation coefficient (for uncertain complex numbers).

    >>> fmt = create_format(un)
    >>> formatted = apply_format(un, fmt)
    >>> formatted
    FormattedUncertainReal(x=0.00314, u=0.00027, df=9, label=None)
    >>> formatted.x
    0.00314

We can specify that 3 significant digits will be used for the uncertainty (and value) and
truncate the degrees of freedom to 2 digits of precision

    >>> fmt = create_format(un, digits=3, df_precision=2)
    >>> formatted = apply_format(un, fmt)
    >>> formatted
    FormattedUncertainReal(x=0.003142, u=0.000272, df=9.87, label=None)
    >>> formatted.x
    0.003142
    >>> formatted.df
    9.87

The following functions are available.

.. automodule:: GTC.formatting
   :members: apply_format, create_format, to_string, Format
   :member-order: bysource
