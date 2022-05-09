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

As an example, consider an uncertain number that has a value of 3.14159e-6 with a
standard uncertainty of 1.3247e-8 and 9.246 degrees of freedom

    >>> un = ureal(3.14159e-6, 1.3247e-8, 9.246)

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
    '0.000003142(13)'

The second is to use the :func:`~GTC.formatting.to_string` function

    >>> to_string(un, fmt)
    '0.000003142(13)'

Scientific notation is also supported. For example,

    >>> '{:e}'.format(un)
    '3.142(13)e-06'

Alternatively, the :func:`~GTC.formatting.create_format` and
:func:`~GTC.formatting.to_string` functions can be used

    >>> fmt = create_format(un, type='e')
    >>> to_string(un, fmt)
    '3.142(13)e-06'

The order of characters in the `format_spec` is important. Python supports
a specific grammar when using the :func:`format` function (see :ref:`formatspec`).
The GTC-specific field -- *style* -- must occur after the builtin
fields. The equivalent format-specification grammar for GTC is:

``[[fill]align][sign][#][0][width][grouping][.digits][type][style]``

Note the use *digits* (not *precision*) and *style* (there are two styles: ``L`` (:math:`\LaTeX`) and ``U`` (Unicode)).

The following examples illustrate for an uncertain number.


Use 3 significant digits, scientific notation and bracket mode

    >>> fmt = create_format(un, digits=3, type='e')
    >>> to_string(un, fmt)
    '3.1416(132)e-06'
    >>> '{:.3e}'.format(un)
    '3.1416(132)e-06'


Use 2 significant digits, scientific notation, and unicode style

    >>> fmt = create_format(un, digits=2, type='e', style='U')
    >>> to_string(un, fmt)
    '3.142(13)×10⁻⁶'
    >>> '{:.2EU}'.format(un)
    '3.142(13)×10⁻⁶'

Use 4 significant digits, scientific notation, and :math:`\LaTeX` style

    >>> fmt = create_format(un, digits=4, type='e', style='L')
    >>> to_string(un, fmt)
    '3.14159\\left(1325\\right)\\times10^{-6}'
    >>> '{:.4eL}'.format(un)
    '3.14159\\left(1325\\right)\\times10^{-6}'

In this case, the text output may not be easy to interpret in Python, but 
it becomes :math:`3.14159\left(1325\right)\times10^{-6}` when
rendered in a :math:`\LaTeX` document.

Fill with ``*``, align right ``>``, use a ``+`` sign, a width of 20 characters and 1 significant digit

    >>> fmt = create_format(un, fill='*', align='>', sign='+', width=20, digits=1)
    >>> to_string(un, fmt)
    '******+0.00000314(1)'
    >>> '{:*>+20.1}'.format(un)
    '******+0.00000314(1)'

Generating formatted attributes
-------------------------------

The :func:`~GTC.formatting.apply_format` function formats an uncertain number.
The returned object is a :obj:`~collections.namedtuple` with attributes for the value,
uncertainty, degrees of freedom, and correlation coefficient (for uncertain
complex numbers) that have been manipulated to have exactly the specified number of digits.

    >>> fmt = create_format(un)
    >>> formatted = apply_format(un, fmt)
    >>> formatted
    FormattedUncertainReal(x=3.142e-06, u=1.3e-08, df=9, label=None, si_prefix=None)
    >>> formatted.x
    3.142e-06

..
    We can specify that 3 significant digits will be used for the uncertainty (and value) and 
    2 digits of precision for the degrees of freedom

        >>> fmt = create_format(un, digits=3, df_precision=2)
        >>> formatted = apply_format(un, fmt)
        >>> formatted
        FormattedUncertainReal(x=3.1416, u=0.0132, df=9.24, label=None, si_prefix=None)
        >>> formatted.x
        3.1416
        >>> formatted.df
        9.24

The following functions are available.

.. automodule:: GTC.formatting
   :members: apply_format, create_format, to_string, Format
   :member-order: bysource
