.. _formatting-uncertain-numbers:

Formatting Uncertain Numbers
============================

Formatting an uncertain number is useful for display purposes. When an uncertain
number is formatted the value and the uncertainty are rounded to the specified
number of significant digits (based on the uncertainty component) and the degrees
of freedom and the correlation coefficient are rounded to the specified precision
(the number of digits after the decimal point) so that an uncertain number can be
displayed in a more user-friendly manner.

There are three functions available to help format and display uncertain numbers:

* :func:`~GTC.formatting.create_format`
* :func:`~GTC.formatting.apply_format`
* :func:`~GTC.formatting.to_string`

.. invisible-code-block: pycon

    >>> SKIP_IF_PYTHON_27()

As an example, consider an uncertain number that has a value of 3.14159e-6 with a
standard uncertainty of 1.3247e-8 and 9.246 degrees of freedom

    >>> un = ureal(3.14159e-6, 1.3247e-8, 9.246)

We create an object to control the format of this uncertain number using the
:func:`~GTC.formatting.create_format` function

    >>> fmt = create_format(un)

The *fmt* object contains the information necessary to format *un*

    >>> fmt
    Format(format_spec='.2fB', df_precision=0, r_precision=3)

The information contained in the previous output indicates that 2 significant
digits in the uncertainty component will be retained, ``.2``, fixed-point
notation is used for the value and the uncertainty, ``f``, the value and
uncertainty will be represented using the bracket mode, ``B``, the degrees
of freedom will be rounded to a precision of 0 digits after the decimal point,
and, the correlation coefficient will be rounded to a precision of 3 digits
after the decimal point.

Note that the ``.2f`` field in the `format_spec` value is treated differently in
GTC. Typically, this field corresponds to the *precision* that is used for displaying
floating-point numbers and refers to the number of digits after the decimal place.
In GTC, this field corresponds to the number of *significant digits* to retain in
the uncertainty component.

There are two ways to display the string representation of `un`. The first way
is to use the value of the `format_spec` with the builtin :func:`format` function

    >>> '{:.2fB}'.format(un)
    '0.000003142(13)'

and the second way is to use the :func:`~GTC.formatting.to_string` function

    >>> to_string(un, fmt)
    '0.000003142(13)'

To make this result more readable, we can use scientific notation

    >>> '{:e}'.format(un)
    '3.142(13)e-06'

The equivalent way using the :func:`~GTC.formatting.create_format` and
:func:`~GTC.formatting.to_string` functions is

    >>> fmt = create_format(un, type='e')
    >>> to_string(un, fmt)
    '3.142(13)e-06'

There are other options for formatting the string representation of an
uncertain number. There are two format modes [``B`` (Bracket) and ``P``
(Plus-minus)], two formatting styles [``L`` (:math:`\LaTeX`) and ``U``
(Unicode)], and, an option for using an SI prefix [``S``].

The order of the characters in the `format_spec` is important. Python supports
a specific grammar when using the :func:`format` function (see :ref:`formatspec`).
The GTC-specific fields -- *mode*, *style*, *si* -- must occur after the builtin
fields. The equivalent format-specification grammar for GTC is:

``[[fill]align][sign][#][0][width][grouping][.digits][type][mode][style][si]``

Note the use and location of *digits*, *mode*, *style* and *si*.

The following examples illustrate how to combine these options to get
the desired string representation of an uncertain number.

Use 1 significant digit, scientific notation and plus-minus mode

    >>> fmt = create_format(un, digits=1, type='e', mode='P')
    >>> to_string(un, fmt)
    '(3.14+/-0.01)e-06'
    >>> '{:.1eP}'.format(un)
    '(3.14+/-0.01)e-06'

Use 3 significant digits, scientific notation and bracket mode

    >>> fmt = create_format(un, digits=3, type='e', mode='B')
    >>> to_string(un, fmt)
    '3.1416(132)e-06'
    >>> '{:.3eB}'.format(un)
    '3.1416(132)e-06'

Use 2 significant digits, scientific notation, plus-minus mode, and unicode style

    >>> fmt = create_format(un, digits=2, type='e', mode='P', style='U')
    >>> to_string(un, fmt)
    '(3.142±0.013)×10⁻⁶'
    >>> '{:.2ePU}'.format(un)
    '(3.142±0.013)×10⁻⁶'

Use 1 significant digit, bracket mode, unicode style and an SI prefix

    >>> fmt = create_format(un, digits=1, mode='B', style='U', si=True)
    >>> to_string(un, fmt)
    '3.14(1) µ'
    >>> '{:.1BUS}'.format(un)
    '3.14(1) µ'

Use 4 significant digits, scientific notation, bracket mode, and :math:`\LaTeX` style

    >>> fmt = create_format(un, digits=4, type='e', mode='B', style='L')
    >>> to_string(un, fmt)
    '3.14159\\left(1325\\right)\\times10^{-6}'
    >>> '{:.4eBL}'.format(un)
    '3.14159\\left(1325\\right)\\times10^{-6}'

Although the output text may not be easy to interpret in Python, when the text is
rendered in a :math:`\LaTeX` document it becomes :math:`3.14159\left(1325\right)\times10^{-6}`

Fill with ``*``, align right ``>``, use a ``+`` sign, a width of 20 characters and 1 significant digit

    >>> fmt = create_format(un, fill='*', align='>', sign='+', width=20, digits=1)
    >>> to_string(un, fmt)
    '******+0.00000314(1)'
    >>> '{:*>+20.1}'.format(un)
    '******+0.00000314(1)'

The :func:`~GTC.formatting.apply_format` function formats an uncertain number.
The returned object is a :obj:`~collections.namedtuple` with the value,
uncertainty, degrees of freedom, and correlation coefficient (for uncertain
complex numbers) rounded to the specified number of digits.

    >>> fmt = create_format(un)
    >>> formatted = apply_format(un, fmt)
    >>> formatted
    FormattedUncertainReal(x=3.142e-06, u=1.3e-08, df=9, label=None, si_prefix=None)
    >>> formatted.x
    3.142e-06

We can specify that the format should use 3 significant digits in the uncertainty
component, 2 digits of precision for the degrees of freedom and use an SI prefix

    >>> fmt = create_format(un, digits=3, df_precision=2, si=True)
    >>> formatted = apply_format(un, fmt)
    >>> formatted
    FormattedUncertainReal(x=3.1416, u=0.0132, df=9.24, label=None, si_prefix='u')
    >>> formatted.x
    3.1416
    >>> formatted.df
    9.24

The following functions are available.

.. automodule:: GTC.formatting
   :members: apply_format, create_format, to_string, Format
   :member-order: bysource
