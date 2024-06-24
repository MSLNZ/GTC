# -*- coding: utf-8 -*-
import locale
import re
import math
import cmath
from collections import namedtuple

from GTC import (
    inf,
    inf_dof,
)

from GTC.named_tuples import (
    StandardUncertainty,
    FormattedUncertainReal,
    FormattedUncertainComplex,
)

from GTC.reporting import (
    is_ureal, 
    is_ucomplex,
)

__all__ = (
    'apply_format',
    'create_format',
    'to_string'
)

# The regular expression to parse a format specification (format_spec)
# with additional (and optional) character(s) at the end for GTC-specific fields.
#
# format_spec ::= [[fill]align][sign][#][0][width][grouping][.precision][type][style]
# https://docs.python.org/3/library/string.html#format-specification-mini-language
_format_spec_regex = re.compile(
    # the builtin grammar fields
    r'((?P<fill>.)(?=[<>=^]))?'
    r'(?P<align>[<>=^])?'
    r'(?P<sign>[ +-])?'
    r'(?P<hash>#)?'
    r'(?P<zero>0)?'
    r'(?P<width>\d+)?'
    r'(?P<grouping>[_,])?'
    r'((\.)(?P<precision>\d+))?'
    r'(?P<type>[bcdeEfFgGnosxX%])?'
    
    # Latex or Unicode -- these characters cannot be in <type>
    r'(?P<style>[LU])?'
    
    # the regex must match until the end of the string
    r'$'
)

_exponent_regex = re.compile(r'[eE][+-]\d+')

_unicode_superscripts = {
    ord('+'): '\u207A',
    ord('-'): '\u207B',
    ord('0'): '\u2070',
    ord('1'): '\u00B9',
    ord('2'): '\u00B2',
    ord('3'): '\u00B3',
    ord('4'): '\u2074',
    ord('5'): '\u2075',
    ord('6'): '\u2076',
    ord('7'): '\u2077',
    ord('8'): '\u2078',
    ord('9'): '\u2079',
}

_Rounded = namedtuple('Rounded', 'value precision type exponent suffix')


class Format:
    """Format specification for an uncertain number.

    Do not instantiate this class directly. The proper way to create a
    *Format* object is via the :func:`create_format` function.

    This class does not have any user-facing attributes. It is only meant to
    be passed as an argument to :func:`apply_format` or :func:`to_string`.

    .. versionadded:: 1.4.0
    """

    def __init__(self, **kwargs):
        def get(key, default):
            value = kwargs.get(key)
            if value is None:
                return default
            return value

        # builtin grammar fields
        self._fill = get('fill', '')
        self._align = get('align', '')
        self._sign = get('sign', '')
        self._hash = get('hash', '')
        self._zero = get('zero', '')
        self._width = get('width', '')
        self._grouping = get('grouping', '')
        self._precision = int(get('precision', 2))
        self._type = get('type', 'f')

        # GTC grammar fields
        self._style = get('style', '')

        # these attributes are used when rounding
        self._digits = int(get('digits', 2))
        self._u_exponent = int(get('u_exponent', 0))
        self._df_precision = int(get('df_precision', 0))
        self._r_precision = int(get('r_precision', 3))

        # keeps a record of whether the Format was created for
        # an uncertain number with an uncertainty of 0, NaN or INF
        self._nonzero_and_finite = bool(get('nonzero_and_finite', True))

    def __repr__(self):
        # Use .digits instead of .precision in the result.
        # This will allow users to see what the equivalent format_spec
        # string would be, instead of them importing and calling
        # create_format() to simply print an uncertain number.
        spec = f'{self._fill}{self._align}{self._sign}{self._hash}{self._zero}{self._width}{self._grouping}' \
               f'.{self._digits}{self._type}{self._style}'
        return f'Format(format_spec={spec!r}, df_precision={self._df_precision}, r_precision={self._r_precision})'

    def _result(self, text):
        """Formats the result.

        Uses the fill, align, zero and width format-specification fields.

        :param text: The text to format.
        :type text: str

        :return: The `text` formatted.
        :rtype: str
        """
        fmt = f'{self._fill}{self._align}{self._zero}{self._width}s'
        return f'{text:{fmt}}'

    def _value(self, value, precision=None, type=None, sign=None, hash=None):
        """Format a value.

        Uses the sign, hash symbol, grouping, precision and type
        format-specification fields.

        :param value: The value to format.
        :type value: int, float, complex
        :param precision: Indicates how many digits should be displayed after
                          the decimal point for presentation types f and F,
                          or before and after the decimal point for
                          presentation types g or G.
        :type precision: int
        :param type: Can be one of: e, E, f, F, g, G, n
        :type type: str
        :param sign: Can be one of: +, -, ' ' (a space)
        :type sign: str
        :param hash: Can be either # or '' (an empty string)
        :type hash: str

        :return: The `value` formatted.
        :rtype: str
        """
        if sign is None:
            sign = self._sign

        if precision is None:
            precision = self._precision

        if type is None:
            type = self._type

        if hash is None:
            hash = self._hash

        if type == 'n':
            fmt = f'%{sign}{hash}.{precision}f'
            return locale.format_string(fmt, value, grouping=True)

        return f'{value:{sign}{hash}{self._grouping}.{precision}{type}}'

    def _uncertainty(self, uncertainty, precision=None, type='f', hash=None):
        """Format an uncertainty.

        Uses the sign, hash symbol, grouping, precision and type
        format-specification fields.

        :param uncertainty: The uncertainty to format.
        :type uncertainty: float
        :param precision: Indicates how many digits should be displayed after
                          the decimal point for presentation types f and F,
                          or before and after the decimal point for
                          presentation types g or G.
        :type precision: int
        :param type: Can be one of: e, E, f, F, g, G, n
        :type type: str
        :param hash: Can be either # or '' (an empty string)
        :type hash: str

        :return: The `uncertainty` formatted.
        :rtype: str
        """
        return self._value(uncertainty, precision=precision,
                           type=type, sign='', hash=hash)


def parse(format_spec):
    """Parse a format specification into its grammar fields.

    :param format_spec: A format specification. Supports the builtin fields
                        (see :ref:`formatspec`) plus additional characters,
                        that must occur after the builtin ``type`` field,
                        which are used to decide how an uncertain number
                        will be converted to a string.
    :type format_spec: str

    :return: The grammar fields.
    :rtype: dict
    """
    match = _format_spec_regex.match(format_spec)
    if not match:
        raise ValueError(f'Invalid format specifier {format_spec!r}')
    return match.groupdict()


def apply_format(un, fmt):
    """Apply the format to an uncertain number.

    .. versionadded:: 1.4.0

    :param un: An uncertain number.
    :type un: :class:`~.lib.UncertainReal` or :class:`~.lib.UncertainComplex`

    :param fmt: The format to apply to `un`. 
    :type fmt: :class:`Format`

    :rtype: :obj:`~named_tuples.FormattedUncertainReal` or :obj:`~named_tuples.FormattedUncertainComplex`
    
    .. note::
    
        Although the format type may be specified as ``'%'``, this will be interpreted as ``'f'`` and the value and uncertainty will
        not be multiplied by 100. See :func:`create_format` for more details.
    """
    if fmt._type == '%':
        # JSB: It is a bad idea to apply type=% to an uncertain number. In the
        # string representation it is okay because a "%" symbol is printed at
        # the end; however, in a FormattedUncertain* object a user would not
        # have an obvious way of knowing whether the value and uncertainty
        # were multiplied by 100.
        #
        # Create a copy of Format so that the % to f substitution is only valid
        # in the scope of this function. This ensures that the input Format
        # object remains unaltered for the end user.
        #
        # Since dropping support for Python 2 is already planned, just call
        # items() instead of checking whether to call iteritems() or items().
        kwargs = dict((k[1:], v) for k, v in vars(fmt).items())
        kwargs['type'] = 'f'
        fmt = Format(**kwargs)

    if is_ureal(un):
        x, u = _round_ureal(un, fmt)
        dof = _truncate_dof(un.df, fmt._df_precision)
        if fmt._nonzero_and_finite:
            x_value = x.value
        else:
            x_value = round(x.value, x.precision)
        return FormattedUncertainReal(x_value, u.value, dof, un.label)
    elif is_ucomplex(un):
        re_x, re_u = _round_ureal(un.real, fmt)
        im_x, im_u = _round_ureal(un.imag, fmt)
        df = _truncate_dof(un.df, fmt._df_precision)
        r = round(un.r, fmt._r_precision)
        if fmt._nonzero_and_finite:
            real, imag = re_x.value, im_x.value
        else:
            real = round(re_x.value, re_x.precision)
            imag = round(im_x.value, im_x.precision)
        return FormattedUncertainComplex(
            complex(real, imag),
            StandardUncertainty(re_u.value, im_u.value),
            r, df, un.label)
    else:
        raise RuntimeError(f"unexpected type: {un!r}")


def create_format(obj, digits=None, df_precision=None, r_precision=None,
                  style=None, **kwargs):
    r"""Create a format specification.

    Formatting an uncertain number rounds the value and the uncertainty to the
    specified number of significant digits (based on the uncertainty component),
    and truncates the degrees of freedom and rounds the correlation coefficient
    to the specified precision (the number of digits after the decimal point).

    .. versionadded:: 1.4.0

    :param obj: An object to use to create the format specification. If an
        :class:`~.lib.UncertainReal` or an :class:`~.lib.UncertainComplex`,
        then the uncertainty component is used when creating the format.
        Otherwise this function assumes that the uncertainty component was
        passed in as `obj`.
    :type obj: :class:`float`, :class:`complex`, :class:`~.lib.UncertainReal`,
        :class:`~.lib.UncertainComplex` or :class:`~.named_tuples.StandardUncertainty`

    :param digits: The number of significant digits in the uncertainty
        component to retain. Default is 2.
    :type digits: :class:`int`

    :param df_precision: The number of digits that should be kept after the
        decimal point for the degrees of freedom. The value is truncated.
        Default is 0.
    :type df_precision: :class:`int`

    :param r_precision: The number of digits that should be kept after the
        decimal point for the correlation coefficient. The value is rounded.
        Default is 3.
    :type r_precision: :class:`int`

    :param style: The style to use when formatting an uncertain number as a
        string. Can be either ``'L'`` (:math:`\LaTeX`) or ``'U'`` (Unicode).
        When a style is used, the + sign and any leading 0's are removed from
        the exponent, for example, ``e+06`` becomes :math:`10^{6}` instead of
        :math:`10^{+06}`. Also, if the exponential term is equal to ``e+00``
        then the exponential term is completely removed (i.e., it becomes an
        empty string instead of :math:`10^{+00}`). Default is to not use
        styling.
    :type style: :class:`str`

    :param \**kwargs:

        All additional keyword arguments correspond to the format-specification
        fields (see :ref:`formatspec`). These fields are used when formatting
        an uncertain number as a string.

        - *fill* (:class:`str`): Can be any character, except for
          ``'{'`` or ``'}'``.
        - *align* (:class:`str`): Can be one of ``'<'``, ``'>'``,
          ``'='`` or ``'^'``.
        - *sign* (:class:`str`): Can be one of ``'+'``, ``'-'`` or
          ``' '`` (i.e., a *space*).
        - *hash* (:class:`bool`): Whether to include the ``#`` field.
        - *zero* (:class:`bool`): Whether to include the ``0`` field.
        - *width* (:class:`int`): The total width of the string.
        - *grouping* (:class:`str`): Can be one of ``','`` or ``'_'``.
        - *type* (:class:`str`): Can be one of ``'e'``, ``'E'``, ``'f'``,
          ``'F'``, ``'g'``, ``'G'``, ``'n'`` or ``'%'``.

        .. note::
           The ``'%'`` type applies to both the value and standard uncertainty.
           In keeping with the behaviour of ``'%'`` for floats, the value and 
           standard uncertainty will be multiplied by 100 and displayed in 
           fixed ``'f'`` format followed by a percent sign.

        .. note::
           The *precision* field is treated differently for uncertain
           numbers in GTC. The *digits* keyword argument (number of
           significant digits) is used instead of the typical concept
           of *precision* (number of digits, either after the decimal
           place or in total depending on the value of *type*).

        .. note::
           If the uncertainty component is 0 then the string representation
           of the uncertain number does not include the uncertainty in
           parentheses and *digits* is equivalent to *precision*.

           >>> ur = ureal(3.1415926536, 0)
           >>> f'{ur:.5f}'
           '3.14159'

    :rtype: :class:`Format`
    """
    # need to check for spelling mistakes of a named keyword argument since this
    # can be frustrating for an end user if, for example, digit=3 was specified.
    # The proper name to use is digits=3. In this situation the digits value would
    # remain at 2 and this would confuse the user why they only see 2 significant
    # digits in the result since no error was raised in their code notifying them
    # of their spelling mistake
    expected = ('fill', 'align', 'sign', 'hash', 'zero',
                'width', 'grouping', 'precision', 'type')
    for key in kwargs:
        if key not in expected:
            raise ValueError(f'Unrecognised argument {key!r}')

    kwargs['style'] = style
    kwargs['df_precision'] = df_precision
    kwargs['r_precision'] = r_precision
    kwargs['hash'] = '#' if kwargs.get('hash') else ''
    kwargs['zero'] = '0' if kwargs.get('zero') else ''

    if digits is None:
        kwargs['digits'] = kwargs.get('precision')
    else:
        kwargs['digits'] = digits

    try:
        u = obj.u
    except AttributeError:
        u = obj

    fmt = Format(**kwargs)
    _update_format(u, fmt)

    if fmt._style not in ('', 'L', 'U'):
        raise ValueError(
            f'Formatting style {fmt._style!r} is not supported. '
            f'Must be L or U'
        )

    if fmt._digits <= 0:
        raise ValueError(
            f'The number of digits must be > 0 '
            f'[digits={fmt._digits}]'
        )

    if fmt._type == 'n' and fmt._grouping:
        raise ValueError(f"Cannot specify {fmt._grouping!r} with 'n'")

    return fmt


def to_string(obj, fmt):
    """Convert a numeric object to a string.

    .. versionadded:: 1.4.0

    :param obj: A numeric object.
    :type obj: :class:`int`, :class:`float`, :class:`complex`,
        :class:`~.lib.UncertainReal`, :class:`~.lib.UncertainComplex`
        or :class:`~.named_tuples.StandardUncertainty`

    :param fmt: The format to use to convert `obj` to a string.
        See :func:`create_format` for more details.
    :type fmt: :class:`Format`

    :rtype: :class:`str`
    """
    def move_percent_symbol(text):
        symbol = r'\%' if fmt._style == 'L' else '%'
        return f"{text.replace(symbol, '')}{symbol}"

    if isinstance(obj, (int, float)):
        r = _round(obj, fmt)
        v_str = fmt._value(r.value, precision=r.precision, type=r.type)
        result = _stylize(v_str + r.suffix, fmt)
        return fmt._result(result)

    if isinstance(obj, (complex, StandardUncertainty)):
        r = _round(obj.real, fmt)
        re_val = fmt._value(r.value, precision=r.precision, type=r.type)
        re_str = _stylize(re_val + r.suffix, fmt)

        i = _round(obj.imag, fmt)
        im_val = fmt._value(i.value, precision=i.precision, type=i.type, sign='+')
        im_str = _stylize(im_val + i.suffix, fmt)

        b1, b2 = _stylize('(', fmt), _stylize(')', fmt)
        result = f'{b1}{re_str}{im_str}j{b2}'
        if fmt._type == '%':
            result = move_percent_symbol(result)
        return fmt._result(result)

    if is_ureal(obj):
        real, imag = obj, None
    elif is_ucomplex(obj):
        real, imag = obj.real, obj.imag
    else:
        raise RuntimeError(f"unexpected type: {obj!r}")

    result = _stylize(_to_string_ureal(real, fmt), fmt)
    if imag is not None:
        imag_str = _to_string_ureal(imag, fmt, sign='+')
        b1, b2 = _stylize('(', fmt), _stylize(')', fmt)
        result = f'{b1}{result}{_stylize(imag_str, fmt)}j{b2}'
        if fmt._type == '%':
            result = move_percent_symbol(result)

    return fmt._result(result)


def _nan_or_inf(*args):
    """Check if any of the args are infinity or NaN.

    args: float, complex

    returns: bool
    """
    for arg in args:
        # TODO use cmath.isfinite and math.isfinite when
        #  dropping Python 2.7 support
        if isinstance(arg, complex):
            if cmath.isinf(arg) or cmath.isnan(arg):
                return True
        else:
            if math.isinf(arg) or math.isnan(arg):
                return True
    return False


def _order_of_magnitude(value):
    """Return the order of magnitude of `value`.

    value: float

    returns: int

    Examples
    --------
    0.0123 -> -2
    0.123  -> -1
    0      ->  0
    1.23   ->  0
    12.3   ->  1
    """
    if value == 0:
        return 0
    return int(math.floor(math.log10(math.fabs(value))))


def _update_format(uncertainty, fmt):
    """Update the `precision` and `u_exponent` attributes of `fmt`.

    `fmt` gets modified, so this function does not return anything.

    uncertainty: float, complex, StandardUncertainty
    fmt: Format

    returns: None
    """
    if isinstance(uncertainty, (complex, StandardUncertainty)):
        # Real and imaginary component uncertainties are different.
        # Find the lesser uncertainty.
        u = min(uncertainty.real, uncertainty.imag)
        # However, if one component has no uncertainty use the other
        if u == 0:
            u = max(uncertainty.real, uncertainty.imag)
    else:
        u = uncertainty

    if u == 0 or _nan_or_inf(u):
        fmt._precision = fmt._digits
        fmt._nonzero_and_finite = False
        return

    exponent = _order_of_magnitude(u)
    if exponent - fmt._precision + 1 >= 0:
        fmt._precision = 0
    else:
        fmt._precision = int(fmt._precision - exponent + 1)

    u_exponent = exponent - fmt._digits + 1

    # edge case, for example, if 0.099 rounds to 0.1
    rounded = round(u, -u_exponent)
    e_rounded = _order_of_magnitude(rounded)
    if e_rounded > exponent:
        u_exponent += 1

    fmt._u_exponent = u_exponent


def _stylize(text, fmt):
    """Apply the formatting style to `text`.

    text: str
    fmt: Format

    returns: the stylized text
    """
    if not fmt._style or not text:
        return text

    replacements = []
    exponent = ''
    exp_number = None
    exp_match = _exponent_regex.search(text)
    if exp_match:
        # don't care whether it starts with e or E and
        # don't want to include the + symbol
        group = exp_match.group()
        exp_number = int(group[1:])

    if fmt._style == 'U':
        if exp_match and exp_number != 0:
            e = f'{exp_number}'
            translated = e.translate(_unicode_superscripts)
            exponent = f'\u00D710{translated}'

    elif fmt._style == 'L':
        if exp_match and exp_number != 0:
            exponent = rf'\times10^{{{exp_number}}}'

        replacements = [
            ('(', r'\left('),
            (')', r'\right)'),
            ('nan', r'\mathrm{NaN}'),
            ('NAN', r'\mathrm{NaN}'),
            ('inf', r'\infty'),  # must come before 'INF'
            ('INF', r'\infty'),
            ('%', r'\%'),
        ]

    else:
        assert False, 'should not get here'

    if exp_match:
        start, end = exp_match.span()
        text = f'{text[:start]}{exponent}{text[end:]}'

    for old, new in replacements:
        text = text.replace(old, new)

    return text


def _round(value, fmt, exponent=None):
    """Round `value` to the appropriate number of significant digits.

    value: int | float
    fmt: Format
    exponent: int | None

    returns: _Rounded
    """
    if not fmt._nonzero_and_finite or _nan_or_inf(value):
        return _Rounded(value, fmt._precision, fmt._type, 0, '')

    if exponent is None:
        exponent = _order_of_magnitude(value)

    _type = fmt._type
    f_or_g_as_f = (_type in 'fF') or \
                  ((_type in 'gGn') and
                   (-4 <= exponent < exponent - fmt._u_exponent))

    if f_or_g_as_f:
        factor = 1.0
        digits = -fmt._u_exponent
        precision = max(digits, 0)
        suffix = ''
    elif _type == '%':
        factor = 0.01
        digits = -fmt._u_exponent - 2
        precision = max(digits, 0)
        suffix = '%'
    else:
        factor = 10. ** exponent
        digits = max(exponent - fmt._u_exponent, 0)
        precision = digits
        suffix = f'{factor:.0{_type}}'[1:]

    if _type in 'eg%':
        _type = 'f'
    elif _type in 'EG':
        _type = 'F'

    val = round(value / factor, digits)
    return _Rounded(val, precision, _type, exponent, suffix)


def _truncate_dof(dof, precision):
    """Truncate the degrees of freedom to the specified precision.

    dof: float
    precision: int

    returns: `dof` rounded
    """
    if _nan_or_inf(dof):
        return dof

    if dof > inf_dof:
        return inf

    factor = 10. ** (-precision)
    return round(factor * math.floor(dof / factor), precision)


def _round_ureal(ureal, fmt):
    """Round an UncertainReal.

    This function ensures that both x and u get scaled by the same factor.

    ureal: UncertainReal
    fmt: Format

    :return: tuple -> (x: _Rounded, u: _Rounded)
    """
    x, u = ureal.x, ureal.u
    maximum = round(max(math.fabs(x), u), -fmt._u_exponent)
    rounded = _round(maximum, fmt)
    x_rounded = _round(x, fmt, exponent=rounded.exponent)
    u_rounded = _round(u, fmt, exponent=rounded.exponent)
    return x_rounded, u_rounded


def _to_string_ureal(ureal, fmt, sign=None):
    """Convert an UncertainReal to a string.

    ureal: UncertainReal
    fmt: Format
    sign: str, one of <space> + -

    returns: `ureal` as a string
    """
    x, u = ureal.x, ureal.u

    if u == 0:
        return fmt._result(fmt._value(x, sign=sign))

    if _nan_or_inf(x, u):
        x_str = fmt._value(x, sign=sign)
        u_str = fmt._uncertainty(u, type=None)
        result = f'{x_str}({u_str})'
        # move an exponential term (if it exists) to the end of the string
        exp = _exponent_regex.search(result)
        if exp:
            start, end = exp.span()
            result = f'{result[:start]}{result[end:]}{exp.group()}'
        return result

    x_rounded, u_rounded = _round_ureal(ureal, fmt)

    u_r = u_rounded.value
    oom = _order_of_magnitude(u_r)
    precision = x_rounded.precision

    if precision > 0 and oom >= 0:
        # the uncertainty straddles the decimal point so
        # keep the decimal point in the result
        u_str = fmt._uncertainty(u_r, precision=precision, type=u_rounded.type)
    else:
        hash_, type_ = None, u_rounded.type
        if oom < 0:
            if fmt._hash:
                hash_ = ''
            else:
                type_ = 'f'
        u_str = fmt._uncertainty(round(u_r * 10. ** precision),
                                 precision=0, type=type_, hash=hash_)

    x_str = fmt._value(x_rounded.value, precision=precision, sign=sign,
                       type=x_rounded.type)

    return f'{x_str}({u_str}){x_rounded.suffix}'
