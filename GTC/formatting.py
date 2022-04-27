# -*- coding: utf-8 -*-
import re
import math
import cmath
from collections import namedtuple

from GTC import (
    inf,
    inf_dof,
)
from GTC.named_tuples import StandardUncertainty

# The regular expression to parse a format specification (format_spec)
# with additional (and optional) characters at the end for GTC-specific fields.
#
# format_spec ::= [[fill]align][sign][#][0][width][grouping][.precision][type][mode][style][si]
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
    
    # Bracket or Raw "+/-"
    # NOTE: these characters cannot be in <type>
    r'(?P<mode>[BR])?'

    # Latex or Unicode
    # NOTE: these characters cannot be in <type> nor <mode>
    r'(?P<style>[LU])?'

    # SI prefix
    # NOTE: this character cannot be in <type>, <mode> nor <style>
    r'(?P<si>S)?'
    
    # the regex must match until the end of the string
    r'$'
)

_exponent_regex = re.compile(r'[eE][+-]\d+')

# TODO replace u'' with '' when dropping support for
#  Python 2.7. There a multiple occurrences in this module.
_exponent_table = {
    ord('e'): u'\u00D710',
    ord('E'): u'\u00D710',
    ord('+'): u'\u207A',
    ord('-'): u'\u207B',
    ord('0'): u'\u2070',
    ord('1'): u'\u00B9',
    ord('2'): u'\u00B2',
    ord('3'): u'\u00B3',
    ord('4'): u'\u2074',
    ord('5'): u'\u2075',
    ord('6'): u'\u2076',
    ord('7'): u'\u2077',
    ord('8'): u'\u2078',
    ord('9'): u'\u2079',
}

_si_map = {i*3: pre for i, pre in enumerate('yzafpnum kMGTPEZY', start=-8)}

_Rounded = namedtuple('Rounded', 'value precision type exponent suffix')

# TODO review the typename value
_FormattedUncertainReal = namedtuple('FormattedUncertainReal', 'x u df label')
_FormattedUncertainComplex = namedtuple('FormattedUncertainComplex', 'x u r df label')


class Format(object):

    def __init__(self, **kwargs):
        """Format specification of an uncertain number.

        Do not instantiate this class directly.
        """
        def get(key, default):
            value = kwargs.get(key)
            if value is None:
                return default
            return value

        # builtin grammar fields
        self.fill = get('fill', '')
        self.align = get('align', '')
        self.sign = get('sign', '')
        self.hash = get('hash', '')
        self.zero = get('zero', '')
        self.width = get('width', '')
        self.grouping = get('grouping', '')
        self.precision = int(get('precision', 2))
        self.type = get('type', 'f')

        # GTC grammar fields
        self.mode = get('mode', 'B')
        self.style = get('style', '')
        self.si = get('si', '')

        # these attributes are used when rounding
        self.digits = int(get('digits', 2))
        self.u_exponent = 0
        self.df_precision = int(get('df_precision', 0))
        self.r_precision = int(get('r_precision', 3))

    def __repr__(self):
        spec = '{fill}{align}{sign}{hash}{zero}{width}{grouping}' \
               '.{digits}{type}{mode}{style}{si}'.format(
                fill=self.fill,
                align=self.align,
                sign=self.sign,
                hash=self.hash,
                zero=self.zero,
                width=self.width,
                grouping=self.grouping,
                digits=self.digits,  # use digits, not precision
                type=self.type,
                mode=self.mode,
                style=self.style,
                si=self.si)
        return 'Format(format_spec={!r}, df_precision={}, r_precision={})'.format(
            spec, self.df_precision, self.r_precision)

    def result(self, text):
        """Formats the result.

        Uses the fill, align, zero and width format-specification fields.

        :param text: The text to format.
        :type text: str

        :return: The `text` formatted.
        :rtype: str
        """
        return u'{0:{fill}{align}{zero}{width}s}'.format(
            text,
            fill=self.fill,
            align=self.align,
            zero=self.zero,
            width=self.width
        )

    def value(self, value, precision=None, type=None, sign=None):
        """Format a value.

        Uses the sign, hash symbol, grouping, precision and type
        format-specification fields.

        :param value: The value to format.
        :type value: int, float, complex
        :param precision: Indicates how many digits should be displayed after
                          the decimal point for presentation types ``f`` and
                          ``F``, or before and after the decimal point for
                          presentation types ``g`` or ``G``.
        :type precision: int
        :param type: Can be one of: ``e``, ``E``, ``f``, ``F``, ``g`` or ``G``
        :type type: str
        :param sign: Can be one of: ``+``, ``-``, ``' '`` (i.e., a 'space')
        :type sign: str

        :return: The `value` formatted.
        :rtype: str
        """
        if sign is None:
            sign = self.sign

        if precision is None:
            precision = self.precision

        if type is None:
            type = self.type

        return '{0:{sign}{hash}{grouping}.{precision}{type}}'.format(
            value,
            sign=sign,
            hash=self.hash,
            grouping=self.grouping,
            precision=precision,
            type=type
        )

    def uncertainty(self, uncertainty, precision=None, type='f'):
        """Format an uncertainty.

        Uses the sign, hash symbol, grouping, precision and type
        format-specification fields.

        :param uncertainty: The uncertainty to format.
        :type uncertainty: float
        :param precision: Indicates how many digits should be displayed after
                          the decimal point for presentation types ``f`` and
                          ``F``, or before and after the decimal point for
                          presentation types ``g`` or ``G``.
        :type precision: int
        :param type: Can be one of: ``e``, ``E``, ``f``, ``F``, ``g`` or ``G``
        :type type: str

        :return: The `uncertainty` formatted.
        :rtype: str
        """
        return self.value(uncertainty, precision=precision, type=type, sign='')


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
        raise ValueError('Invalid format specifier {!r}'.format(format_spec))
    return match.groupdict()


def apply_format(un, fmt):
    """Apply the format to an uncertain number.

    .. versionadded:: 1.4.0

    :param un: An uncertain number.
    :type un: :class:`~GTC.lib.UncertainReal` or :class:`~GTC.lib.UncertainComplex`
    :param fmt: The format to apply to `un`. See :func:`create_format`.
    :type fmt: :class:`Format`

    :return: The uncertain number with the format applied.
    :rtype collections.namedtuple
    """
    try:
        # TODO Need to know if `obj` is UncertainReal or UncertainComplex.
        #  We could check isinstance(), but then we will need to deal with
        #  circular import issues with lib.py.
        #  An UncertainReal object has no attribute _value.
        un._value
        real, imag = un.real, un.imag

        re_x = _round(real.x, fmt).value
        re_u = _round(real.u, fmt).value
        im_x = _round(imag.x, fmt).value
        im_u = _round(imag.u, fmt).value
        df = _round_dof(un.df, fmt.df_precision)
        r = round(un.r, fmt.r_precision)

        return _FormattedUncertainComplex(
            complex(re_x, im_x),
            StandardUncertainty(re_u, im_u),
            r, df, un.label)

    except AttributeError:
        x = _round(un.x, fmt).value
        u = _round(un.u, fmt).value
        dof = _round_dof(un.df, fmt.df_precision)
        return _FormattedUncertainReal(x, u, dof, un.label)


def create_format(obj, digits=None, df_precision=None, r_precision=None,
                  mode=None, style=None, si=None, **kwargs):
    r"""Create a format specification.

    .. versionadded:: 1.4.0

    :param obj: An object to use create the format specification.
    :type obj: float, complex, :class:`~GTC.lib.UncertainReal`,
               :class:`~GTC.lib.UncertainComplex`
               or :class:`~GTC.named_tuples.StandardUncertainty`
    :param digits: The number of significant digits in the uncertainty
                   component to retain. Default is 2.
    :type digits: int
    :param df_precision: The number of decimal places for the
                         degrees-of-freedom to retain. Default is 0.
    :type df_precision: int
    :param r_precision: The number of decimal places for the correlation
                        coefficient to retain. Default is 3.
    :type r_precision: int
    :param mode: The mode to use. Must be one of:

               - B: bracket notation, e.g., 3.142(13)
               - R: raw plus-minus notation, e.g., 3.142+/-0.013 (TODO link to GUM)

    :type mode: str
    :param style: The style to use. One of:

               - L: latex notation, stuff like \infty \times \pm \mathrm
               - U: unicode, e.g., (12.3±5.0)×10⁻¹² or 12.3(2)×10⁶

    :type style: str
    :param si: Whether to use an SI prefix. Only the truthiness of the
               value is checked, so it can be any type and value.
               Enabling this feature would convert, for example,
               1.23(6)e+07 to 12.3(6) M.
    :type si: Any

    :param \**kwargs:

            All additional keyword arguments correspond to the
            format-specification fields (see :ref:`formatspec`).

            * fill (:class:`str`): Can be any character, except for ``{`` or ``}``.
            * align (:class:`str`): Can be one of ``<``, ``>`` or ``^``.
            * sign (:class:`str`): Can be one of ``+``, ``-`` or ``' '`` (i.e., a 'space').
            * hash (Any): Whether to include the ``#`` symbol. Only the
                          truthiness of the value is checked, so it can
                          be any type and value.
            * zero (Any): Whether to include the ``0`` symbol. Only the
                          truthiness of the value is checked, so it
                          can be any type and value.
            * width (:class:`int`): The width of the returned string.
            * grouping (:class:`str`): Can be one of ``,`` or ``_``.
            * type (:class:`str`): Can be one of ``e`, ``E``, ``f``, ``F``,
                                   ``g``, ``G``, ``n`` or ``%``
                TODO should % only make the uncertainty be a percentage of the
                 value instead of operating on both the value and uncertainty?
                 Should it (and/or n) even be a supported option?

    :return: The format specification.
    :rtype: :class:`Format`
    """
    kwargs['mode'] = mode
    kwargs['style'] = style
    kwargs['df_precision'] = df_precision
    kwargs['r_precision'] = r_precision

    if digits is not None and digits < 1:
        raise ValueError('digits must be >= 1')
    kwargs['digits'] = digits or kwargs.get('precision')

    if si:
        kwargs['si'] = 'S'

    if kwargs.get('hash'):
        kwargs['hash'] = '#'

    if kwargs.get('zero'):
        kwargs['zero'] = '0'

    try:
        u = obj.u
    except AttributeError:
        u = obj

    fmt = Format(**kwargs)
    _determine_num_digits(u, fmt)
    return fmt


def to_string(obj, fmt):
    """Convert a numeric object to a string.

    .. versionadded:: 1.4.0

    :param obj: A numeric object.
    :type obj: int, float, complex, :class:`~GTC.lib.UncertainReal`,
               :class:`~GTC.lib.UncertainComplex`
               or :class:`~GTC.named_tuples.StandardUncertainty`
    :param fmt: The format to use to convert `obj`. See :func:`create_format`.
    :type fmt: :class:`Format`

    :return: The string representation of `obj`.
    :rtype: str
    """
    if isinstance(obj, (int, float)):
        r = _round(obj, fmt)
        v_str = fmt.value(r.value, precision=r.precision, type=r.type)
        result = _stylize(v_str + r.suffix, fmt)
        return fmt.result(result)

    if isinstance(obj, (complex, StandardUncertainty)):
        r = _round(obj.real, fmt)
        re_val = fmt.value(r.value, precision=r.precision, type=r.type)
        re_str = _stylize(re_val + r.suffix, fmt)

        i = _round(obj.imag, fmt)
        im_val = fmt.value(i.value, precision=i.precision, type=i.type, sign='+')
        im_str = _stylize(im_val + i.suffix, fmt)

        result = u'({0}{1}j)'.format(re_str, im_str)
        return fmt.result(result)

    try:
        # TODO Need to know if `obj` is UncertainReal or UncertainComplex.
        #  We could check isinstance(), but then we will need to deal with
        #  circular import issues with lib.py.
        #  An UncertainReal object has no attribute _value.
        obj._value
        real, imag = obj.real, obj.imag
    except AttributeError:
        real, imag = obj, None

    result = _stylize(_to_string_ureal(real, fmt), fmt)
    if imag is not None:
        imag_str = _to_string_ureal(imag, fmt, sign='+')
        result = u'({0}{1}j)'.format(result, _stylize(imag_str, fmt))
    return fmt.result(result)


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


def _determine_num_digits(uncertainty, fmt):
    """Determine the number of significant digits in `uncertainty`.

    The Format `fmt` gets modified, so this function does not return anything.

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

    # set these values for backwards compatibility
    if u == 0 or _nan_or_inf(u):
        fmt.precision = 6
        fmt.u_exponent = 0
        return

    exponent = _order_of_magnitude(u)
    if exponent - fmt.precision + 1 >= 0:
        fmt.precision = 0
    else:
        fmt.precision = int(fmt.precision - exponent + 1)

    u_exponent = exponent - fmt.digits + 1

    # edge case, for example, if 0.099 rounds to 0.1
    rounded = round(u, -u_exponent)
    e_rounded = _order_of_magnitude(rounded)
    if e_rounded > exponent:
        u_exponent += 1

    fmt.u_exponent = u_exponent


def _si_prefix_factor(exponent):
    """Determine the SI prefix and scaling factor.

    :param exponent: The exponent, e.g., 10 ** exponent
    :type exponent: int

    :returns: tuple -> (prefix: str, factor: float)
    """
    mod = exponent % 3
    prefix = _si_map.get(exponent - mod)
    factor = 10. ** mod
    if exponent < 0 and prefix is None:
        prefix = 'y'
        factor = 10. ** (exponent + 24)
    elif 0 <= exponent < 3:
        prefix = ''
        factor = 1.0
    elif prefix is None:
        prefix = 'Y'
        factor = 10. ** (exponent - 24)
    return prefix, factor


def _stylize(text, fmt):
    """Apply the formatting style to `text`.

    text: str
    fmt: Format

    returns: the stylized text
    """
    if not fmt.style:
        return text

    if fmt.style == 'U':
        exp = _exponent_regex.search(text)
        if exp:
            start, end = exp.span()
            e = u'{}'.format(exp.group())
            translated = e.translate(_exponent_table)
            text = u'{0}{1}{2}'.format(text[:start], translated, text[end:])

        mapping = {r'\+/\-': u'\u00B1', r'u': u'\u00B5'}
        for pattern, repl in mapping.items():
            text = re.sub(pattern, repl, text)
        return text

    raise ValueError(
        'The formatting style {!r} is not supported'.format(fmt.style)
    )


def _round(value, fmt, exponent=None):
    """Round `value` to the appropriate number of significant digits.

    value: int | float
    fmt: Format
    exponent: int | None

    returns: _Rounded
    """
    _type = 'F' if fmt.type in 'EFG' else 'f'

    if _nan_or_inf(value):
        # value precision type exponent suffix
        return _Rounded(value, 0, _type, 0, '')

    if exponent is None:
        exponent = _order_of_magnitude(value)

    f_or_g_as_f = (fmt.type in 'fF') or \
                  ((fmt.type in 'gG') and
                   (-4 <= exponent < exponent - fmt.u_exponent))

    if f_or_g_as_f:
        factor = 1.0
        precision = max(-fmt.u_exponent, 0)
        digits = -fmt.u_exponent
        suffix = ''
    else:
        factor = 10. ** exponent
        precision = max(exponent - fmt.u_exponent, 0)
        digits = precision
        suffix = '{0:.0{1}}'.format(factor, fmt.type)[1:]

    val = round(value / factor, digits)
    return _Rounded(val, precision, _type, exponent, suffix)


def _round_dof(dof, precision):
    """Round the degrees of freedom to the specified precision.

    dof: float
    precision: int

    returns: `dof` rounded
    """
    if _nan_or_inf(dof):
        return dof

    if dof > inf_dof:
        return inf

    factor = 10. ** (-precision)
    rounded = round(factor * math.floor(dof / factor), precision)
    # TODO if precision == 0 should an int be returned?
    if precision == 0:
        return int(rounded)
    return rounded


def _to_string_ureal(ureal, fmt, sign=None):
    """Convert an UncertainReal to a string.

    ureal: UncertainReal
    fmt: Format
    sign: str: <space> + -

    returns: `ureal` as a string
    """
    x, u = ureal.x, ureal.u

    if u == 0:
        # TODO Historically, UncertainReal did not include (0) and
        #  UncertainComplex did include (0) with the real and imaginary parts,
        #  e.g.,
        #    ureal(1.23, 0) -> ' 1.23'
        #    ucomplex(1.23+9.87j, 0) -> '(+1.23(0)+9.87(0)j)'
        #  We adopt the UncertainReal version -- do not include the (0)
        return fmt.result(fmt.value(x, sign=sign))

    if _nan_or_inf(x, u):
        x_str = fmt.value(x, sign=sign)
        u_str = fmt.uncertainty(u, type=None)

        if fmt.mode == 'B':
            result = '{0}({1})'.format(x_str, u_str)
        elif fmt.mode == 'R':
            result = '{0}+/-{1}'.format(x_str, u_str)
        else:
            raise ValueError(
                'The formatting mode {!r} is not supported. '
                'Must be B or R'.format(fmt.mode)
            )

        # if there is an exponential term in the result, move it
        # to the end of the string
        exp = _exponent_regex.search(result)
        if exp:
            start, end = exp.span()
            combined = [result[:start], result[end:], exp.group()]
            if fmt.mode == 'R':
                result = '({0}{1}){2}'.format(*combined)
            else:
                result = '{0}{1}{2}'.format(*combined)

        return result

    maximum = round(max(abs(x), u), -fmt.u_exponent)
    result = _round(maximum, fmt)
    exponent, precision = result.exponent, result.precision

    x_result = _round(x, fmt, exponent=exponent)
    u_result = _round(u, fmt, exponent=exponent)

    x_str = fmt.value(x_result.value, precision=precision, sign=sign, type='f')

    if fmt.mode == 'B':
        u_r = u_result.value
        if _order_of_magnitude(u_r) >= 0 and precision > 0:
            # the uncertainty straddles the decimal point so
            # keep the decimal point in the result
            u_str = fmt.uncertainty(u_r, precision=precision)
        else:
            u_str = fmt.uncertainty(
                round(u_r * 10. ** precision), precision=0)
        return '{0}({1}){2}'.format(x_str, u_str, result.suffix)

    elif fmt.mode == 'R':
        u_str = fmt.uncertainty(u_result.value, precision=precision)
        x_u_str = '{0}+/-{1}'.format(x_str, u_str)
        if result.suffix:
            return '({0}){1}'.format(x_u_str, result.suffix)
        return x_u_str

    raise ValueError(
        'The formatting mode {!r} is not supported. '
        'Must be B or R'.format(fmt.mode)
    )
