# -*- coding: utf-8 -*-
import re
import math
import cmath
from collections import (
    namedtuple,
    OrderedDict,
)

from GTC.named_tuples import StandardUncertainty

# The regular expression to parse a format specification (format_spec)
# with additional (and optional) characters at the end for GTC-specific formats
#
# format_spec ::= [[fill]align][sign][#][0][width][grouping][.precision][type][.GTC_df_decimals][GTC_mode][GTC_style]
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
    
    # number of degrees-of-freedom decimal places
    # NOTE: <precision> and/or <type> must also be specified for this to match
    r'((\.)(?P<df_decimals>\d+))?' 
    
    # Bracket or Raw "+/-"
    # NOTE: these characters cannot be in <type>
    r'(?P<mode>[BR])?'

    # Latex, Pretty or SI prefix
    # NOTE: these characters cannot be in <type> nor in <mode>
    r'(?P<style>[LPS])?'
    
    # the regex must match until the end of the string
    r'$'
)

_exponent_regex = re.compile(r'[eE][+-]\d+')

_exponent_table = {
    ord('e'): u' \u00D7 10',
    ord('E'): u' \u00D7 10',
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


Rounded = namedtuple('Rounded', 'value precision exponent exponent_str type')


class Format(object):

    def __init__(self, **kwargs):
        """Format specification of an uncertain number.

        Do not instantiate this class directly.
        """
        # builtin fields
        self.fill = kwargs.get('fill')
        self.align = kwargs.get('align')
        self.sign = kwargs.get('sign')
        self.hash = kwargs.get('hash')
        self.zero = kwargs.get('zero')
        self.width = kwargs.get('width')
        self.grouping = kwargs.get('grouping')
        self.precision = kwargs.get('precision')
        self.type = kwargs.get('type')

        # GTC fields
        self.df_decimals = kwargs.get('df_decimals')
        self.mode = kwargs.get('mode')
        self.style = kwargs.get('style')

        # these attributes are used for rounding the value and uncertainty
        self.digits = None
        self.u_exponent = None

    def __repr__(self):
        if self.df_decimals is None:
            df_decimals = ''
        else:
            df_decimals = '.{:d}'.format(self.df_decimals)
        return 'Format{%s%s%s%s}' % (self.format_spec, df_decimals,
                                     self.mode or '', self.style or '')

    def format(self, obj, **kwargs):
        """Format an object using the builtin :func:`format` function.

        By specifying keyword arguments you can override an attribute of
        the :class:`Format` instance before calling :func:`format` on `obj`.
        Overriding an attribute does not actually change the value of the
        attribute for the :class:`Format` instance. You can think of overriding
        as a temporary modification that occurs only when this :meth:`.format`
        method is called.

        Only the builtin fields can be overridden. To know the names of the
        builtin fields, you can look at the keys that are returned by
        :meth:`.format_spec_dict`. For more information see :ref:`formatspec`.

        :param obj: An object to pass to :func:`format`.

        :return: A formatted version of `obj`.
        :rtype: str
        """
        _dict = self.format_spec_dict
        for k, v in kwargs.items():
            if v is not None:
                # purposely use {:d} since width and precision must be integers
                if k == 'precision':
                    v = '.{:d}'.format(v)
                elif k == 'width':
                    v = '{:d}'.format(v)
            _dict[k] = v
        format_spec = self._join(_dict)
        return '{0:{1}}'.format(obj, format_spec)

    @property
    def format_spec(self):
        """str: Return the format specification as a string."""
        return self._join(self.format_spec_dict)

    @property
    def format_spec_dict(self):
        """dict: Return the format specification as a dictionary."""

        # purposely use {:d} since width and precision must be integers
        width = self.width
        if width is not None:
            width = '{:d}'.format(width)

        precision = self.precision
        if precision is not None:
            precision = '.{:d}'.format(precision)

        # TODO return a regular dict when dropping support for Python < 3.6
        #  since as of Python 3.6 insertion order for a dict is preserved
        return OrderedDict([
            ('fill', self.fill),
            ('align', self.align),
            ('sign', self.sign),
            ('hash', self.hash),
            ('zero', self.zero),
            ('width', width),
            ('grouping', self.grouping),
            ('precision', precision),
            ('type', self.type)
        ])

    @staticmethod
    def _join(d):
        return ''.join(v for v in d.values() if v is not None)


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
    d = match.groupdict()
    for key in ('width', 'precision', 'df_decimals'):
        if d[key] is not None:
            d[key] = int(d[key])
    return d


def create_format(obj, **kwargs):
    r"""Create a format specification.

    :param obj: An object to use create the format specification.
    :type obj: float, complex, :class:`~GTC.lib.UncertainReal`,
               :class:`~GTC.lib.UncertainComplex`
               or :class:`~GTC.named_tuples.StandardUncertainty`

    :param \**kwargs: Keyword arguments:

            * fill (:class:`str`): Can be a single character (see :ref:`formatspec`).
            * align (:class:`str`): Can be one of < > = ^ (see :ref:`formatspec`).
            * sign (:class:`str`): Can be one of + - `space` (see :ref:`formatspec`).
            * width (:class:`int`): The minimum width (see :ref:`formatspec`).
            * digits (:class:`int`): The number of significant digits in the uncertainty component to retain.
            * type (:class:`str`): Can be one of eEfFgGn% (see :ref:`formatspec`)
                                   TODO should % only make the uncertainty be a percentage of the value
                                        instead of operating on both the value and uncertainty?
                                        Should it (and/or n) be an allowed option?
            * df_decimals (:class:`int`): The number of decimal places reported for the degrees-of-freedom.
                                          TODO the df_decimals was used in the un._round() methods
                                               but dof was never included in the output of __str__.
                                               Do we still want df_decimals?
            * mode (:class:`str`): The mode to use. Must be one of:

                   - B: bracket notation, e.g., 3.142(10)
                   - R: raw plus-minus notation, e.g., 3.142 +/- 0.010

            * style (:class:`str`): The style to use. Must be one of:

                   - L: latex notation, stuff like \infty \times \pm \mathrm
                   - P: pretty print, e.g., (12.3 ± 5.0) × 10⁻¹² or 12.3(2) × 10⁶
                   - S: SI prefix, e.g., (12.3 ± 5.0) p or 12.3(2) M

    :return: The format specification.
    :rtype: :class:`Format`
    """
    for item in ('hash', 'zero', 'grouping'):
        if kwargs.get(item) is not None:
            raise ValueError(
                'The formatting option {!r} is currently not supported'.format(item)
            )

    def maybe_update(key, default):
        if kwargs.get(key) is None:
            kwargs[key] = default

    # default values (only if not specified)
    maybe_update('precision', 2)
    maybe_update('type', 'f')
    maybe_update('df_decimals', 0)
    maybe_update('mode', 'B')

    digits = kwargs.pop('digits', None)
    if digits is None:
        digits = kwargs['precision']
    else:
        kwargs['precision'] = digits

    try:
        u = obj.u
    except AttributeError:
        u = obj

    fmt = Format(**kwargs)
    fmt.digits = digits
    _determine_num_digits(u, fmt)
    return fmt


def to_string(obj, fmt):
    """Convert a numeric object to a string.

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
        result = '{0}{1}'.format(
            fmt.format(r.value, precision=r.precision, type=r.type),
            r.exponent_str
        )
        return _stylize(result, fmt)

    if isinstance(obj, (complex, StandardUncertainty)):
        r = _round(obj.real, fmt)
        re_str = '{0:{1}.{2}{3}}{4}'.format(
            r.value, fmt.sign or '', r.precision, r.type, r.exponent_str)

        i = _round(obj.imag, fmt)
        im_str = '{0:+.{1}{2}}{3}'.format(
            i.value, i.precision, i.type, i.exponent_str)

        join = '({0}{1}j)'.format(re_str, im_str)
        result = fmt.format(join, sign=None, precision=None, type='s')
        return _stylize(result, fmt)

    try:
        # TODO Need to know if `obj` is UncertainReal or UncertainComplex.
        #  We could check isinstance(), but then we will need to deal with
        #  circular import issues with lib.py.
        #  An UncertainReal object has no attribute _value.
        obj._value
        real, imag = obj.real, obj.imag
    except AttributeError:
        real, imag = obj, None

    result = _to_string_ureal(real, fmt)
    if imag is not None:
        imag_str = _to_string_ureal(imag, fmt, sign='+')
        result = '({0}{1}j)'.format(result, imag_str)
    result = fmt.format(result, sign=None, precision=None, type='s')
    return _stylize(result, fmt)


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
    return int(math.floor(math.log10(abs(value))))


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


def _stylize(text, fmt):
    """Apply the formatting style to `text`.

    text: str
    fmt: Format

    returns: the stylized text
    """
    if not fmt.style:
        return text

    if fmt.style == 'P':
        # pretty print
        exp = _exponent_regex.search(text)
        if exp:
            start, end = exp.span()
            translated = exp.group().translate(_exponent_table)
            text = '{0}{1}{2}'.format(text[:start], translated, text[end:])

        mapping = {r'\+/\-': u'\u00B1'}
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

    returns: Rounded
    """
    _type = 'F' if fmt.type in 'EFG' else 'f'

    if _nan_or_inf(value):
        # value precision exponent exponent_str type
        return Rounded(value, 0, 0, '', _type)

    if exponent is None:
        exponent = _order_of_magnitude(value)

    f_or_g_as_f = (fmt.type in 'fF') or \
                  ((fmt.type in 'gG') and
                   (-4 <= exponent < exponent - fmt.u_exponent + 1))

    if f_or_g_as_f:
        factor = 1.0
        precision = max(-fmt.u_exponent, 0)
        digits = -fmt.u_exponent
        exponent_str = ''
    else:
        factor = 10. ** exponent
        precision = max(exponent - fmt.u_exponent, 0)
        digits = precision
        exponent_str = '{0:.0{1}}'.format(factor, fmt.type)

    val = round(value / factor, digits)
    return Rounded(val, precision, exponent, exponent_str[1:], _type)


def _to_string_ureal(ureal, fmt, sign=None):
    """Convert an UncertainReal to a string.

    ureal: UncertainReal
    fmt: Format
    sign: str: <space> + -

    returns: `ureal` as a string
    """
    sign = sign or fmt.sign or ''

    x, u = ureal.x, ureal.u

    if u == 0:
        # TODO Historically, UncertainReal did not include (0) and
        #  UncertainComplex did include (0) with the real and imaginary parts,
        #  e.g.,
        #    ureal(1.23, 0) -> ' 1.23'
        #    ucomplex(1.23+9.87j, 0) -> '(+1.23(0)+9.87(0)j)'
        #  We adopt the UncertainReal version -- do not include the (0)
        return fmt.format(x, sign=sign)

    if _nan_or_inf(x, u):
        x_str = fmt.format(x, sign=sign)
        u_str = '{0:.{1}{2}}'.format(u, fmt.precision, fmt.type)

        if fmt.mode == 'B':
            result = '{0}({1})'.format(x_str, u_str)
        else:
            result = '{0}+/-{1}'.format(x_str, u_str)

        # move the exponential to the end
        exp = _exponent_regex.search(result)
        if exp:
            start, end = exp.span()
            result = '{0}{1}{2}'.format(result[:start], result[end:], exp.group())

        return result

    maximum = round(max(abs(x), u), -fmt.u_exponent)
    result = _round(maximum, fmt)
    exponent, precision = result.exponent, result.precision

    x_result = _round(x, fmt, exponent=exponent)
    u_result = _round(u, fmt, exponent=exponent)
    u_r = u_result.value

    x_str = '{0:{1}.{2}f}'.format(x_result.value, sign, precision)

    if fmt.mode == 'B':
        if _order_of_magnitude(u_r) >= 0 and precision > 0:
            # the uncertainty straddles the decimal point so
            # keep the decimal point in the result
            u_str = '{0:.{1}f}'.format(u_r, precision)
        else:
            u_str = '{:.0f}'.format(round(u_r * 10. ** precision))
        return '{0}({1}){2}'.format(x_str, u_str, result.exponent_str)
    elif fmt.mode == 'R':
        u_str = '{0:.{1}f}'.format(u_r, precision)
        x_u_str = '{0}+/-{1}'.format(x_str, u_str)
        if result.exponent_str:
            return '({0}){1}'.format(x_u_str, result.exponent_str)
        return x_u_str

    raise ValueError(
        'The formatting mode {!r} is not supported. '
        'Must be B or R'.format(fmt.mode)
    )
