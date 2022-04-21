# -*- coding: utf-8 -*-
import re
import math
import cmath
from numbers import Number
from collections import OrderedDict

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

        # these attributes are used by the different modes and/or styles
        self.factor = 1.0
        self.u_factor = 1.0
        self.u_precision = 0
        self.u_exponent = 0
        self.digits = 0

    def __repr__(self):
        df_decimals = '' if self.df_decimals is None else '.{:d}'.format(self.df_decimals)
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
    """Convert an object to a string.

    :param obj: An object.
    :type obj: float, complex, :class:`~GTC.lib.UncertainReal`,
               :class:`~GTC.lib.UncertainComplex`
               or :class:`~GTC.named_tuples.StandardUncertainty`
    :param fmt: The format to use to convert `obj`. See :func:`create_format`.
    :type fmt: :class:`Format`

    :return: The string representation of `obj`.
    :rtype: str
    """
    if isinstance(obj, Number):
        result = fmt.format(obj)
    elif isinstance(obj, StandardUncertainty):
        result = fmt.format(complex(obj.real, obj.imag))
    else:
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


def _determine_num_digits(uncertainty, fmt):
    """Determine the number digits that is required to display `uncertainty`.

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

    if u == 0 or _nan_or_inf(u):
        # set these values for backwards compatibility
        fmt.factor = 1.0
        fmt.precision = 6
        fmt.u_factor = 1.0
        fmt.u_precision = 0
        return

    exponent, factor = _exponent_factor(u, fmt)

    precision = 0 if exponent - fmt.precision >= 0 else int(fmt.precision - exponent)
    if 0 < precision < fmt.precision:
        fmt.u_factor = 1.0
        fmt.u_precision = precision
    else:
        fmt.u_factor = factor
        fmt.u_precision = 0

    fmt.factor = factor
    fmt.precision = precision
    fmt.u_exponent = exponent


def _exponent_factor(value, fmt):
    """Get the least power of 10 above `value`.

    value: float
    fmt: Format

    returns: (exponent: int, factor: float)
    """
    if value == 0:
        return 0, 1.0

    log10 = math.log10(abs(value))
    if log10.is_integer():
        log10 += 1
    exponent = math.ceil(log10)
    factor = 10. ** (exponent - fmt.precision)
    return exponent, factor


def _round(value, uncertainty, fmt):
    """Round a value and an uncertainty to the appropriate number of digits.

    value: float
    uncertainty: float
    fmt: Format

    returns: (value_rounded, uncertainty_rounded)
    """
    v = fmt.factor * round(value / fmt.factor)
    u = fmt.u_factor * round(uncertainty / fmt.u_factor, fmt.u_precision)
    if fmt.precision > 1:
        u /= fmt.u_factor
    return v, u


def _parse_for_exponent(text):
    """Check if `text` has an exponent term.

    If it does, then return (before: str, match: str, after: str)
    where `before` and `after` correspond to the substring before
    and after `match` and `match` is the exponent term that was
    found in `text`. Otherwise, returns None.

    Examples
    --------
    '1.2e+02' -> ('1.2', 'e+02', '')
    '1.2345' -> None
    """
    found = _exponent_regex.search(text)
    if found:
        start, end = found.span()
        return text[:start], found.group(), text[end:]


def _stylize(text, fmt):
    """Apply the formatting style to `text`.

    text: str
    fmt: Format

    returns: the stylized text
    """
    if not fmt.style:
        return text

    raise ValueError(
        'The formatting style {!r} is not supported'.format(fmt.style)
    )


def _to_string_ureal(ureal, fmt, **kwargs):
    """Convert an UncertainReal to a string.

    ureal: UncertainReal
    fmt: Format
    kwargs: passed to Format.format()

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
        return fmt.format(x, **kwargs)

    if _nan_or_inf(x, u):
        x_str = fmt.format(x, **kwargs)
        u_str = '{0:{1}}'.format(u, fmt.type)
        if fmt.mode == 'B':
            return '{0}({1})'.format(x_str, u_str)
        return '{0} +/- {1}'.format(x_str, u_str)

    x_r, u_r = _round(x, u, fmt)

    if fmt.mode == 'B':
        if fmt.type in 'fF':
            x_str = fmt.format(x_r, fill=None, align=None, width=None, **kwargs)
            u_str = '{0:.{1}f}'.format(u_r, fmt.u_precision)
            return '{0}({1})'.format(x_str, u_str)

    raise ValueError(
        'The formatting mode {!r} is not supported. '
        'Must be B or R'.format(fmt.mode)
    )
