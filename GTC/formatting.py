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

    def __repr__(self):
        return 'Format<format_spec={!r} df_decimal={} mode={!r} style={!r}>'.\
            format(self.format_spec, self.df_decimals, self.mode, self.style)

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


def create(obj, **kwargs):
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
            * precision (:class:`int`): The number of significant digits in the uncertainty component to retain.
            * type (:class:`str`): Can be one of eEfFgGn% (see :ref:`formatspec`)
                                   TODO should % only make the uncertainty be a percentage of the value
                                        instead of operating on both the value and uncertainty?
                                        Should it (and/or n) be an allowed option?
            * df_decimals (:class:`int`): The number of decimal places reported for the degrees-of-freedom.
                                          TODO the df_decimals was used in the un._round() methods
                                               but dof was never included of the output of __str__.
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
            raise ValueError('The option {!r} is not supported'.format(item))

    def maybe_update(key, default):
        if kwargs.get(key) is None:
            kwargs[key] = default

    # default values (only if not specified)
    maybe_update('precision', 2)
    maybe_update('type', 'f')
    maybe_update('df_decimals', 0)
    maybe_update('mode', 'B')

    try:
        u = obj.u
    except AttributeError:
        u = obj

    # at this point, u is either a float, complex or StandardUncertainty

    fmt = Format(**kwargs)
    if kwargs['mode'] == 'B':
        _create_bracket(u, fmt)
    else:
        raise ValueError('Mode {!r} is not supported'.format(kwargs['mode']))
    return fmt


def convert(obj, fmt):
    """Convert an object to a string.

    :param obj: An object.
    :type obj: float, complex, :class:`~GTC.lib.UncertainReal`,
               :class:`~GTC.lib.UncertainComplex`
               or :class:`~GTC.named_tuples.StandardUncertainty`
    :param fmt: The format to use to convert `obj`. See :func:`create`.
    :type fmt: :class:`Format`

    :return: The string representation of `obj`.
    :rtype: str
    """
    if isinstance(obj, Number):
        return fmt.format(obj)

    if isinstance(obj, StandardUncertainty):
        return fmt.format(complex(obj.real, obj.imag))

    # TODO Need to know if `obj` is UncertainReal or UncertainComplex.
    #      We could check isinstance(), but then we will need to deal
    #      with circular import issues with lib.py. An UncertainReal
    #      object has no attribute _value.
    x = u = re_x = re_u = im_x = im_u = None
    try:
        obj._value
        real, imag = obj.real, obj.imag
        re_x, re_u = real.x, real.u
        im_x, im_u = imag.x, imag.u
    except AttributeError:
        x, u = obj.x, obj.u

    if fmt.mode == 'B':
        # bracket mode
        if im_x is None:
            # UncertainReal
            x_u_str = _convert_bracket_type_f(x, u, fmt)
            return fmt.format(x_u_str, sign=None, precision=None, type='s')
        else:
            # UncertainComplex
            re_str = _convert_bracket_type_f(re_x, re_u, fmt)
            im_str = _convert_bracket_type_f(im_x, im_u, fmt, sign='+')
            out = '({0}{1}j)'.format(re_str, im_str)
            return fmt.format(out, sign=None, precision=None, type='s')

    raise ValueError('Unsupported mode {!r}'.format(fmt.mode))


def _nan_or_inf(*args):
    # check if any of the args are infinity or NaN
    # args: float, complex
    # returns bool
    for arg in args:
        if isinstance(arg, complex):
            if not cmath.isfinite(arg):
                return True
        else:
            if not math.isfinite(arg):
                return True
    return False


def _create_bracket(uncertainty, fmt):
    # uncertainty: float, StandardUncertainty
    # fmt: Format
    # returns None, `fmt` is modified

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

    log10_u = math.log10(u)
    if log10_u.is_integer():
        log10_u += 1

    # The least power of 10 above the value of `u`
    exponent = math.ceil(log10_u)
    factor = 10. ** (exponent - fmt.precision)

    if fmt.type in 'fF':
        precision = 0 if exponent - fmt.precision >= 0 else int(fmt.precision - exponent)
        if 0 < precision < fmt.precision:
            fmt.u_factor = 1.0
            fmt.u_precision = precision
        else:
            fmt.u_factor = factor
            fmt.u_precision = 0
    else:
        raise ValueError('Type {!r} is not supported'.format(fmt.type))

    fmt.factor = factor
    fmt.precision = precision


def _convert_bracket_type_f(value, uncertainty, fmt, **kwargs):
    if uncertainty == 0:
        # TODO Historically, ureal did not include (0) and ucomplex
        #  did include (0) on the real and imaginary parts, e.g.,
        #    ureal -> ' 1.234568'
        #    ucomplex -> '(+1.234568(0)+9.876543(0)j)'
        #  Now, we adopt the ureal version (do not include the (0))
        return fmt.format(value, **kwargs)

    if _nan_or_inf(value, uncertainty):
        return '{0}({1:{2}})'.format(
            fmt.format(value, **kwargs),
            uncertainty,
            fmt.type
        )

    v = fmt.factor * round(value / fmt.factor)
    u = fmt.u_factor * round(uncertainty / fmt.u_factor, fmt.u_precision)
    if fmt.precision > 1:
        u /= fmt.u_factor

    value_str = fmt.format(v, fill=None, align=None, width=None, **kwargs)
    result = '{0}({1:.{2}f})'.format(value_str, u, fmt.u_precision)

    found = _exponent_regex.search(result)
    if found:
        start, end = found.span()
        result = ''.join([result[:start], result[end:], found.group()])

    return result
