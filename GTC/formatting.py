# -*- coding: utf-8 -*-
import re
from collections import OrderedDict

from GTC.named_tuples import GroomedUncertainReal

# The regular expression to parse the format specification (format_spec)
# with an additional (and optional) characters at the end for GTC-specific formats
#
# format_spec ::= [[fill]align][sign][#][0][width][grouping_option][.precision][type][.GTC_df_decimals][GTC_mode]
# https://docs.python.org/3/library/string.html#format-specification-mini-language
_regex = re.compile(
    # the builtin grammar fields
    r'((?P<fill>.)(?=[<>=^]))?'
    r'(?P<align>[<>=^])?'
    r'(?P<sign>[ +-])?'
    r'(?P<hash>#)?'
    r'(?P<zero>0)?'
    r'(?P<width>\d+)?'
    r'(?P<grouping_option>[_,])?'
    r'((\.)(?P<precision>\d+))?'
    r'(?P<type>[bcdeEfFgGnosxX%])?'  # some are only valid for integers, but want to match all possibilities
    
    # number of degrees-of-freedom decimal places
    # NOTE: <type> and/or <precision> must also be specified for this to match
    r'((\.)(?P<df_decimals>\d+))?' 
    
    # display mode: B=bracket, L=latex, R=+/-, S=use SI prefix, U=\u00b1 and superscripts
    # NOTE: these characters cannot be in <type>
    r'(?P<mode>[BLRSU])?'
    
    # the regex must match until the end of the string
    r'$'
)


class Format(object):

    def __init__(self, **kwargs):
        """Format specification for displaying uncertain numbers.

        Do not instantiate this class directly. Use :func:`get_format`.
        """
        def get(key, default):
            v = kwargs.get(key)
            return default if v is None else type(default)(v)

        # builtin fields
        self.fill = kwargs.get('fill') or ''
        self.align = kwargs.get('align') or ''
        self.sign = kwargs.get('sign') or  ''
        self.hash = kwargs.get('hash') or ''
        self.zero = kwargs.get('zero') or ''
        self.width = kwargs.get('width') or ''
        self.grouping_option = kwargs.get('grouping_option') or ''
        self.precision = get('precision', 2)
        self.type = kwargs.get('type') or 'f'

        # GTC-specific fields
        self.mode = kwargs.get('mode') or 'B'
        self.df_decimals = get('df_decimals', 0)

        # TODO should these be defined?
        #self.u_digits = get('u_digits', self.precision)
        #self.re_u_digits = get('re_u_digits', self.u_digits)
        #self.im_u_digits = get('im_u_digits', self.re_u_digits)

        # TODO use a regular dict when dropping support for Python < 3.6
        #  since as of Python 3.6 insertion order for a dict is preserved
        self._dict = OrderedDict([
            ('fill', self.fill),
            ('align', self.align),
            ('sign', self.sign),
            ('hash', self.hash),
            ('zero', self.zero),
            ('width', '{}'.format(self.width)),
            ('grouping_option', self.grouping_option),
            ('precision', '.{}'.format(self.precision)),
            ('type', self.type)
        ])

    def format(self, number, **kwargs):
        """Format a number."""
        if kwargs:
            d = self._dict.copy()
            for k, v in kwargs.items():
                if k == 'precision':
                    v = '.{}'.format(v)
                elif k == 'width':
                    v = '{}'.format(v)
                d[k] = v
            format_spec = ''.join(v for v in d.values())
        else:
            format_spec = self.format_spec
        return '{0:{1}}'.format(number, format_spec)

    @property
    def format_spec(self):
        """Return a `format_spec` string that is compatible with
        the :func:`format` function."""
        return ''.join(v for v in self._dict.values())

    @property
    def format_spec_dict(self):
        """Return a `format_spec` dict that is compatible with
        the :func:`format` function."""
        return self._dict.copy()


def parse(format_spec):
    """Parse a format specification into its grammar fields.

    :param format_spec: A format specification. Supports the builtin format
                        fields (see :ref:`formatspec`) plus additional
                        characters, that must be at the end of `format_spec`,
                        which are used to decide how an uncertain number will
                        be displayed.
    :type format_spec: str

    :return: The grammar fields.
    :rtype: dict
    """
    match = _regex.match(format_spec)
    if not match:
        raise ValueError('Invalid format specifier {!r}'.format(format_spec))
    return match.groupdict()


def create_format(un, **kwargs):
    r"""Create a format for displaying uncertain numbers.

    :param un: An uncertain number
    :type un: :class:`~GTC.lib.UncertainReal` or :class:`~GTC.lib.UncertainComplex`

    :param \**kwargs: Keyword arguments:

            * fill (:class:`str`): Must be a single character (see :ref:`formatspec`)
            * align (:class:`str`): Must be one of <>=^ (see :ref:`formatspec`)
            * sign (:class:`str`): Must be one of +-`space`  (see :ref:`formatspec`)
            * hash (:class:`bool`): Whether to include the ``#`` character (see :ref:`formatspec`)
                                    TODO should hash be a kwarg?
            * zero (:class:`bool`): Whether to include the ``0`` character (see :ref:`formatspec`)
                                    TODO should zero be a kwarg?
            * width (:class:`int`): The minimum width of the value and uncertainty components (see :ref:`formatspec`)
                                    TODO should width be a kwarg?
            * grouping_option (:class:`str`): Must be one of: _, (see :ref:`formatspec`)
                                              TODO should grouping_option be a kwarg?
            * precision (:class:`int`): The number of significant digits in the least uncertainty
                                        component that will be retained. The components of the value
                                        will use the same precision.
            * type (:class:`str`): The presentation type. One of eEfFgGn% (see :ref:`formatspec`)
                                   TODO should % only make the uncertainty be a percentage of the value?
                                    should it even be an allowed kwarg?
            * df_decimals (:class:`int`): The number of decimal places reported for the degrees-of-freedom.
            * mode (:class:`str`): The mode that the uncertainty is displayed in.
                Must be one of::

                   - B: use bracket notation, e.g., 3.142(10)
                   - L: use latex notation, stuff like \mathrm \infty \times
                   - R: use a raw plus-minus sign, e.g., 3.142+/-0.010
                   - U: use unicode plus-minus \u00b1 and superscripts, e.g., (12.3 ± 5.0) × 10⁻¹²

    :return: The format to use to display uncertain numbers.
    :rtype: :class:`Format`
    """
    # TODO The hash, zero and width kwargs may be discarded.
    #  Currently, `un` is not used because display() calls un._round
    #  and str(un) is the only display that is currently handled.
    h = kwargs.get('hash')
    if h:
        kwargs['hash'] = '#'
    z = kwargs.get('zero')
    if z:
        kwargs['zero'] = '0'
    w = kwargs.get('width')
    if w:
        kwargs['width'] = str(w)
    return Format(**kwargs)


def display(un, fmt):
    """Display an uncertain number as a string.

    :param un: An uncertain number.
    :type un: :class:`~GTC.lib.UncertainReal` or :class:`~GTC.lib.UncertainComplex`
    :param fmt: The format to use to display `un`. See :func:`create_format`.
    :type fmt: :class:`Format`

    :return: A string representation of `un`.
    :rtype: str
    """
    if fmt.type in 'bcdoxX':
        raise ValueError('The format type {!r} is only '
                         'valid for integers'.format(fmt.type))

    if fmt.mode == 'B':  # bracket notation
        # TODO Need to handle all of the format_spec fields.
        #  For now only assume that str(un) is being used.
        groomed = un._round(fmt.precision, fmt.df_decimals)
        spec = fmt.format_spec_dict
        spec['precision'] = groomed.precision
        if isinstance(groomed, GroomedUncertainReal):
            # "{1.x: .{0}f}{1.u_digits}".format(gself.precision, gself)
            spec['sign'] = ' '
            return '{0}{1.u_digits}'.format(
                fmt.format(groomed.x, **spec),
                groomed)
        else:
            # "({1.real:+.{0}f}({2}){1.imag:+.{0}f}({3})j)".format(
            #  gself.precision, gself.x, gself.re_u_digits, gself.im_u_digits)
            spec['sign'] = '+'
            return '({0}({2.re_u_digits}){1}({2.im_u_digits})j)'.format(
                fmt.format(groomed.x.real, **spec),
                fmt.format(groomed.x.imag, **spec),
                groomed)

    raise ValueError('Unsupported mode {!r}'.format(fmt.mode))
