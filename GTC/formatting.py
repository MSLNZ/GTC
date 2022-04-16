# -*- coding: utf-8 -*-
import re
from collections import OrderedDict

from GTC.named_tuples import GroomedUncertainReal

# The regular expression to parse the format specification (format_spec)
# with an additional (and optional) characters at the end for GTC-specific formats
#
# format_spec ::= [[fill]align][sign][#][0][width][grouping][.precision][type][.GTC_df_decimals][GTC_mode]
# https://docs.python.org/3/library/string.html#format-specification-mini-language
_regex = re.compile(
    # the builtin grammar fields
    r'((?P<fill>.)(?=[<>=^]))?'
    r'(?P<align>[<>=^])?'
    r'(?P<sign>[ +-])?'
    r'(?P<hash>#)?'
    r'(?P<zero>0)?'
    r'(?P<width>\d+)?'
    r'(?P<grouping>[_,])?'
    r'((\.)(?P<precision>\d+))?'
    r'(?P<type>[bcdeEfFgGnosxX%])?'  # some are only valid for integers, but want to match all possibilities
    
    # number of degrees-of-freedom decimal places
    # NOTE: <type> and/or <precision> must also be specified for this to match
    r'((\.)(?P<df_decimals>\d+))?' 
    
    # display mode: Bracket, Latex, Pretty \u00b1 and superscripts, Raw +/-, SI prefix
    # NOTE: these characters cannot be in <type>
    r'(?P<mode>[BLPRS])?'
    
    # the regex must match until the end of the string
    r'$'
)


class Format(object):

    def __init__(self, **kwargs):
        """Format specification for displaying uncertain numbers.

        Do not instantiate this class directly. Use :func:`get_format`.
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

        # GTC-specific fields
        self.mode = kwargs.get('mode') or 'B'
        self.df_decimals = kwargs.get('df_decimals') or 0

        # TODO should these (or some variant) be defined?
        #self.u_digits = kwargs.get('u_digits')
        #self.re_u_digits = kwargs.get('re_u_digits')
        #self.im_u_digits = kwargs.get('im_u_digits')

    def format(self, number, **kwargs):
        """Format a number.

        TODO Including kwargs in this method signature is probably not what
             is wanted. Ideally the Format already knows what to do.
        """
        d = self.format_spec_dict
        if kwargs:
            for k, v in kwargs.items():
                # purposely use {:d} since width and precision must be integers
                if k == 'precision':
                    v = '.{:d}'.format(v)
                elif k == 'width':
                    v = '{:d}'.format(v)
                d[k] = v
        format_spec = ''.join(v for v in d.values())
        return '{0:{1}}'.format(number, format_spec)

    @property
    def format_spec(self):
        """Return a `format_spec` string that is compatible with
        the :func:`format` function."""
        return ''.join(v for v in self.format_spec_dict.values())

    @property
    def format_spec_dict(self):
        """Return a `format_spec` dict that is compatible with
        the :func:`format` function."""

        # purposely use {:d} since width and precision must be integers
        width = '' if self.width is None else '{:d}'.format(self.width)
        precision = '' if self.precision is None else '.{:d}'.format(self.precision)

        # TODO can use a regular dict when dropping support for Python < 3.6
        #  since as of Python 3.6 insertion order for a dict is preserved
        return OrderedDict([
            ('fill', self.fill or ''),
            ('align', self.align or ''),
            ('sign', self.sign or ''),
            ('hash', self.hash or ''),
            ('zero', self.zero or ''),
            ('width', width),
            ('grouping', self.grouping or ''),
            ('precision', precision),
            ('type', self.type or '')
        ])


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
    d = match.groupdict()
    for key in ('width', 'precision', 'df_decimals'):
        if d[key] is not None:
            d[key] = int(d[key])
    return d


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
            * grouping (:class:`str`): Must be one of: _, (see :ref:`formatspec`)
                                       TODO should grouping be a kwarg?
            * precision (:class:`int`): The number of significant digits in the least uncertainty
                                        component that will be retained. The components of the value
                                        will use the same precision.
            * type (:class:`str`): The presentation type. One of eEfFgGn% (see :ref:`formatspec`)
                                   TODO should % only make the uncertainty be a percentage of the value?
                                    should it even be an allowed kwarg?
            * df_decimals (:class:`int`): The number of decimal places reported for the degrees-of-freedom.
            * mode (:class:`str`): The mode that the uncertainty is displayed in.
                Must be one of::

                   - B: bracket notation, e.g., 3.142(10)
                   - L: latex notation, stuff like \mathrm \infty \times
                   - P: pretty-print, e.g., (12.3 ± 5.0) × 10⁻¹²
                   - R: raw, e.g., 3.142+/-0.010
                   - S: SI prefix, e.g., (12.3 ± 5.0)p or -6.742(31)M

                   TODO We may want L or S to also depend on the value of
                    B P or R. If so, we could separate BPR from LS in the
                    regex and consider BPR as 'styles' and LS as 'modes'.

    :return: The format to use to display uncertain numbers.
    :rtype: :class:`Format`
    """
    # TODO The hash and zero kwargs may be discarded.
    #  Currently, `un` is not used here. It is currently only used in display()
    if kwargs.get('hash'):
        kwargs['hash'] = '#'
    if kwargs.get('zero'):
        kwargs['zero'] = '0'
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
    if fmt.mode == 'B':  # bracket notation
        # TODO Need to handle all of the format_spec fields.
        #  For now, assume that only str(un) is being used.
        groomed = un._round(fmt.precision, fmt.df_decimals)
        precision = groomed.precision
        if isinstance(groomed, GroomedUncertainReal):
            # "{1.x: .{0}f}{1.u_digits}".format(gself.precision, gself)
            return '{0}{1.u_digits}'.format(
                fmt.format(groomed.x, precision=precision),
                groomed)
        else:
            # "({1.real:+.{0}f}({2}){1.imag:+.{0}f}({3})j)".format(
            #  gself.precision, gself.x, gself.re_u_digits, gself.im_u_digits)
            return '({0}({2.re_u_digits}){1}({2.im_u_digits})j)'.format(
                fmt.format(groomed.x.real, precision=precision),
                fmt.format(groomed.x.imag, precision=precision),
                groomed)

    raise ValueError('Unsupported mode {!r}'.format(fmt.mode))
