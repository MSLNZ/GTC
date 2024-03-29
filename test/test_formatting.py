# -*- coding: utf-8 -*-
import locale
import math
import sys
import unittest

from GTC import ureal, ucomplex, inf_dof, inf, nan
from GTC.formatting import (
    Format,
    create_format,
    to_string,
    parse,
    apply_format,
    _nan_or_inf,
    _order_of_magnitude,
    _truncate_dof,
)
from GTC.lib import UncertainReal, UncertainComplex

original_locale = locale.getlocale(locale.LC_NUMERIC)


def tearDownModule():
    locale.setlocale(locale.LC_NUMERIC, original_locale)


class TestFormatting(unittest.TestCase):

    def test_parse_raises(self):
        # want the exception types raised by formatting.parse to
        # match what the builtin format(float, format_spec) would raise
        def check(exception, format_spec):
            self.assertRaises(exception, format, 1.0, format_spec)
            self.assertRaises(exception, parse, format_spec)

        # format_spec must be a str
        check(TypeError, 1.2)
        check(TypeError, 1.2+2.3j)
        check(TypeError, ureal(1, 0.1))
        check(TypeError, ureal(1, 0.1).x)
        check(TypeError, ureal(1, 0.1).u)
        check(TypeError, ureal(1, 0.1).real)
        check(TypeError, ureal(1, 0.1).imag)
        check(TypeError, ucomplex(1+1j, 0.1))
        check(TypeError, ucomplex(1+1j, 0.1).x)
        check(TypeError, ucomplex(1+1j, 0.1).u)
        check(TypeError, ucomplex(1+1j, 0.1).real)
        check(TypeError, ucomplex(1+1j, 0.1).imag)

        # invalid format strings
        check(ValueError, 'A')  # invalid <type>, <style> or <fill> without <align>
        check(ValueError, '-5.2A')  # invalid <type>
        check(ValueError, '.')  # <decimal> without <precision>
        check(ValueError, '2.f')  # <decimal> without <precision>
        check(ValueError, '===')  # multiple <fill> characters
        check(ValueError, '**<.4G')  # multiple <fill> characters
        check(ValueError, '<<<.4G')  # multiple <fill> characters
        check(ValueError, '#+.2f')  # <hash> before <sign>
        check(ValueError, '0#.2f')  # <digit> before <hash>
        check(ValueError, ',3.2f')  # <grouping> before <width>
        check(ValueError, '0-.4G')  # <sign> after <zero>
        check(ValueError, '#-.4G')  # <sign> after <hash>
        check(ValueError, '=7^2,.3f')  # <width> before <align>
        check(ValueError, '=^20,3f')  # <width> after <grouping> or forgot the <decimal> before <precision>
        check(ValueError, '!5.2f')  # invalid <sign> character
        check(ValueError, '5!.2f')  # invalid <grouping> character
        check(ValueError, '!.2f')  # <fill> without <align> or invalid <sign> character
        check(ValueError, '5.2fA')  # invalid <option> character and too many builtin fields
        check(ValueError, 'LU')  # two styles specified

    def test_parse(self):
        # also call the builtin format(float, format_spec) to verify
        # that the formatting.parse function is okay
        def _parse(format_spec, check=True):
            if check:  # must ignore for GTC-specific fields
                format(1.0, format_spec)
            return parse(format_spec)

        def expect(**kwargs):
            out = {'fill': None, 'align': None, 'sign': None, 'hash': None,
                   'zero': None, 'width': None, 'grouping': None,
                   'precision': None, 'type': None, 'style': None}
            out.update(**kwargs)
            return out

        # check the builtin-supported fields
        self.assertEqual(_parse('G'),
                         expect(type='G'))
        self.assertEqual(_parse('='),
                         expect(align='='))
        self.assertEqual(_parse(' ='),
                         expect(fill=' ', align='='))
        self.assertEqual(_parse('<<'),
                         expect(fill='<', align='<'))
        self.assertEqual(_parse(' 10.1'),
                         expect(sign=' ', width='10', precision='1'))
        self.assertEqual(_parse('0'),
                         expect(zero='0'))
        self.assertEqual(_parse('0.0'),
                         expect(zero='0', precision='0'))
        self.assertEqual(_parse('02'),
                         expect(zero='0', width='2'))
        self.assertEqual(_parse('02.0'),
                         expect(zero='0', width='2', precision='0'))
        self.assertEqual(_parse('.10'),
                         expect(precision='10'))
        self.assertEqual(_parse('07.2f'),
                         expect(zero='0', width='7', precision='2', type='f'))
        self.assertEqual(_parse('*<-06,.4E'),
                         expect(fill='*', align='<', sign='-', zero='0',
                                width='6', grouping=',', precision='4', type='E'))

        # additional GTC-specific fields
        self.assertEqual(_parse('U', False),
                         expect(style='U'))
        self.assertEqual(_parse('GL', False),
                         expect(type='G', style='L'))
        self.assertEqual(_parse('.2U', False),
                         expect(precision='2', style='U'))
        self.assertEqual(_parse('9U', False),
                         expect(width='9', style='U'))
        self.assertEqual(_parse('.7', False),
                         expect(precision='7'))
        self.assertEqual(_parse('e', False),
                         expect(type='e'))
        self.assertEqual(_parse('.2f', False),
                         expect(precision='2', type='f'))
        self.assertEqual(_parse(' ^16.4fL', False),
                         expect(fill=' ', align='^', width='16',
                                precision='4', type='f', style='L'))
        self.assertEqual(_parse('^^03', False),
                         expect(fill='^', align='^', zero='0', width='3'))
        self.assertEqual(_parse('^^03L', False),
                         expect(fill='^', align='^', zero='0', width='3', style='L'))
        self.assertEqual(_parse('*> #011,.2gL', False),
                         expect(fill='*', align='>', sign=' ', hash='#',
                                zero='0', width='11', grouping=',', precision='2',
                                type='g', style='L'))

    def test_Format(self):
        f = Format()
        self.assertEqual(repr(f), "Format(format_spec='.2f', df_precision=0, r_precision=3)")
        self.assertEqual(str(f), "Format(format_spec='.2f', df_precision=0, r_precision=3)")
        self.assertEqual(f._digits, 2)
        self.assertEqual(f._u_exponent, 0)
        self.assertEqual(f._df_precision, 0)
        self.assertEqual(f._r_precision, 3)

        f = Format(fill='*', align='>', sign=' ', hash='#', zero='0',
                   width=20, grouping=',', digits=3, type='g',
                   style='L', df_precision=1)
        self.assertEqual(repr(f), "Format(format_spec='*> #020,.3gL', df_precision=1, r_precision=3)")
        self.assertEqual(f._digits, 3)
        self.assertEqual(f._u_exponent, 0)
        self.assertEqual(f._df_precision, 1)
        self.assertEqual(f._r_precision, 3)

        f = Format(width=10, sign='+', r_precision=1, df_precision=None)
        self.assertEqual(repr(f), "Format(format_spec='+10.2f', df_precision=0, r_precision=1)")
        self.assertEqual(f._digits, 2)
        self.assertEqual(f._u_exponent, 0)
        self.assertEqual(f._df_precision, 0)
        self.assertEqual(f._r_precision, 1)

        f = Format(precision=0, digits=1, r_precision=None, df_precision=7)
        self.assertEqual(f._precision, 0)
        self.assertEqual(f._digits, 1)
        self.assertEqual(f._u_exponent, 0)
        self.assertEqual(f._df_precision, 7)
        self.assertEqual(f._r_precision, 3)

        f = Format(precision=0, digits=1, r_precision=6, df_precision=4)
        self.assertEqual(f._precision, 0)
        self.assertEqual(f._digits, 1)
        self.assertEqual(f._u_exponent, 0)
        self.assertEqual(f._df_precision, 4)
        self.assertEqual(f._r_precision, 6)

        f = Format()
        number = -9.3+123.456789j
        self.assertEqual(f._value(number), '{:.2f}'.format(number))
        number = 123.456789
        self.assertEqual(f._value(number, precision=4, type='f', sign=' '),
                         '{: .4f}'.format(number))

        f = Format(precision=4, sign='+')
        number = 123.456789
        self.assertEqual(f._value(number), '{:+.4f}'.format(number))
        number = -9.3+123.456789j
        self.assertEqual(f._value(number), '{:+.4f}'.format(number))

        f = Format(precision=4, sign='+', width=20, fill='*', align='>')
        number = 123.456789
        self.assertEqual(f._result(f._value(number)),
                         '{:*>+20.4f}'.format(number))

        f = Format(precision=4, sign='+', type='e')
        number = 123.456789
        self.assertEqual(f._value(number), '{:+.4e}'.format(number))
        number = -9.3+123.456789j
        self.assertEqual(f._value(number, type='E'), '{:+.4E}'.format(number))

        f = Format(grouping=',', precision=0, type='f')
        number = 123456789
        self.assertEqual(f._value(number), '{:,.0f}'.format(number))

        f = create_format(1.23)
        self.assertEqual(f._precision, 3)
        self.assertEqual(f._digits, 2)
        self.assertEqual(f._u_exponent, -1)

        f = create_format(1.234, precision=4)
        self.assertEqual(f._precision, 5)
        self.assertEqual(f._digits, 4)
        self.assertEqual(f._u_exponent, -3)

        f = create_format(0, digits=20)
        self.assertEqual(f._precision, 20)
        self.assertEqual(f._digits, 20)
        self.assertEqual(f._u_exponent, 0)

    def test_create_format_raises(self):
        for d in [0, -1, 0.99999, '0']:
            self.assertRaises(ValueError, create_format, 1, digits=d)
            self.assertRaises(ValueError, create_format, 1, precision=d)

        for s in 'luKJA*{&':
            self.assertRaises(ValueError, create_format, 1, style=s)

        # expect 'digits', not 'digit' (checks for a spelling mistake)
        self.assertRaises(ValueError, create_format, 1, digit=3)

    def test_nan_or_inf(self):
        nan_or_inf = _nan_or_inf
        self.assertTrue(not nan_or_inf())
        self.assertTrue(not nan_or_inf(0, 1.0, 1.0j, 1.0-1.0j, complex(1, 1), 1e300))
        self.assertTrue(nan_or_inf(0, 1.0, 1.0j, 1.0-1.0j, complex(1, 1), 1e300, nan))
        self.assertTrue(nan_or_inf(0, 1.0, inf, 1.0-1.0j, complex(1, 1), 1e300))
        self.assertTrue(nan_or_inf(1.0, complex(1, nan), 1.0j))
        self.assertTrue(nan_or_inf(complex(inf, 1.0)))
        self.assertTrue(nan_or_inf(complex(-inf, 1.0)))
        self.assertTrue(nan_or_inf(complex(nan, 1.0)))
        self.assertTrue(nan_or_inf(complex(1.0, inf)))
        self.assertTrue(nan_or_inf(complex(1.0, -inf)))
        self.assertTrue(nan_or_inf(complex(1.0, nan)))
        self.assertTrue(nan_or_inf(complex(inf, inf)))
        self.assertTrue(nan_or_inf(complex(inf, -inf)))
        self.assertTrue(nan_or_inf(complex(inf, nan)))
        self.assertTrue(nan_or_inf(complex(nan, inf)))
        self.assertTrue(nan_or_inf(complex(nan, -inf)))
        self.assertTrue(nan_or_inf(complex(nan, nan)))

    def test_order_of_magnitude(self):
        self.assertEqual(_order_of_magnitude(0), 0)
        self.assertEqual(_order_of_magnitude(0.000000000000123456789), -13)
        self.assertEqual(_order_of_magnitude(0.00000000000123456789), -12)
        self.assertEqual(_order_of_magnitude(0.0000000000123456789), -11)
        self.assertEqual(_order_of_magnitude(0.000000000123456789), -10)
        self.assertEqual(_order_of_magnitude(0.00000000123456789), -9)
        self.assertEqual(_order_of_magnitude(0.0000000123456789), -8)
        self.assertEqual(_order_of_magnitude(0.000000123456789), -7)
        self.assertEqual(_order_of_magnitude(0.00000123456789), -6)
        self.assertEqual(_order_of_magnitude(0.0000123456789), -5)
        self.assertEqual(_order_of_magnitude(0.000123456789), -4)
        self.assertEqual(_order_of_magnitude(0.00123456789), -3)
        self.assertEqual(_order_of_magnitude(0.0123456789), -2)
        self.assertEqual(_order_of_magnitude(0.123456789), -1)
        self.assertEqual(_order_of_magnitude(1.23456789), 0)
        self.assertEqual(_order_of_magnitude(12.3456789), 1)
        self.assertEqual(_order_of_magnitude(123.456789), 2)
        self.assertEqual(_order_of_magnitude(1234.56789), 3)
        self.assertEqual(_order_of_magnitude(12345.6789), 4)
        self.assertEqual(_order_of_magnitude(123456.789), 5)
        self.assertEqual(_order_of_magnitude(1234567.89), 6)
        self.assertEqual(_order_of_magnitude(12345678.9), 7)
        self.assertEqual(_order_of_magnitude(123456789.), 8)
        self.assertEqual(_order_of_magnitude(1234567890.), 9)
        self.assertEqual(_order_of_magnitude(12345678900.), 10)
        self.assertEqual(_order_of_magnitude(123456789000.), 11)
        self.assertEqual(_order_of_magnitude(1234567890000.), 12)
        self.assertEqual(_order_of_magnitude(12345678900000.), 13)

    def test_repr_ureal(self):
        def check(ur, expected):
            # different ways to get the same result
            self.assertEqual(repr(ur), expected)
            self.assertEqual('{!r}'.format(ur), expected)

        check(ureal(1.23456789, 0.001),
              'ureal(1.23456789,0.001,inf)')
        check(ureal(-1, 1.23456789e-7, df=7),
              'ureal(-1.0,1.23456789e-07,7.0)')
        check(ureal(3, 0.01, df=inf_dof+1),
              'ureal(3.0,0.01,inf)')
        check(ureal(1.23456789e10, 10),
              'ureal(12345678900.0,10.0,inf)')
        check(ureal(1.23456789e18, 10, label='numbers'),
              "ureal(1.23456789e+18,10.0,inf, label='numbers')")
        check(ureal(1.23456789e-9, 2.1e-11),
              'ureal(1.23456789e-09,2.1e-11,inf)')
        check(ureal(3.141592653589793, 0.01, df=3, label='PI'),
              "ureal(3.141592653589793,0.01,3.0, label='PI')")

    def test_repr_ucomplex(self):
        def check(uc, expected):
            # different ways to get the same result
            self.assertEqual(repr(uc), expected)
            self.assertEqual('{!r}'.format(uc), expected)

        check(ucomplex(1.23456789+0.12345j, 0.001),
              'ucomplex((1.23456789+0.12345j), u=[0.001,0.001], r=0.0, df=inf)')
        check(ucomplex(1.23456789+0.12345j, [0.001, 0.002]),
              'ucomplex((1.23456789+0.12345j), u=[0.001,0.002], r=0.0, df=inf)')
        check(ucomplex(1.23456789 + 0.12345j, [1, 1, 1, 1]),
              'ucomplex((1.23456789+0.12345j), u=[1.0,1.0], r=1.0, df=inf)')
        check(ucomplex(1.23456789 + 0.12345j, 0.1, df=8),
              'ucomplex((1.23456789+0.12345j), u=[0.1,0.1], r=0.0, df=8.0)')
        check(ucomplex(1.23456789e13 + 0.12345e10j, 1.3e9, df=inf_dof*2),
              'ucomplex((12345678900000+1234500000j), u=[1300000000.0,1300000000.0], r=0.0, df=inf)')
        check(ucomplex(1.23456789 + 0.12345j, 0.1, df=8, label='MSL'),
              "ucomplex((1.23456789+0.12345j), u=[0.1,0.1], r=0.0, df=8.0, label='MSL')")

    def test_str_ureal(self):
        def check(ur, expected):
            # different ways to get the same result
            self.assertEqual(str(ur), expected)
            self.assertEqual('{}'.format(ur), expected)
            self.assertEqual('{!s}'.format(ur), expected)
            self.assertEqual('{: }'.format(ur), expected)
            self.assertEqual('{: f}'.format(ur), expected)
            self.assertEqual('{: .2}'.format(ur), expected)
            self.assertEqual('{: .2f}'.format(ur), expected)

        check(ureal(1.23456789, 1000), ' 0(1000)')
        check(ureal(1.23456789, 100), ' 0(100)')
        check(ureal(1.23456789, 10), ' 1(10)')
        check(ureal(1.23456789, 1), ' 1.2(1.0)')
        check(ureal(1.23456789, 0), ' 1.23')
        check(ureal(1.23456789, 0.1), ' 1.23(10)')
        check(ureal(1.23456789, 0.01), ' 1.235(10)')
        check(ureal(1.23456789, 0.001), ' 1.2346(10)')
        check(ureal(1.23456789, 0.0001), ' 1.23457(10)')
        check(ureal(1.23456789, 0.00001), ' 1.234568(10)')
        check(ureal(1.23456789, 0.000001), ' 1.2345679(10)')
        check(ureal(1.23456789, 0.0000001), ' 1.23456789(10)')
        check(ureal(1.23456789, 0.00000001), ' 1.234567890(10)')
        check(ureal(1.23456789, 0.000000001), ' 1.2345678900(10)')
        check(ureal(-1.23456789, 0.0001234567), '-1.23457(12)')
        check(ureal(1.23456789e6, 3.421e4), ' 1235000(34000)')
        check(ureal(1.23456789e17, 3.421e11), ' 123456790000000000(340000000000)')
        check(ureal(1.23456789e-9, 2.1e-11), ' 0.000000001235(21)')

    def test_str_ucomplex(self):
        def check(uc, expected):
            # different ways to get the same result
            self.assertEqual(str(uc), expected)
            self.assertEqual('{}'.format(uc), expected)
            self.assertEqual('{!s}'.format(uc), expected)
            self.assertEqual('{:+.2f}'.format(uc), expected)
            self.assertEqual('{:+.2}'.format(uc), expected)
            self.assertEqual('{:+f}'.format(uc), expected)

        check(ucomplex(1.23456789 + 9.87654321j, 1000),
              '(+0(1000)+0(1000)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 100),
              '(+0(100)+10(100)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 10),
              '(+1(10)+10(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 1),
              '(+1.2(1.0)+9.9(1.0)j)')

        check(ucomplex(1.23456789 + 9.87654321j, 0.1),
              '(+1.23(10)+9.88(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.01),
              '(+1.235(10)+9.877(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.001),
              '(+1.2346(10)+9.8765(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.0001),
              '(+1.23457(10)+9.87654(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.00001),
              '(+1.234568(10)+9.876543(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.000001),
              '(+1.2345679(10)+9.8765432(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.0000001),
              '(+1.23456789(10)+9.87654321(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.00000001),
              '(+1.234567890(10)+9.876543210(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.000000001),
              '(+1.2345678900(10)+9.8765432100(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.0000000001),
              '(+1.23456789000(10)+9.87654321000(10)j)')
        check(ucomplex(1.23456789e16 + 9.87654321e14j, 1e13),
              '(+12346000000000000(10000000000000)+988000000000000(10000000000000)j)')
        check(ucomplex(1.23456789e-16 + 9.87654321e-14j, 1e-18),
              '(+0.0000000000000001235(10)+0.0000000000000987654(10)j)')

    def test_bracket_nan_inf_ureal(self):
        ur = UncertainReal._elementary(inf, inf, inf, None, True)
        self.assertEqual(str(ur), ' inf(inf)')
        self.assertEqual('{}'.format(ur), ' inf(inf)')
        self.assertEqual('{!s}'.format(ur), ' inf(inf)')
        for t in ['f', 'g', 'e']:
            fmt = create_format(ur, type=t)
            self.assertEqual(to_string(ur, fmt), 'inf(inf)')
            self.assertEqual(to_string(ur.x, fmt), 'inf')
            self.assertEqual(to_string(ur.u, fmt), 'inf')
        for t in ['F', 'G', 'E']:
            fmt = create_format(ur, type=t)
            self.assertEqual(to_string(ur, fmt), 'INF(INF)')
            self.assertEqual(to_string(ur.x, fmt), 'INF')
            self.assertEqual(to_string(ur.u, fmt), 'INF')

        ur = UncertainReal._elementary(inf, nan, inf, None, True)
        self.assertEqual(str(ur), ' inf(nan)')
        self.assertEqual('{}'.format(ur), ' inf(nan)')
        self.assertEqual('{!s}'.format(ur), ' inf(nan)')
        for t in ['f', 'g', 'e']:
            fmt = create_format(ur, type=t)
            self.assertEqual(to_string(ur, fmt), 'inf(nan)')
            self.assertEqual(to_string(ur.x, fmt), 'inf')
            self.assertEqual(to_string(ur.u, fmt), 'nan')
        for t in ['F', 'G', 'E']:
            fmt = create_format(ur, type=t)
            self.assertEqual(to_string(ur, fmt), 'INF(NAN)')
            self.assertEqual(to_string(ur.x, fmt), 'INF')
            self.assertEqual(to_string(ur.u, fmt), 'NAN')

        ur = UncertainReal._elementary(-inf, nan, inf, None, True)
        self.assertEqual(str(ur), '-inf(nan)')
        self.assertEqual('{}'.format(ur), '-inf(nan)')
        self.assertEqual('{!s}'.format(ur), '-inf(nan)')
        for t in ['f', 'g', 'e']:
            fmt = create_format(ur, type=t)
            self.assertEqual(to_string(ur, fmt), '-inf(nan)')
            self.assertEqual(to_string(ur.x, fmt), '-inf')
            self.assertEqual(to_string(ur.u, fmt), 'nan')
        for t in ['F', 'G', 'E']:
            fmt = create_format(ur, type=t)
            self.assertEqual(to_string(ur, fmt), '-INF(NAN)')
            self.assertEqual(to_string(ur.x, fmt), '-INF')
            self.assertEqual(to_string(ur.u, fmt), 'NAN')

        ur = UncertainReal._elementary(nan, inf, inf, None, True)
        self.assertEqual(str(ur), ' nan(inf)')
        self.assertEqual('{}'.format(ur), ' nan(inf)')
        self.assertEqual('{!s}'.format(ur), ' nan(inf)')
        for t in ['f', 'g', 'e']:
            fmt = create_format(ur, type=t)
            self.assertEqual(to_string(ur, fmt), 'nan(inf)')
            self.assertEqual(to_string(ur.x, fmt), 'nan')
            self.assertEqual(to_string(ur.u, fmt), 'inf')
        for t in ['F', 'G', 'E']:
            fmt = create_format(ur, type=t)
            self.assertEqual(to_string(ur, fmt), 'NAN(INF)')
            self.assertEqual(to_string(ur.x, fmt), 'NAN')
            self.assertEqual(to_string(ur.u, fmt), 'INF')

        ur = UncertainReal._elementary(nan, nan, inf, None, True)
        self.assertEqual(str(ur), ' nan(nan)')
        self.assertEqual('{}'.format(ur), ' nan(nan)')
        self.assertEqual('{!s}'.format(ur), ' nan(nan)')
        for t in ['f', 'g', 'e']:
            fmt = create_format(ur, type=t)
            self.assertEqual(to_string(ur, fmt), 'nan(nan)')
            self.assertEqual(to_string(ur.x, fmt), 'nan')
            self.assertEqual(to_string(ur.u, fmt), 'nan')
        for t in ['F', 'G', 'E']:
            fmt = create_format(ur, type=t)
            self.assertEqual(to_string(ur, fmt), 'NAN(NAN)')
            self.assertEqual(to_string(ur.x, fmt), 'NAN')
            self.assertEqual(to_string(ur.u, fmt), 'NAN')

        ur = UncertainReal._elementary(3.141, inf, inf, None, True)
        self.assertEqual(str(ur), ' 3.14(inf)')
        self.assertEqual('{: F}'.format(ur), ' 3.14(INF)')
        self.assertEqual('{: .3f}'.format(ur), ' 3.141(inf)')

        ur = UncertainReal._elementary(3.141e8, inf, inf, None, True)
        self.assertEqual(str(ur), ' 314100000.00(inf)')
        self.assertEqual('{: .1F}'.format(ur), ' 314100000.0(INF)')
        self.assertEqual('{: .1e}'.format(ur), ' 3.1(inf)e+08')
        self.assertEqual('{: .4E}'.format(ur), ' 3.1410(INF)E+08')

        ur = UncertainReal._elementary(3.141, nan, inf, None, True)
        self.assertEqual(str(ur), ' 3.14(nan)')
        self.assertEqual('{: F}'.format(ur), ' 3.14(NAN)')
        self.assertEqual('{: .1F}'.format(ur), ' 3.1(NAN)')

        ur = UncertainReal._elementary(nan, 3.141, inf, None, True)
        self.assertEqual(str(ur), ' nan(3.141)')
        self.assertEqual('{: F}'.format(ur), ' NAN(3.141)')

        ur = UncertainReal._elementary(nan, 3.141e8, inf, None, True)
        self.assertEqual(str(ur), ' nan(314100000)')
        self.assertEqual('{: E}'.format(ur), ' NAN(3)E+08')
        self.assertEqual('{: e}'.format(ur), ' nan(3)e+08')

    def test_bracket_nan_inf_ucomplex(self):
        uc = UncertainComplex._elementary(
            complex(inf, inf), inf, inf,
            None, inf, None, True
        )

        self.assertEqual(str(uc), '(+inf(inf)+inf(inf)j)')
        self.assertEqual('{}'.format(uc), '(+inf(inf)+inf(inf)j)')
        self.assertEqual('{!s}'.format(uc), '(+inf(inf)+inf(inf)j)')
        for t in ['f', 'g', 'e']:
            fmt = create_format(uc, type=t, sign='+')
            self.assertEqual(to_string(uc, fmt), '(+inf(inf)+inf(inf)j)')
            self.assertEqual(to_string(uc.x, fmt), '(+inf+infj)')
            self.assertEqual(to_string(uc.u, fmt), '(+inf+infj)')
            fmt = create_format(uc, type=t)
            self.assertEqual(to_string(uc, fmt), '(inf(inf)+inf(inf)j)')
            self.assertEqual(to_string(uc.x, fmt), '(inf+infj)')
            self.assertEqual(to_string(uc.u, fmt), '(inf+infj)')
        for t in ['F', 'G', 'E']:
            fmt = create_format(uc, type=t, sign='+')
            self.assertEqual(to_string(uc, fmt), '(+INF(INF)+INF(INF)j)')
            self.assertEqual(to_string(uc.x, fmt), '(+INF+INFj)')
            self.assertEqual(to_string(uc.u, fmt), '(+INF+INFj)')
            fmt = create_format(uc, type=t)
            self.assertEqual(to_string(uc, fmt), '(INF(INF)+INF(INF)j)')
            self.assertEqual(to_string(uc.x, fmt), '(INF+INFj)')
            self.assertEqual(to_string(uc.u, fmt), '(INF+INFj)')

        uc = UncertainComplex._elementary(
            complex(nan, -inf), inf, nan,
            None, inf, None, True
        )

        self.assertEqual(str(uc), '(+nan(inf)-inf(nan)j)')
        self.assertEqual('{}'.format(uc), '(+nan(inf)-inf(nan)j)')
        self.assertEqual('{!s}'.format(uc), '(+nan(inf)-inf(nan)j)')
        for t in ['f', 'g', 'e']:
            fmt = create_format(uc, type=t, sign='+')
            self.assertEqual(to_string(uc, fmt), '(+nan(inf)-inf(nan)j)')
            self.assertEqual(to_string(uc.x, fmt), '(+nan-infj)')
            self.assertEqual(to_string(uc.u, fmt), '(+inf+nanj)')
            fmt = create_format(uc, type=t)
            self.assertEqual(to_string(uc, fmt), '(nan(inf)-inf(nan)j)')
            self.assertEqual(to_string(uc.x, fmt), '(nan-infj)')
            self.assertEqual(to_string(uc.u, fmt), '(inf+nanj)')
        for t in ['F', 'G', 'E']:
            fmt = create_format(uc, type=t, sign='+')
            self.assertEqual(to_string(uc, fmt), '(+NAN(INF)-INF(NAN)j)')
            self.assertEqual(to_string(uc.x, fmt), '(+NAN-INFj)')
            self.assertEqual(to_string(uc.u, fmt), '(+INF+NANj)')
            fmt = create_format(uc, type=t)
            self.assertEqual(to_string(uc, fmt), '(NAN(INF)-INF(NAN)j)')
            self.assertEqual(to_string(uc.x, fmt), '(NAN-INFj)')
            self.assertEqual(to_string(uc.u, fmt), '(INF+NANj)')

        uc = UncertainComplex._elementary(
            complex(1.2, -3.4), inf, nan,
            None, inf, None, True
        )
        fmt = create_format(uc, sign='+')
        self.assertEqual(to_string(uc, fmt), '(+1.20(inf)-3.40(nan)j)')
        self.assertEqual(to_string(uc.x, fmt), '(+1.20-3.40j)')
        self.assertEqual(to_string(uc.u, fmt), '(+inf+nanj)')

        uc = UncertainComplex._elementary(
            complex(nan, -inf), 1.2, 3.4,
            None, inf, None, True
        )
        fmt = create_format(uc, sign='+')
        self.assertEqual(to_string(uc, fmt), '(+nan(1.200)-inf(3.400)j)')
        self.assertEqual(to_string(uc.x, fmt), '(+nan-infj)')
        self.assertEqual(to_string(uc.u, fmt), '(+1.2+3.4j)')

    def test_bracket_type_f_ureal(self):
        ur = ureal(1.23456789, 0.0123456789)

        fmt = create_format(ur, digits=1)
        self.assertEqual(to_string(ur, fmt),   '1.23(1)')
        self.assertEqual('{:.1}'.format(ur),   '1.23(1)')  # f and B are defaults
        self.assertEqual('{:.1f}'.format(ur),  '1.23(1)')
        self.assertEqual(to_string(ur.x, fmt), '1.23')
        self.assertEqual(to_string(ur.u, fmt), '0.01')
        self.assertEqual(to_string(1, fmt),    '1.00')
        self.assertEqual(to_string(nan, fmt),  'nan')
        self.assertEqual(to_string(-inf, fmt), '-inf')

        fmt = create_format(ur, digits=2)
        self.assertEqual(to_string(ur, fmt),   '1.235(12)')
        self.assertEqual('{:.2f}'.format(ur),  '1.235(12)')
        self.assertEqual(to_string(ur.x, fmt), '1.235')
        self.assertEqual(to_string(ur.u, fmt), '0.012')

        fmt = create_format(ur, digits=3)
        self.assertEqual(to_string(ur, fmt),   '1.2346(123)')
        self.assertEqual('{:.3f}'.format(ur),  '1.2346(123)')
        self.assertEqual(to_string(ur.x, fmt), '1.2346')
        self.assertEqual(to_string(ur.u, fmt), '0.0123')

        fmt = create_format(ur, digits=9)
        self.assertEqual(to_string(ur, fmt),   '1.2345678900(123456789)')
        self.assertEqual('{:.9f}'.format(ur),  '1.2345678900(123456789)')
        self.assertEqual(to_string(ur.x, fmt), '1.2345678900')
        self.assertEqual(to_string(ur.u, fmt), '0.0123456789')

        fmt = create_format(ur, digits=14)
        self.assertEqual(to_string(ur, fmt),   '1.234567890000000(12345678900000)')
        self.assertEqual('{:.14f}'.format(ur), '1.234567890000000(12345678900000)')
        self.assertEqual(to_string(ur.x, fmt), '1.234567890000000')
        self.assertEqual(to_string(ur.u, fmt), '0.012345678900000')

        u = ur * (10 ** -20)
        fmt = create_format(u, type='f', digits=4, width=39, fill=' ', align='>')
        self.assertEqual(to_string(u, fmt),   '      0.0000000000000000000123457(1235)')
        self.assertEqual(to_string(u.x, fmt), '            0.0000000000000000000123457')
        self.assertEqual(to_string(u.u, fmt), '            0.0000000000000000000001235')

        u = ur * (10 ** -19)
        fmt = create_format(u, type='f', digits=4, width=39, fill=' ', align='>')
        self.assertEqual(to_string(u, fmt),   '       0.000000000000000000123457(1235)')
        self.assertEqual(to_string(u.x, fmt), '             0.000000000000000000123457')
        self.assertEqual(to_string(u.u, fmt), '             0.000000000000000000001235')

        u = ur * (10 ** -18)
        fmt = create_format(u, type='f', digits=4, width=39, fill=' ', align='<')
        self.assertEqual(to_string(u, fmt),   '0.00000000000000000123457(1235)        ')
        self.assertEqual(to_string(u.x, fmt), '0.00000000000000000123457              ')
        self.assertEqual(to_string(u.u, fmt), '0.00000000000000000001235              ')

        u = ur * (10 ** -12)
        fmt = create_format(u, type='f', digits=4, width=39, fill='-', align='^')
        self.assertEqual(to_string(u, fmt),   '-------0.00000000000123457(1235)-------')
        self.assertEqual(to_string(u.x, fmt), '----------0.00000000000123457----------')
        self.assertEqual(to_string(u.u, fmt), '----------0.00000000000001235----------')

        u = ur * (10 ** -6)
        fmt = create_format(u, type='f', digits=4, width=19)
        self.assertEqual(to_string(u, fmt),   '0.00000123457(1235)')
        self.assertEqual(to_string(u.x, fmt), '0.00000123457      ')
        self.assertEqual(to_string(u.u, fmt), '0.00000001235      ')

        u = ur * (10 ** 0)
        fmt = create_format(u, type='f', digits=4, width=15, fill=' ', align='>')
        self.assertEqual(to_string(u, fmt),   '  1.23457(1235)')
        self.assertEqual(to_string(u.x, fmt), '        1.23457')
        self.assertEqual(to_string(u.u, fmt), '        0.01235')

        u = ur * (10 ** 1)
        fmt = create_format(u, type='f', digits=4, width=15, fill=' ', align='>')
        self.assertEqual(to_string(u, fmt),   '  12.3457(1235)')
        self.assertEqual(to_string(u.x, fmt), '        12.3457')
        self.assertEqual(to_string(u.u, fmt), '         0.1235')

        u = ur * (10 ** 2)
        fmt = create_format(u, type='f', digits=4, width=15, fill=' ', align='>')
        self.assertEqual(to_string(u, fmt),   ' 123.457(1.235)')
        self.assertEqual(to_string(u.x, fmt), '        123.457')
        self.assertEqual(to_string(u.u, fmt), '          1.235')

        u = ur * (10 ** 3)
        fmt = create_format(u, type='f', digits=4, width=15, fill=' ', align='>')
        self.assertEqual(to_string(u, fmt),   ' 1234.57(12.35)')
        self.assertEqual(to_string(u.x, fmt), '        1234.57')
        self.assertEqual(to_string(u.u, fmt), '          12.35')

        u = ur * (10 ** 4)
        fmt = create_format(u, type='f', digits=4, width=15, fill=' ', align='>')
        self.assertEqual(to_string(u, fmt),   ' 12345.7(123.5)')
        self.assertEqual(to_string(u.x, fmt), '        12345.7')
        self.assertEqual(to_string(u.u, fmt), '          123.5')

        u = ur * (10 ** 5)
        fmt = create_format(u, type='f', digits=4, width=15, fill=' ', align='>')
        self.assertEqual(to_string(u, fmt),   '   123457(1235)')
        self.assertEqual(to_string(u.x, fmt), '         123457')
        self.assertEqual(to_string(u.u, fmt), '           1235')

        u = ur * (10 ** 6)
        fmt = create_format(u, type='f', digits=4, sign='+', width=20, fill=' ', align='>')
        self.assertEqual(to_string(u, fmt),   '     +1234570(12350)')
        self.assertEqual(to_string(u.x, fmt), '            +1234570')
        self.assertEqual(to_string(u.u, fmt), '              +12350')

        u = ur * (10 ** 7)
        fmt = create_format(u, type='f', digits=4, width=16, fill=' ', align='>')
        self.assertEqual(to_string(u, fmt),   '12345700(123500)')
        self.assertEqual(to_string(u.x, fmt), '        12345700')
        self.assertEqual(to_string(u.u, fmt), '          123500')

        u = ur * (10 ** 8)
        fmt = create_format(u, type='f', digits=4)
        self.assertEqual(to_string(u, fmt),   '123457000(1235000)')
        self.assertEqual(to_string(u.x, fmt), '123457000')
        self.assertEqual(to_string(u.u, fmt),   '1235000')

        u = ur * (10 ** 18)
        fmt = create_format(u, type='f', digits=4)
        self.assertEqual(to_string(u, fmt),   '1234570000000000000(12350000000000000)')
        self.assertEqual(to_string(u.x, fmt), '1234570000000000000')
        self.assertEqual(to_string(u.u, fmt),   '12350000000000000')

        ur = ureal(1.23456789, 1234.56789)
        fmt = create_format(ur, sign=' ', digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), ' 0(1200)')
        self.assertEqual(to_string(ur.x, fmt),     ' 0')
        self.assertEqual(to_string(ur.u, fmt),  ' 1200')

        ur = ureal(1.23456789, 123.456789)
        fmt = create_format(ur, sign=' ', digits=2, type='f')
        self.assertEqual(to_string(ur, fmt),     ' 0(120)')
        self.assertEqual(to_string(ur.x, fmt),   ' 0')
        self.assertEqual(to_string(ur.u, fmt), ' 120')

        ur = ureal(1.23456789, 12.3456789)
        fmt = create_format(ur, sign=' ', digits=2, type='f')
        self.assertEqual(to_string(ur, fmt),    ' 1(12)')
        self.assertEqual(to_string(ur.x, fmt),  ' 1')
        self.assertEqual(to_string(ur.u, fmt), ' 12')

        ur = ureal(1.23456789, 1.23456789)
        fmt = create_format(ur, sign=' ', digits=2, type='f')
        self.assertEqual(to_string(ur, fmt),   ' 1.2(1.2)')
        self.assertEqual(to_string(ur.x, fmt), ' 1.2')
        self.assertEqual(to_string(ur.u, fmt), ' 1.2')

        ur = ureal(1.23456789, 0.123456789)
        fmt = create_format(ur, sign=' ', digits=2, type='f')
        self.assertEqual(to_string(ur, fmt),   ' 1.23(12)')
        self.assertEqual(to_string(ur.x, fmt), ' 1.23')
        self.assertEqual(to_string(ur.u, fmt), ' 0.12')

        ur = ureal(1.23456789, 0.0123456789)
        fmt = create_format(ur, sign=' ', digits=2, type='f')
        self.assertEqual(to_string(ur, fmt),   ' 1.235(12)')
        self.assertEqual(to_string(ur.x, fmt), ' 1.235')
        self.assertEqual(to_string(ur.u, fmt), ' 0.012')

        ur = ureal(1.23456789, 0.00123456789)
        fmt = create_format(ur, sign=' ', digits=2, type='f')
        self.assertEqual(to_string(ur, fmt),   ' 1.2346(12)')
        self.assertEqual(to_string(ur.x, fmt), ' 1.2346')
        self.assertEqual(to_string(ur.u, fmt), ' 0.0012')

        ur = ureal(1.23456789, 0.000123456789)
        fmt = create_format(ur, sign=' ', digits=2, type='f')
        self.assertEqual(to_string(ur, fmt),   ' 1.23457(12)')
        self.assertEqual(to_string(ur.x, fmt), ' 1.23457')
        self.assertEqual(to_string(ur.u, fmt), ' 0.00012')

        ur = ureal(1.23456789, 0.000000123456789)
        fmt = create_format(ur, sign=' ', digits=2, type='f')
        self.assertEqual(to_string(ur, fmt),   ' 1.23456789(12)')
        self.assertEqual(to_string(ur.x, fmt), ' 1.23456789')
        self.assertEqual(to_string(ur.u, fmt), ' 0.00000012')

        ur = ureal(1.23456789, 0.000000000123456789)
        fmt = create_format(ur, sign=' ', digits=2, type='f')
        self.assertEqual(to_string(ur, fmt),   ' 1.23456789000(12)')
        self.assertEqual(to_string(ur.x, fmt), ' 1.23456789000')
        self.assertEqual(to_string(ur.u, fmt), ' 0.00000000012')

        ur = ureal(1.23456789e-4, 0.000000000123456789)
        fmt = create_format(ur, sign=' ', digits=2, type='f')
        self.assertEqual(to_string(ur, fmt),   ' 0.00012345679(12)')
        self.assertEqual(to_string(ur.x, fmt), ' 0.00012345679')
        self.assertEqual(to_string(ur.u, fmt), ' 0.00000000012')

        ur = ureal(1.23456789e4, 0.000000123456789)
        fmt = create_format(ur, sign=' ', digits=2, type='f')
        self.assertEqual(to_string(ur, fmt),   ' 12345.67890000(12)')
        self.assertEqual(to_string(ur.x, fmt), ' 12345.67890000')
        self.assertEqual(to_string(ur.u, fmt),     ' 0.00000012')

        ur = ureal(1.23456789, 0.0123456789)
        fmt = create_format(ur.x)  # use the value and the default kwargs
        self.assertEqual(to_string(ur, fmt),   '1.2(0.0)')
        self.assertEqual(to_string(ur.x, fmt), '1.2')
        self.assertEqual(to_string(ur.u, fmt), '0.0')

        ur = ureal(1.23456789, 0.0123456789)
        fmt = create_format(ur.u)  # use the uncertainty and the default kwargs
        self.assertEqual(to_string(ur, fmt),   '1.235(12)')
        self.assertEqual(to_string(ur.x, fmt), '1.235')
        self.assertEqual(to_string(ur.u, fmt), '0.012')

        ur = ureal(123456789., 1234.56789)

        fmt = create_format(ur, digits=6, type='e')
        self.assertEqual(to_string(ur, fmt),   '1.2345678900(123457)e+08')
        self.assertEqual(to_string(ur.x, fmt), '1.2345678900e+08')
        self.assertEqual(to_string(ur.u, fmt), '1.23457e+03')

        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt),   '123457000(1000)')
        self.assertEqual(to_string(ur.x, fmt), '123457000')
        self.assertEqual(to_string(ur.u, fmt), '1000')

        ur = ureal(1.23456789, 0.12345)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '1.2(1)')
        fmt = create_format(ur, digits=4, type='f')
        self.assertEqual(to_string(ur, fmt), '1.2346(1235)')

        ur = ureal(1.23456789, 0.945)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '1.2(9)')
        fmt = create_format(-ur, digits=2, type='f')
        self.assertEqual(to_string(-ur, fmt), '-1.23(94)')

        ur = ureal(1.23456789, 0.95)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '1.2(9)')
        fmt = create_format(ur, digits=3, type='f', sign='+')
        self.assertEqual(to_string(ur, fmt), '+1.235(950)')

        ur = ureal(1.23456789, 0.951)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '1(1)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '1.23(95)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '1.235(951)')

        ur = ureal(1.23456789, 0.999999999999)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '1(1)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '1.2(1.0)')
        fmt = create_format(ur, digits=5, type='f')
        self.assertEqual(to_string(ur, fmt), '1.2346(1.0000)')

        ur = ureal(1.23456789, 1.5)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '1(2)')

        ur = ureal(1.23456789, 9.5)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(10)')

        ur = ureal(1.23456789, 10.00)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(10)')

        ur = ureal(123.456789, 0.321)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '123.5(3)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '123.46(32)')

        ur = ureal(123.456789, 0.95)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '123.5(9)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '123.457(950)')

        ur = ureal(123.456789, 0.951)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '123(1)')
        fmt = create_format(ur, digits=4, type='f')
        self.assertEqual(to_string(ur, fmt), '123.4568(9510)')

        ur = ureal(123.456789, 0.999999999999999)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '123(1)')
        fmt = create_format(-ur, digits=6, type='f')
        self.assertEqual(to_string(-ur, fmt), '-123.45679(1.00000)')

        ur = ureal(0.9876, 0.1234)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '1.0(1)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '0.988(123)')

        ur = ureal(0.000003512, 0.00000006551)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.00000351(7)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '0.000003512(66)')

        ur = ureal(0.000003512, 0.0000008177)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.0000035(8)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '0.000003512(818)')

        ur = ureal(0.000003512, 0.000009773)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.00000(1)')
        fmt = create_format(ur, digits=4, type='f')
        self.assertEqual(to_string(ur, fmt), '0.000003512(9773)')

        ur = ureal(0.000003512, 0.00001241)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.00000(1)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '0.000004(12)')

        ur = ureal(0.000003512, 0.0009998)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.000(1)')
        fmt = create_format(ur, digits=4, type='f')
        self.assertEqual(to_string(ur, fmt), '0.0000035(9998)')

        ur = ureal(0.000003512, 0.006563)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.000(7)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '0.0000(66)')

        ur = ureal(0.000003512, 0.09564)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.0(1)')
        fmt = create_format(ur, digits=4, type='f')
        self.assertEqual(to_string(ur, fmt), '0.00000(9564)')

        ur = ureal(0.000003512, 0.7772)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.0(8)')

        ur = ureal(0.000003512, 9.75)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(10)')

        ur = ureal(0.000003512, 33.97)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(30)')

        ur = ureal(0.000003512, 715.5)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(700)')
        fmt = create_format(ur, digits=5, type='f')
        self.assertEqual(to_string(ur, fmt), '0.00(715.50)')

        ur = ureal(0.07567, 0.00000007018)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.07567000(7)')
        fmt = create_format(ur, digits=5, type='f')
        self.assertEqual(to_string(ur, fmt), '0.075670000000(70180)')

        ur = ureal(0.07567, 0.0000003645)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.0756700(4)')
        fmt = create_format(-ur, digits=3, type='f')
        self.assertEqual(to_string(-ur, fmt), '-0.075670000(365)')

        ur = ureal(0.07567, 0.000005527)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.075670(6)')
        fmt = create_format(ur, digits=2, type='F', sign=' ')
        self.assertEqual(to_string(ur, fmt), ' 0.0756700(55)')

        ur = ureal(0.07567, 0.00004429)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.07567(4)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '0.075670(44)')

        ur = ureal(0.07567, 0.0008017)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.0757(8)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '0.07567(80)')

        ur = ureal(0.07567, 0.006854)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.076(7)')
        fmt = create_format(ur, digits=4, type='f')
        self.assertEqual(to_string(ur, fmt), '0.075670(6854)')

        ur = ureal(0.07567, 0.06982)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.08(7)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '0.076(70)')

        ur = ureal(0.07567, 0.7382)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0.1(7)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '0.076(738)')

        ur = ureal(0.07567, 7.436)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(7)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '0.1(7.4)')

        ur = ureal(0.07567, 48.75)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(50)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '0.1(48.8)')

        ur = ureal(0.07567, 487.9)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(500)')
        fmt = create_format(ur, digits=5, type='f')
        self.assertEqual(to_string(ur, fmt), '0.08(487.90)')

        ur = ureal(8.545, 0.00000007513)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '8.54500000(8)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '8.545000000(75)')

        ur = ureal(8.545, 0.000009935)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '8.54500(1)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '8.5450000(99)')

        ur = ureal(8.545, 0.003243)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '8.545(3)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '8.54500(324)')

        ur = ureal(8.545, 0.0812)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '8.54(8)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '8.545(81)')

        ur = ureal(8.545, 0.4293)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '8.5(4)')
        fmt = create_format(ur, digits=4, type='f')
        self.assertEqual(to_string(ur, fmt), '8.5450(4293)')

        ur = ureal(8.545, 6.177)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '9(6)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '8.5(6.2)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '8.54(6.18)')
        fmt = create_format(ur, digits=4, type='f')
        self.assertEqual(to_string(ur, fmt), '8.545(6.177)')
        fmt = create_format(ur, digits=7, type='f')
        self.assertEqual(to_string(ur, fmt), '8.545000(6.177000)')

        ur = ureal(8.545, 26.02)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '10(30)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '8.5(26.0)')

        ur = ureal(8.545, 406.1)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(400)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '9(406)')

        ur = ureal(8.545, 3614.0)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(4000)')
        fmt = create_format(ur, digits=5, type='f')
        self.assertEqual(to_string(ur, fmt), '8.5(3614.0)')

        ur = ureal(89.95, 0.00000006815)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '89.95000000(7)')
        fmt = create_format(ur, digits=4, type='f')
        self.assertEqual(to_string(ur, fmt), '89.95000000000(6815)')

        ur = ureal(89.95, 0.0000002651)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '89.9500000(3)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '89.95000000(27)')

        ur = ureal(89.95, 0.0001458)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '89.9500(1)')
        fmt = create_format(ur, digits=4, type='f')
        self.assertEqual(to_string(ur, fmt), '89.9500000(1458)')

        ur = ureal(89.95, 0.009532)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '89.95(1)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '89.9500(95)')

        ur = ureal(89.95, 0.09781)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '90.0(1)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '89.950(98)')

        ur = ureal(89.95, 0.7335)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '90.0(7)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '89.95(73)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '89.950(734)')

        ur = ureal(89.95, 3.547)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '90(4)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '90.0(3.5)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '89.95(3.55)')
        fmt = create_format(ur, digits=4, type='f')
        self.assertEqual(to_string(ur, fmt), '89.950(3.547)')

        ur = ureal(89.95, 31.4)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '90(30)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '90(31)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '90.0(31.4)')

        ur = ureal(89.95, 623.1)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '100(600)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '90(620)')

        ur = ureal(89.95, 2019.0)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(2000)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '90(2020)')

        ur = ureal(89.95, 94600.0)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(90000)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '100(94600)')

        ur = ureal(58740.0, 0.00000001402)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '58740.00000000(1)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '58740.000000000(14)')

        ur = ureal(58740.0, 0.000000975)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '58740.000000(1)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '58740.00000000(97)')

        ur = ureal(58740.0, 0.0001811)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '58740.0000(2)')
        fmt = create_format(ur, digits=4, type='f')
        self.assertEqual(to_string(ur, fmt), '58740.0000000(1811)')

        ur = ureal(58740.0, 0.04937)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '58740.00(5)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '58740.000(49)')

        ur = ureal(58740.0, 0.6406)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '58740.0(6)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '58740.000(641)')

        ur = ureal(58740.0, 9.357)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '58740(9)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '58740.0(9.4)')

        ur = ureal(58740.0, 99.67)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '58700(100)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '58740(100)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '58740.0(99.7)')

        ur = ureal(58740.0, 454.6)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '58700(500)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '58740(455)')

        ur = ureal(58740.0, 1052.0)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '59000(1000)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '58700(1100)')

        ur = ureal(58740.0, 87840.0)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '60000(90000)')
        fmt = create_format(ur, digits=3, type='f')
        self.assertEqual(to_string(ur, fmt), '58700(87800)')

        ur = ureal(58740.0, 5266000.0)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(5000000)')
        fmt = create_format(ur, digits=4, type='f')
        self.assertEqual(to_string(ur, fmt), '59000(5266000)')

        ur = ureal(58740.0, 97769999.0)
        fmt = create_format(ur, digits=1, type='f')
        self.assertEqual(to_string(ur, fmt), '0(100000000)')
        fmt = create_format(ur, digits=2, type='f')
        self.assertEqual(to_string(ur, fmt), '0(98000000)')
        fmt = create_format(ur, digits=5, type='f')
        self.assertEqual(to_string(ur, fmt), '59000(97770000)')

    def test_bracket_type_f_ucomplex(self):
        uc = ucomplex(1.23456789e6 + 9.87654321j, [1234.56789, 0.342567])
        fmt = create_format(uc, sign='+')
        self.assertEqual(to_string(uc, fmt),   '(+1234567.89(1234.57)+9.88(34)j)')
        self.assertEqual(to_string(uc.x, fmt), '(+1234567.89+9.88j)')
        self.assertEqual(to_string(uc.u, fmt), '(+1234.57+0.34j)')
        self.assertEqual(to_string(9.87654321j, fmt), '(+0.00+9.88j)')
        self.assertEqual(to_string(1.23456789e6 + 9.87654321j, fmt), '(+1234567.89+9.88j)')
        self.assertEqual(to_string(complex(1.23456789e6, 9.87654321), fmt), '(+1234567.89+9.88j)')

        uc = ucomplex(1.23456789e6 + 9.87654321j, [0.342567, 13.56789])
        fmt = create_format(uc, sign='+')
        self.assertEqual(to_string(uc, fmt),   '(+1234567.89(34)+9.88(13.57)j)')
        self.assertEqual(to_string(uc.x, fmt), '(+1234567.89+9.88j)')
        self.assertEqual(to_string(uc.u, fmt), '(+0.34+13.57j)')

        uc = ucomplex(12.3456789 + 0.87654321j, [0.342567, 0.00056789])

        fmt = create_format(uc, sign='+', digits=4)
        self.assertEqual(to_string(uc, fmt),   '(+12.3456789(3425670)+0.8765432(5679)j)')
        self.assertEqual(to_string(uc.x, fmt), '(+12.3456789+0.8765432j)')
        self.assertEqual(to_string(uc.u, fmt), '(+0.3425670+0.0005679j)')

        fmt = create_format(uc, sign=' ', fill=' ', align='>', width=40)
        self.assertEqual(to_string(uc, fmt),   '         ( 12.34568(34257)+0.87654(57)j)')
        self.assertEqual(to_string(uc.x, fmt), '                    ( 12.34568+0.87654j)')
        self.assertEqual(to_string(uc.u, fmt), '                     ( 0.34257+0.00057j)')

        fmt = create_format(uc, fill=' ', align='>', width=40)
        self.assertEqual(to_string(uc, fmt),   '          (12.34568(34257)+0.87654(57)j)')
        self.assertEqual(to_string(uc.x, fmt), '                     (12.34568+0.87654j)')
        self.assertEqual(to_string(uc.u, fmt), '                      (0.34257+0.00057j)')

        uc = ucomplex(12.3456789 - 0.87654321j, [0.342567, 0.00056789])
        fmt = create_format(uc, sign='+', fill=' ', align='>', width=40)
        self.assertEqual(to_string(uc, fmt),   '         (+12.34568(34257)-0.87654(57)j)')
        self.assertEqual(to_string(uc.x, fmt), '                    (+12.34568-0.87654j)')
        self.assertEqual(to_string(uc.u, fmt), '                     (+0.34257+0.00057j)')

        fmt = create_format(uc, fill='*', align='<', width=35)
        self.assertEqual(to_string(uc, fmt),   '(12.34568(34257)-0.87654(57)j)*****')
        self.assertEqual(to_string(uc.x, fmt), '(12.34568-0.87654j)****************')
        self.assertEqual(to_string(uc.u, fmt), '(0.34257+0.00057j)*****************')

    def test_uncertainty_is_zero(self):
        ur = ureal(1.23456789, 0)

        fmt = create_format(ur, type='f')
        self.assertEqual('{:.2f}'.format(ur.x), '1.23')
        self.assertEqual(to_string(ur, fmt),    '1.23')
        self.assertEqual(to_string(ur.x, fmt),  '1.23')
        self.assertEqual(to_string(ur.u, fmt),  '0.00')

        fmt = create_format(ur, type='f', digits=4)
        self.assertEqual('{:.4f}'.format(ur.x), '1.2346')
        self.assertEqual(to_string(ur, fmt),    '1.2346')
        self.assertEqual(to_string(ur.x, fmt),  '1.2346')
        self.assertEqual(to_string(ur.u, fmt),  '0.0000')

        fmt = create_format(ur, type='g')
        self.assertEqual('{:.2g}'.format(ur.x), '1.2')
        self.assertEqual(to_string(ur, fmt),    '1.2')
        self.assertEqual(to_string(ur.x, fmt),  '1.2')
        self.assertEqual(to_string(ur.u, fmt),  '0')

        fmt = create_format(ur, type='g', digits=5)
        self.assertEqual('{:.5g}'.format(ur.x), '1.2346')
        self.assertEqual(to_string(ur, fmt),    '1.2346')
        self.assertEqual(to_string(ur.x, fmt),  '1.2346')
        self.assertEqual(to_string(ur.u, fmt),  '0')

        fmt = create_format(ur, type='E')
        self.assertEqual('{:.2E}'.format(ur.x), '1.23E+00')
        self.assertEqual(to_string(ur, fmt),    '1.23E+00')
        self.assertEqual(to_string(ur.x, fmt),  '1.23E+00')
        self.assertEqual(to_string(ur.u, fmt),  '0.00E+00')

        fmt = create_format(ur, type='E', digits=1)
        self.assertEqual('{:.1E}'.format(ur.x), '1.2E+00')
        self.assertEqual(to_string(ur, fmt),    '1.2E+00')
        self.assertEqual(to_string(ur.x, fmt),  '1.2E+00')
        self.assertEqual(to_string(ur.u, fmt),  '0.0E+00')

        uc = ucomplex(12.3456789 + 0.87654321j, 0)

        fmt = create_format(uc, type='f')
        self.assertEqual('{:.2f}'.format(uc.x), '12.35+0.88j')
        self.assertEqual(to_string(uc, fmt),   '(12.35+0.88j)')
        self.assertEqual(to_string(uc.x, fmt), '(12.35+0.88j)')
        self.assertEqual(to_string(uc.u, fmt), '(0.00+0.00j)')

        fmt = create_format(uc, type='f', digits=4)
        self.assertEqual('{:.4f}'.format(uc.x), '12.3457+0.8765j')
        self.assertEqual(to_string(uc, fmt),   '(12.3457+0.8765j)')
        self.assertEqual(to_string(uc.x, fmt), '(12.3457+0.8765j)')
        self.assertEqual(to_string(uc.u, fmt), '(0.0000+0.0000j)')

        fmt = create_format(uc, type='g')
        self.assertEqual('{:.2g}'.format(uc.x), '12+0.88j')
        self.assertEqual(to_string(uc, fmt),   '(12+0.88j)')
        self.assertEqual(to_string(uc.x, fmt), '(12+0.88j)')
        self.assertEqual(to_string(uc.u, fmt), '(0+0j)')

        fmt = create_format(uc, type='g', digits=5)
        self.assertEqual('{:.5g}'.format(uc.x), '12.346+0.87654j')
        self.assertEqual(to_string(uc, fmt),   '(12.346+0.87654j)')
        self.assertEqual(to_string(uc.x, fmt), '(12.346+0.87654j)')
        self.assertEqual(to_string(uc.u, fmt), '(0+0j)')

        fmt = create_format(uc, type='E')
        self.assertEqual('{:.2E}'.format(uc.x), '1.23E+01+8.77E-01j')
        self.assertEqual(to_string(uc, fmt),   '(1.23E+01+8.77E-01j)')
        self.assertEqual(to_string(uc.x, fmt), '(1.23E+01+8.77E-01j)')
        self.assertEqual(to_string(uc.u, fmt), '(0.00E+00+0.00E+00j)')

        fmt = create_format(uc, type='e', digits=1)
        self.assertEqual('{:.1e}'.format(uc.x), '1.2e+01+8.8e-01j')
        self.assertEqual(to_string(uc, fmt),   '(1.2e+01+8.8e-01j)')
        self.assertEqual(to_string(uc.x, fmt), '(1.2e+01+8.8e-01j)')
        self.assertEqual(to_string(uc.u, fmt), '(0.0e+00+0.0e+00j)')

    def test_bracket_type_e_ureal(self):
        ur = ureal(1.23456789, 0.0001)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt),   '1.2346(1)e+00')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt),   '1.234568(100)e+00')

        ur = ureal(1.23456789, 0.96)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1(1)e+00')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '1.23(96)e+00')

        ur = ureal(1.23456789, 1.0)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1(1)e+00')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '1.23(1.00)e+00')

        ur = ureal(123.456789, 0.1)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1.235(1)e+02')
        fmt = create_format(ur, digits=4, type='e')
        self.assertEqual(to_string(ur, fmt), '1.234568(1000)e+02')

        ur = ureal(123.456789, 0.950)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1.235(9)e+02')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '1.2346(95)e+02')

        ur = ureal(123.456789, 0.951)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1.23(1)e+02')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '1.23457(951)e+02')

        ur = ureal(123.456789, 1.0)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1.23(1)e+02')
        fmt = create_format(ur, digits=2, type='E')
        self.assertEqual(to_string(ur, fmt), '1.235(10)E+02')

        ur = ureal(123.456789, 9.123)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1.23(9)e+02')
        fmt = create_format(ur, digits=4, type='e')
        self.assertEqual(to_string(ur, fmt), '1.23457(9123)e+02')

        ur = ureal(123.456789, 9.9)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1.2(1)e+02')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '1.235(99)e+02')

        ur = ureal(123.456789, 94.9)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1.2(9)e+02')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '1.235(949)e+02')

        ur = ureal(-1.23456789, 0.0123456789)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '-1.23(1)e+00')
        fmt = create_format(ur, digits=5, type='e')
        self.assertEqual(to_string(ur, fmt), '-1.234568(12346)e+00')

        ur = ureal(1.257e-6, 0.00007453e-6)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1.25700(7)e-06')
        fmt = create_format(ur, digits=3, type='E', sign='+')
        self.assertEqual(to_string(ur, fmt), '+1.2570000(745)E-06')

        ur = ureal(1.257e-6, 0.00909262e-6)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1.257(9)e-06')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '1.2570(91)e-06')

        ur = ureal(1.257e-6, 0.1174e-6)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1.3(1)e-06')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '1.257(117)e-06')

        ur = ureal(1.257e-6, 7.287e-6)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1(7)e-06')
        fmt = create_format(ur, digits=4, type='e')
        self.assertEqual(to_string(ur, fmt), '1.257(7.287)e-06')

        ur = ureal(1.257e-6, 67.27e-6)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(7)e-05')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '0.1(6.7)e-05')

        ur = ureal(1.257e-6, 124.1e-6)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(1)e-04')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '0.0(1.2)e-04')

        ur = ureal(1.257e-6, 4583.0e-6)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(5)e-03')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '0.00(4.58)e-03')

        ur = ureal(1.257e-6, 74743.0e-6)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(7)e-02')

        ur = ureal(1.257e-6, 4575432.0e-6)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(5)e+00')

        ur = ureal(7.394e-3, 0.00002659e-3)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '7.39400(3)e-03')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '7.3940000(266)e-03')

        ur = ureal(7.394e-3, 0.0007031e-3)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '7.3940(7)e-03')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '7.39400(70)e-03')

        ur = ureal(7.394e-3, 0.003659e-3)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '7.394(4)e-03')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '7.3940(37)e-03')

        ur = ureal(7.394e-3, 0.04227e-3)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '7.39(4)e-03')
        fmt = create_format(ur, digits=4, type='e')
        self.assertEqual(to_string(ur, fmt), '7.39400(4227)e-03')

        ur = ureal(7.394e-3, 0.9072e-3)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '7.4(9)e-03')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '7.394(907)e-03')

        ur = ureal(7.394e-3, 4.577e-3)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '7(5)e-03')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '7.4(4.6)e-03')

        ur = ureal(7.394e-3, 93.41e-3)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1(9)e-02')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '0.74(9.34)e-02')

        ur = ureal(7.394e-3, 421.0e-3)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(4)e-01')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '0.1(4.2)e-01')

        ur = ureal(7.394e-3, 9492.0e-3)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(9)e+00')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '0.01(9.49)e+00')

        ur = ureal(7.394e-3, 39860.0e-3)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(4)e+01')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '0.0(4.0)e+01')

        ur = ureal(2.675e-2, 0.0000019e-2)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '2.675000(2)e-02')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '2.67500000(190)e-02')

        ur = ureal(2.675e-2, 0.00975e-2)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '2.67(1)e-02')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '2.67500(975)e-02')

        ur = ureal(2.675e-2, 0.08942e-2)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '2.67(9)e-02')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '2.675(89)e-02')

        ur = ureal(2.675e-2, 0.8453e-2)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '2.7(8)e-02')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '2.67(85)e-02')

        ur = ureal(2.675e-2, 8.577e-2)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '3(9)e-02')
        fmt = create_format(ur, digits=2, type='E')
        self.assertEqual(to_string(ur, fmt), '2.7(8.6)E-02')
        fmt = create_format(ur, digits=3, type='E')
        self.assertEqual(to_string(ur, fmt), '2.67(8.58)E-02')

        ur = ureal(2.675e-2, 12.37e-2)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(1)e-01')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '0.27(1.24)e-01')

        ur = ureal(2.675e-2, 226.5e-2)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(2)e+00')
        fmt = create_format(ur, digits=4, type='e')
        self.assertEqual(to_string(ur, fmt), '0.027(2.265)e+00')

        ur = ureal(2.675e-2, 964900.0e-2)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(1)e+04')
        fmt = create_format(ur, digits=6, type='e')
        self.assertEqual(to_string(ur, fmt), '0.00003(9.64900)e+03')

        ur = ureal(0.9767, 0.00000001084)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9.7670000(1)e-01')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '9.767000000(108)e-01')

        ur = ureal(0.9767, 0.0000009797)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9.76700(1)e-01')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '9.7670000(98)e-01')

        ur = ureal(0.9767, 0.004542)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9.77(5)e-01')
        fmt = create_format(ur, digits=5, type='e')
        self.assertEqual(to_string(ur, fmt), '9.767000(45420)e-01')

        ur = ureal(0.9767, 0.02781)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9.8(3)e-01')
        fmt = create_format(-ur, digits=3, type='e')
        self.assertEqual(to_string(-ur, fmt), '-9.767(278)e-01')

        ur = ureal(0.9767, 0.4764)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1.0(5)e+00')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '9.8(4.8)e-01')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '9.77(4.76)e-01')
        fmt = create_format(ur, digits=4, type='e')
        self.assertEqual(to_string(ur, fmt), '9.767(4.764)e-01')

        ur = ureal(0.9767, 4.083)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1(4)e+00')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '0.98(4.08)e+00')

        ur = ureal(0.9767, 45.14)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(5)e+01')
        fmt = create_format(ur, digits=4, type='e')
        self.assertEqual(to_string(ur, fmt), '0.098(4.514)e+01')

        ur = ureal(0.9767, 692500.)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(7)e+05')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '0.00(6.92)e+05')

        ur = ureal(2.952, 0.00000006986)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '2.95200000(7)e+00')
        fmt = create_format(ur, digits=5, type='e')
        self.assertEqual(to_string(ur, fmt), '2.952000000000(69860)e+00')

        ur = ureal(2.952, 0.04441)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '2.95(4)e+00')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '2.9520(444)e+00')

        ur = ureal(2.952, 0.1758)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '3.0(2)e+00')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '2.952(176)e+00')

        ur = ureal(2.952, 1.331)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '3(1)e+00')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '3.0(1.3)e+00')

        ur = ureal(2.952, 34.6)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(3)e+01')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '0.30(3.46)e+01')

        ur = ureal(2.952, 46280.)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(5)e+04')
        fmt = create_format(ur, digits=5, type='e')
        self.assertEqual(to_string(ur, fmt), '0.0003(4.6280)e+04')

        ur = ureal(96.34984, 0.00000002628)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9.634984000(3)e+01')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '9.63498400000(263)e+01')

        ur = ureal(96.34984, 0.00008999)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9.634984(9)e+01')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '9.63498400(900)e+01')

        ur = ureal(96.34984, 0.3981)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9.63(4)e+01')
        fmt = create_format(ur, digits=4, type='e')
        self.assertEqual(to_string(ur, fmt), '9.63498(3981)e+01')

        ur = ureal(96.34984, 7.17)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9.6(7)e+01')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '9.635(717)e+01')

        ur = ureal(96.34984, 1074.0)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(1)e+03')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '0.10(1.07)e+03')

        ur = ureal(92270.0, 0.00000004531)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9.227000000000(5)e+04')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '9.22700000000000(453)e+04')

        ur = ureal(92270., 0.007862)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9.2270000(8)e+04')
        fmt = create_format(ur, digits=2, type='e')
        self.assertEqual(to_string(ur, fmt), '9.22700000(79)e+04')

        ur = ureal(92270., 0.2076)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9.22700(2)e+04')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '9.2270000(208)e+04')

        ur = ureal(92270., 2.202)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9.2270(2)e+04')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '9.227000(220)e+04')

        ur = ureal(92270., 49.12)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9.227(5)e+04')
        fmt = create_format(ur, digits=4, type='e')
        self.assertEqual(to_string(ur, fmt), '9.227000(4912)e+04')

        ur = ureal(92270., 19990.)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '9(2)e+04')
        fmt = create_format(ur, digits=6, type='e')
        self.assertEqual(to_string(ur, fmt), '9.22700(1.99900)e+04')

        ur = ureal(92270., 740800.)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '1(7)e+05')
        fmt = create_format(ur, digits=3, type='e')
        self.assertEqual(to_string(ur, fmt), '0.92(7.41)e+05')

        ur = ureal(92270., 1380000.)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(1)e+06')
        fmt = create_format(ur, digits=5, type='e')
        self.assertEqual(to_string(ur, fmt), '0.0923(1.3800)e+06')

        ur = ureal(92270., 29030000.)
        fmt = create_format(ur, digits=1, type='e')
        self.assertEqual(to_string(ur, fmt), '0(3)e+07')
        fmt = create_format(ur, digits=7, type='e')
        self.assertEqual(to_string(ur, fmt), '0.009227(2.903000)e+07')

    def test_type_g(self):
        ur = ureal(43.842, 0.0123)
        self.assertEqual('{:.1g}'.format(ur), '43.84(1)')

        ur = ureal(4384.2, 1.23)
        self.assertEqual('{:.3g}'.format(ur), '4384.20(1.23)')
        self.assertEqual('{:.1G}'.format(ur), '4.384(1)E+03')

        ur = ureal(123456789., 1234.56789)
        self.assertEqual('{:.4g}'.format(ur), '1.23456789(1235)e+08')
        self.assertEqual('{:.2G}'.format(ur), '1.234568(12)E+08')

        ur = ureal(7.2524e-8, 5.429e-10)
        self.assertEqual('{:.2g}'.format(ur), '7.252(54)e-08')
        self.assertEqual('{:.1G}'.format(ur), '7.25(5)E-08')

        ur = ureal(7.2524e4, 5.429e3)
        self.assertEqual('{:.4G}'.format(ur), '7.2524(5429)E+04')
        self.assertEqual('{:.1g}'.format(ur), '7.3(5)e+04')

        uc = ucomplex(12.3456789 + 0.87654321j, 0.31532)
        self.assertEqual('{:.3G}'.format(uc), '(12.346(315)+0.877(315)j)')

        uc = ucomplex(1.23e6 - 87.e3j, (313e4, 4.75e2))
        self.assertEqual('{:.2g}'.format(uc), '(1.23000(3.13000)e+06-8.700(48)e+04j)')

    def test_unicode(self):
        ur = ureal(18.5424, 0.94271)

        for t in ['f', 'F']:
            fmt = create_format(ur, digits=2, type=t, style='U')
            self.assertEqual(to_string(ur, fmt),   '18.54(94)')
            self.assertEqual(to_string(ur.x, fmt), '18.54')
            self.assertEqual(to_string(ur.u, fmt), '0.94')

        for t in ['e', 'E']:
            fmt = create_format(ur, digits=2, type=t, style='U')
            self.assertEqual(to_string(ur, fmt),   u'1.854(94)×10¹')
            self.assertEqual(to_string(ur.x, fmt), u'1.854×10¹')
            self.assertEqual(to_string(ur.u, fmt), u'9.4×10⁻¹')

        ur = ureal(1.23456789, 0.123456789)
        self.assertEqual('{:.3eU}'.format(ur), '1.235(123)')
        self.assertEqual(u'{:.3EU}'.format(ur * 1e-6), u'1.235(123)×10⁻⁶')
        self.assertEqual(u'{:.3EU}'.format(ur * 1e12), u'1.235(123)×10¹²')
        self.assertEqual(u'{:.3eU}'.format(ur * 1e100), u'1.235(123)×10¹⁰⁰')
        self.assertEqual(u'{:.3EU}'.format(ur * 1e-100), u'1.235(123)×10⁻¹⁰⁰')

        uc = ucomplex(18.5424+1.2j, 0.94271)

        for t in ['f', 'F']:
            fmt = create_format(uc, digits=2, type=t, style='U')
            self.assertEqual(to_string(uc, fmt),   '(18.54(94)+1.20(94)j)')
            self.assertEqual(to_string(uc.x, fmt), '(18.54+1.20j)')
            self.assertEqual(to_string(uc.u, fmt), '(0.94+0.94j)')

        for t in ['e', 'E']:
            fmt = create_format(uc, digits=2, type=t, style='U')
            self.assertEqual(to_string(uc, fmt),   u'(1.854(94)×10¹+1.20(94)j)')
            self.assertEqual(to_string(uc.x, fmt), u'(1.854×10¹+1.20j)')
            self.assertEqual(to_string(uc.u, fmt), u'(9.4×10⁻¹+9.4×10⁻¹j)')

    @unittest.skipIf(sys.version_info.major == 2, 'not supported for this version of Python')
    def test_hash_symbol(self):
        ur = ureal(5.4, 1.2)

        fmt = create_format(ur, digits=1, hash='#')
        self.assertEqual('{:#.1}'.format(ur), '5.(1.)')
        self.assertEqual(to_string(ur, fmt), '5.(1.)')
        self.assertEqual(to_string(ur.x, fmt), '5.')
        self.assertEqual(to_string(ur.u, fmt), '1.')
        self.assertEqual(to_string(3, fmt), '3.')

        fmt = create_format(ur, digits=2, hash=True)
        self.assertEqual('{:#.2}'.format(ur), '5.4(1.2)')
        self.assertEqual(to_string(ur, fmt), '5.4(1.2)')
        self.assertEqual(to_string(ur.x, fmt), '5.4')
        self.assertEqual(to_string(ur.u, fmt), '1.2')
        self.assertEqual(to_string(3, fmt), '3.0')

        uc = ucomplex(5.4 + 7.2j, (1.4, 1.2))
        fmt = create_format(uc, digits=1, hash='#')
        self.assertEqual('{:#.1}'.format(uc), '(5.(1.)+7.(1.)j)')
        self.assertEqual(to_string(uc, fmt), '(5.(1.)+7.(1.)j)')
        self.assertEqual(to_string(uc.x, fmt), '(5.+7.j)')
        self.assertEqual(to_string(uc.u, fmt), '(1.+1.j)')

        ur = ureal(1, 0.001)
        fmt = create_format(ur, hash=1, digits=1)
        self.assertEqual('{:#.1}'.format(ur),  '1.000(1)')
        self.assertEqual(to_string(ur, fmt),   '1.000(1)')
        self.assertEqual(to_string(ur.x, fmt), '1.000')
        self.assertEqual(to_string(ur.u, fmt), '0.001')

        ur = ureal(1, 0.1)
        fmt = create_format(ur, hash=True, digits=1)
        self.assertEqual('{:#.1}'.format(ur),  '1.0(1)')
        self.assertEqual(to_string(ur, fmt),   '1.0(1)')
        self.assertEqual(to_string(ur.x, fmt), '1.0')
        self.assertEqual(to_string(ur.u, fmt), '0.1')

        ur = ureal(1, 1)
        fmt = create_format(ur, hash='#', digits=1)
        self.assertEqual('{:#.1}'.format(ur),  '1.(1.)')
        self.assertEqual(to_string(ur, fmt),   '1.(1.)')
        self.assertEqual(to_string(ur.x, fmt), '1.')
        self.assertEqual(to_string(ur.u, fmt), '1.')

        ur = ureal(1, 1)
        fmt = create_format(ur, hash=False, digits=1)
        self.assertEqual('{:.1}'.format(ur),   '1(1)')
        self.assertEqual(to_string(ur, fmt),   '1(1)')
        self.assertEqual(to_string(ur.x, fmt), '1')
        self.assertEqual(to_string(ur.u, fmt), '1')

        ur = ureal(1, 0.9876)
        fmt = create_format(ur, hash='#', digits=1)
        self.assertEqual('{:#.1}'.format(ur),  '1.(1.)')
        self.assertEqual(to_string(ur, fmt),   '1.(1.)')
        self.assertEqual(to_string(ur.x, fmt), '1.')
        self.assertEqual(to_string(ur.u, fmt), '1.')

        ur = ureal(1, 0.9876)
        fmt = create_format(ur, hash='#', digits=2)
        self.assertEqual('{:#.2}'.format(ur),  '1.00(99)')
        self.assertEqual(to_string(ur, fmt),   '1.00(99)')
        self.assertEqual(to_string(ur.x, fmt), '1.00')
        self.assertEqual(to_string(ur.u, fmt), '0.99')

        ur = ureal(1, 10)
        fmt = create_format(ur, hash=1, digits=1)
        self.assertEqual('{:#.1}'.format(ur),  '0.(10.)')
        self.assertEqual(to_string(ur, fmt),   '0.(10.)')
        self.assertEqual(to_string(ur.x, fmt), '0.')
        self.assertEqual(to_string(ur.u, fmt), '10.')

        ur = ureal(1, 1000)
        fmt = create_format(ur, hash=True, digits=1)
        self.assertEqual('{:#.1}'.format(ur),  '0.(1000.)')
        self.assertEqual(to_string(ur, fmt),   '0.(1000.)')
        self.assertEqual(to_string(ur.x, fmt), '0.')
        self.assertEqual(to_string(ur.u, fmt), '1000.')

        ur = ureal(12345, 9876)
        fmt = create_format(ur, type='e', hash=True)
        self.assertEqual(to_string(ur, fmt),     '1.23(99)e+04')
        self.assertEqual(to_string(ur.x, fmt),   '1.23e+04')
        self.assertEqual('{:#.2e}'.format(ur.x), '1.23e+04')
        self.assertEqual(to_string(ur.u, fmt),   '9.9e+03')
        self.assertEqual('{:#.1e}'.format(ur.u), '9.9e+03')

        fmt = create_format(ur, hash=True)
        self.assertEqual(to_string(ur, fmt),   '12300.(9900.)')
        self.assertEqual(to_string(ur.x, fmt), '12300.')
        self.assertEqual(to_string(ur.u, fmt), '9900.')

        ur = ureal(10, 10)
        fmt = create_format(ur, type='e', digits=1, hash=True)
        self.assertEqual(to_string(ur, fmt),     '1.(1.)e+01')
        self.assertEqual(to_string(ur.x, fmt),   '1.e+01')
        self.assertEqual('{:#.0e}'.format(ur.x), '1.e+01')
        self.assertEqual(to_string(ur.u, fmt),   '1.e+01')
        self.assertEqual('{:#.0e}'.format(ur.u), '1.e+01')

    def test_grouping_field(self):
        ur = ureal(123456789, 123456)

        fmt = create_format(ur, digits=6, grouping=',')
        self.assertEqual('{:,.6}'.format(ur),  '123,456,789(123,456)')
        self.assertEqual(to_string(ur, fmt),   '123,456,789(123,456)')
        self.assertEqual(to_string(ur.x, fmt), '123,456,789')
        self.assertEqual(to_string(ur.u, fmt), '123,456')

        fmt = create_format(ur, digits=2, grouping=',')
        self.assertEqual('{:,.2}'.format(ur),  '123,460,000(120,000)')
        self.assertEqual(to_string(ur, fmt),   '123,460,000(120,000)')
        self.assertEqual(to_string(ur.x, fmt), '123,460,000')
        self.assertEqual(to_string(ur.u, fmt), '120,000')

        if sys.version_info[:2] >= (3, 6):
            fmt = create_format(ur, digits=1, grouping='_')
            self.assertEqual('{:_.1}'.format(ur),  '123_500_000(100_000)')
            self.assertEqual(to_string(ur, fmt),   '123_500_000(100_000)')
            self.assertEqual(to_string(ur.x, fmt), '123_500_000')
            self.assertEqual(to_string(ur.u, fmt), '100_000')

        uc = ucomplex(123456789-987654321j, (123456, 654321))

        fmt = create_format(uc, digits=3, grouping=',')
        self.assertEqual('{:,.3}'.format(uc),  '(123,457,000(123,000)-987,654,000(654,000)j)')
        self.assertEqual(to_string(uc, fmt),   '(123,457,000(123,000)-987,654,000(654,000)j)')
        self.assertEqual(to_string(uc.x, fmt), '(123,457,000-987,654,000j)')
        self.assertEqual(to_string(uc.u, fmt), '(123,000+654,000j)')

        if sys.version_info[:2] >= (3, 6):
            fmt = create_format(uc, digits=4, grouping='_', sign=' ')
            self.assertEqual('{: _.4}'.format(uc), '( 123_456_800(123_500)-987_654_300(654_300)j)')
            self.assertEqual(to_string(uc, fmt),   '( 123_456_800(123_500)-987_654_300(654_300)j)')
            self.assertEqual(to_string(uc.x, fmt), '( 123_456_800-987_654_300j)')
            self.assertEqual(to_string(uc.u, fmt), '( 123_500+654_300j)')

    @unittest.skipIf(sys.version_info[:2] <= (3, 9), 'not supported for this version of Python')
    def test_zero_field(self):
        ur = ureal(1.342, 0.0041)
        fmt = create_format(ur, digits=1, zero=True, width=15)
        self.assertEqual('{:015.1}'.format(ur), '1.342(4)0000000')
        self.assertEqual(to_string(ur, fmt),    '1.342(4)0000000')
        self.assertEqual(to_string(ur.x, fmt),  '1.3420000000000')
        self.assertEqual(to_string(ur.u, fmt),  '0.0040000000000')

        fmt = create_format(ur, digits=1, zero=False, width=15)
        self.assertEqual('{:15.1}'.format(ur),  '1.342(4)       ')
        self.assertEqual(to_string(ur, fmt),    '1.342(4)       ')
        self.assertEqual(to_string(ur.x, fmt),  '1.342          ')
        self.assertEqual(to_string(ur.u, fmt),  '0.004          ')

        uc = ucomplex(5.4 + 7.2j, (1.4, 1.2))
        fmt = create_format(uc, digits=2, zero='0', width=24, align='>', sign='+')
        self.assertEqual('{:>+024.2}'.format(uc), '000(+5.4(1.4)+7.2(1.2)j)')
        self.assertEqual(to_string(uc, fmt),      '000(+5.4(1.4)+7.2(1.2)j)')
        self.assertEqual(to_string(uc.x, fmt),    '0000000000000(+5.4+7.2j)')
        self.assertEqual(to_string(uc.u, fmt),    '0000000000000(+1.4+1.2j)')

    def test_truncate_dof(self):
        dof = _truncate_dof(inf, None)
        self.assertTrue(math.isinf(dof))

        dof = _truncate_dof(nan, None)
        self.assertTrue(math.isnan(dof))

        dof = _truncate_dof(inf_dof + 1.0, None)
        self.assertTrue(math.isinf(dof))

        dof = _truncate_dof(1e308, None)
        self.assertTrue(math.isinf(dof))

        dof = _truncate_dof(9.386583765876, 0)
        self.assertEqual(dof, 9.0)

        dof = _truncate_dof(9.386583765876, 1)
        self.assertEqual(dof, 9.3)

        dof = _truncate_dof(9.386583765876, 2)
        self.assertEqual(dof, 9.38)

        dof = _truncate_dof(9.386583765876, 3)
        self.assertEqual(dof, 9.386)

        dof = _truncate_dof(9.386583765876, 4)
        self.assertEqual(dof, 9.3865)

        dof = _truncate_dof(7.9999999, 0)
        self.assertEqual(dof, 7.0)

        dof = _truncate_dof(7.9999999, 1)
        self.assertEqual(dof, 7.9)

        dof = _truncate_dof(7.9999999, 5)
        self.assertEqual(dof, 7.99999)

    def test_apply_format(self):
        ur = ureal(1.23456789, 0.004371543, df=8.835223, label='MSL')

        fmt = create_format(ur)
        formatted = apply_format(ur, fmt)
        self.assertEqual(str(formatted),
                         "FormattedUncertainReal("
                         "x=1.2346, "
                         "u=0.0044, "
                         "df=8.0, "
                         "label='MSL')")

        fmt = create_format(ur, digits=1, df_precision=3)
        formatted = apply_format(ur, fmt)
        self.assertEqual(str(formatted),
                         "FormattedUncertainReal("
                         "x=1.235, "
                         "u=0.004, "
                         "df=8.835, "
                         "label='MSL')")

        uc = ucomplex(5.47265413 + 7.27262513j, (1.46184, 0.52141, 0.52141, 1.272654431))

        fmt = create_format(uc)
        formatted = apply_format(uc, fmt)
        self.assertEqual(str(formatted),
                         'FormattedUncertainComplex('
                         'x=(5.5+7.3j), '
                         'u=StandardUncertainty(real=1.2, imag=1.1), '
                         'r=0.382, '
                         'df=inf, '
                         'label=None)')

        fmt = create_format(uc, digits=3, r_precision=1, df_precision=99)
        formatted = apply_format(uc, fmt)
        self.assertEqual(str(formatted),
                         'FormattedUncertainComplex('
                         'x=(5.47+7.27j), '
                         'u=StandardUncertainty(real=1.21, imag=1.13), '
                         'r=0.4, '
                         'df=inf, '
                         'label=None)')

        self.assertRaises(RuntimeError, apply_format, 1.0, fmt)
        self.assertRaises(RuntimeError, apply_format, 1.0j, fmt)

        ur = ureal(1.23456789, 0)
        fmt = create_format(ur)
        formatted = apply_format(ur, fmt)
        self.assertEqual(str(formatted),
                         'FormattedUncertainReal('
                         'x=1.23, '
                         'u=0.0, '
                         'df=inf, '
                         'label=None)')
        fmt = create_format(ur, digits=5)
        formatted = apply_format(ur, fmt)
        self.assertEqual(str(formatted),
                         'FormattedUncertainReal('
                         'x=1.23457, '
                         'u=0.0, '
                         'df=inf, '
                         'label=None)')

        ur = UncertainReal._elementary(12345.6789, inf, 23.8231, 'hello', True)
        fmt = create_format(ur, digits=1)
        formatted = apply_format(ur, fmt)
        self.assertEqual(str(formatted),
                         "FormattedUncertainReal("
                         "x=12345.7, "
                         "u=inf, "
                         "df=23.0, "
                         "label='hello')")

        uc = ucomplex(98.7654321-0.786124j, 0)
        fmt = create_format(uc)
        formatted = apply_format(uc, fmt)
        self.assertEqual(str(formatted),
                         'FormattedUncertainComplex('
                         'x=(98.77-0.79j), '
                         'u=StandardUncertainty(real=0.0, imag=0.0), '
                         'r=0.0, '
                         'df=inf, '
                         'label=None)')
        fmt = create_format(uc, digits=1)
        formatted = apply_format(uc, fmt)
        self.assertEqual(str(formatted),
                         'FormattedUncertainComplex('
                         'x=(98.8-0.8j), '
                         'u=StandardUncertainty(real=0.0, imag=0.0), '
                         'r=0.0, '
                         'df=inf, '
                         'label=None)')

        uc = UncertainComplex._elementary(
            complex(inf, -1.0), inf, nan,
            None, 12.34, 'abc', True
        )
        fmt = create_format(uc)
        formatted = apply_format(uc, fmt)
        self.assertEqual(str(formatted),
                         "FormattedUncertainComplex("
                         "x=(inf-1j), "
                         "u=StandardUncertainty(real=inf, imag=nan), "
                         "r=0.0, "
                         "df=12.0, "
                         "label='abc')")

    def test_latex(self):
        ur = ureal(1.23456789, 0.123456789)
        self.assertEqual('{:.3eL}'.format(ur), r'1.235\left(123\right)')
        self.assertEqual('{:.3EL}'.format(ur * 1e-6), r'1.235\left(123\right)\times10^{-6}')
        self.assertEqual('{:.3EL}'.format(ur * 1e12), r'1.235\left(123\right)\times10^{12}')
        self.assertEqual('{:.3eL}'.format(ur * 1e100), r'1.235\left(123\right)\times10^{100}')
        self.assertEqual('{:.3EL}'.format(ur * 1e-100), r'1.235\left(123\right)\times10^{-100}')

        ur = UncertainReal._elementary(3.14159, nan, inf, None, True)
        self.assertEqual('{:fL}'.format(ur), r'3.14\left(\mathrm{NaN}\right)')
        self.assertEqual('{:.4fL}'.format(ur), r'3.1416\left(\mathrm{NaN}\right)')

        ur = UncertainReal._elementary(nan, 3.142, inf, None, True)
        self.assertEqual('{:L}'.format(ur), r'\mathrm{NaN}\left(3.142\right)')

        ur = UncertainReal._elementary(-inf, inf, inf, None, True)
        self.assertEqual('{:FL}'.format(ur), r'-\infty\left(\infty\right)')

    def test_percent_type(self):
        ur = ureal(0.1548175123, 0.0123456)

        fmt = create_format(ur, digits=1, type='%')
        self.assertEqual('{:.1%}'.format(ur),  '15(1)%')
        self.assertEqual(to_string(ur, fmt),   '15(1)%')
        self.assertEqual(to_string(ur.x, fmt), '15%')
        self.assertEqual(to_string(ur.u, fmt), '1%')

        formatted = apply_format(ur, fmt)
        self.assertEqual(fmt._type, '%')
        self.assertEqual(str(formatted),
                         'FormattedUncertainReal('
                         'x=0.15, '
                         'u=0.01, '
                         'df=inf, '
                         'label=None)')

        fmt = create_format(ur, digits=3, type='%', style='L')
        self.assertEqual('{:.3%L}'.format(ur), r'15.48\left(1.23\right)\%')
        self.assertEqual(to_string(ur, fmt),   r'15.48\left(1.23\right)\%')
        self.assertEqual(to_string(ur.x, fmt), r'15.48\%')
        self.assertEqual(to_string(ur.u, fmt), r'1.23\%')

        formatted = apply_format(ur, fmt)
        self.assertEqual(fmt._type, '%')
        self.assertEqual(str(formatted),
                         'FormattedUncertainReal('
                         'x=0.1548, '
                         'u=0.0123, '
                         'df=inf, '
                         'label=None)')

        ur = ureal(0.1548175123, 0.000123456)

        fmt = create_format(ur, type='%', style='L')
        self.assertEqual('{:%L}'.format(ur),  r'15.482\left(12\right)\%')
        self.assertEqual(to_string(ur, fmt),  r'15.482\left(12\right)\%')
        self.assertEqual(to_string(ur.x, fmt), r'15.482\%')
        self.assertEqual(to_string(ur.u, fmt), r'0.012\%')

        formatted = apply_format(ur, fmt)
        self.assertEqual(fmt._type, '%')
        self.assertEqual(str(formatted),
                         'FormattedUncertainReal('
                         'x=0.15482, '
                         'u=0.00012, '
                         'df=inf, '
                         'label=None)')

        uc = ucomplex(1548175.123+4321.45j, (123.456, 1.32))

        fmt = create_format(uc, digits=1, type='%')
        self.assertEqual('{:.1%}'.format(uc),  '(154817500(12300)+432100(100)j)%')
        self.assertEqual(to_string(uc, fmt),   '(154817500(12300)+432100(100)j)%')
        self.assertEqual(to_string(uc.x, fmt), '(154817500+432100j)%')
        self.assertEqual(to_string(uc.u, fmt), '(12300+100j)%')

        formatted = apply_format(uc, fmt)
        self.assertEqual(fmt._type, '%')
        self.assertEqual(str(formatted),
                         'FormattedUncertainComplex('
                         'x=(1548175+4321j), '
                         'u=StandardUncertainty(real=123.0, imag=1.0), '
                         'r=0.0, '
                         'df=inf, '
                         'label=None)')

        uc = ucomplex(0.062872513, 0.00023164)

        fmt = create_format(uc, digits=3, type='%', style='L')
        self.assertEqual('{:.3%L}'.format(uc),  r'\left(6.2873\left(232\right)+0.0000\left(232\right)j\right)\%')
        self.assertEqual(to_string(uc, fmt),   r'\left(6.2873\left(232\right)+0.0000\left(232\right)j\right)\%')
        self.assertEqual(to_string(uc.x, fmt), r'\left(6.2873+0.0000j\right)\%')
        self.assertEqual(to_string(uc.u, fmt), r'\left(0.0232+0.0232j\right)\%')

        formatted = apply_format(uc, fmt)
        self.assertEqual(fmt._type, '%')
        self.assertEqual(str(formatted),
                         'FormattedUncertainComplex('
                         'x=(0.062873+0j), '
                         'u=StandardUncertainty(real=0.000232, imag=0.000232), '
                         'r=0.0, '
                         'df=inf, '
                         'label=None)')

    def test_type_n_raises(self):
        # can't specify both grouping and n
        self.assertRaises(ValueError, create_format, 1.0, grouping='_', type='n')
        self.assertRaises(ValueError, create_format, 1.0, grouping=',', type='n')

    def test_type_n_swiss(self):
        # this locale is interesting because it can have non-ascii characters
        if sys.platform == 'win32':
            loc = 'German_Switzerland'
        elif sys.platform == 'darwin':
            loc = 'de_CH'
        else:
            loc = 'de_CH.utf8'
        locale.setlocale(locale.LC_NUMERIC, loc)

        ur = ureal(1.23456789, 0.987654321)
        fmt = create_format(ur, type='n')
        if sys.platform == 'darwin':
            self.assertEqual(to_string(ur, fmt),    '1,23(99)')
            self.assertEqual(to_string(ur.x, fmt),  '1,23')
            self.assertEqual('{:.3n}'.format(ur.x), '1,23')
            self.assertEqual(to_string(ur.u, fmt),  '0,99')
            self.assertEqual('{:.2n}'.format(ur.u), '0,99')
        else:
            self.assertEqual(to_string(ur, fmt),    '1.23(99)')
            self.assertEqual(to_string(ur.x, fmt),  '1.23')
            self.assertEqual('{:.3n}'.format(ur.x), '1.23')
            self.assertEqual(to_string(ur.u, fmt),  '0.99')
            self.assertEqual('{:.2n}'.format(ur.u), '0.99')

        ur = ureal(1.2345678987e6, 0.987654321)
        fmt = create_format(ur, type='n', digits=4)
        if sys.platform == 'darwin':
            self.assertEqual(to_string(ur, fmt),     '1234567,8987(9877)')
            self.assertEqual(to_string(ur.x, fmt),   '1234567,8987')
            self.assertEqual('{:.11n}'.format(ur.x), '1234567,8987')
            self.assertEqual(to_string(ur.u, fmt),   '0,9877')
            self.assertEqual('{:.4n}'.format(ur.u),  '0,9877')
        elif sys.version_info[:2] == (2, 7) or (sys.platform == 'win32' and
                                                sys.version_info[:2] == (3, 5)):
            if sys.platform == 'win32':
                self.assertEqual(to_string(ur, fmt),     '1\x92234\x92567.8987(9877)')
                self.assertEqual(to_string(ur.x, fmt),   '1\x92234\x92567.8987')
                self.assertEqual('{:.11n}'.format(ur.x), '1\x92234\x92567.8987')
                self.assertEqual(to_string(ur.u, fmt),   '0.9877')
                self.assertEqual('{:.4n}'.format(ur.u),  '0.9877')
            else:
                self.assertEqual(to_string(ur, fmt),     '1\xe2\x80\x99234\xe2\x80\x99567.8987(9877)')
                self.assertEqual(to_string(ur.x, fmt),   '1\xe2\x80\x99234\xe2\x80\x99567.8987')
                self.assertEqual('{:.11n}'.format(ur.x), '1\xe2\x80\x99234\xe2\x80\x99567.8987')
                self.assertEqual(to_string(ur.u, fmt),   '0.9877')
                self.assertEqual('{:.4n}'.format(ur.u),  '0.9877')
        else:
            self.assertEqual(to_string(ur, fmt),     '1’234’567.8987(9877)')
            self.assertEqual(to_string(ur.x, fmt),   '1’234’567.8987')
            self.assertEqual('{:.11n}'.format(ur.x), '1’234’567.8987')
            self.assertEqual(to_string(ur.u, fmt),   '0.9877')
            self.assertEqual('{:.4n}'.format(ur.u),  '0.9877')

        ur = ureal(12345.6789, 9876.54321)
        fmt = create_format(ur, type='n', digits=8)
        if sys.platform == 'darwin':
            self.assertEqual(to_string(ur, fmt),    '12345,6789(9876,5432)')
            self.assertEqual(to_string(ur.x, fmt),  '12345,6789')
            self.assertEqual('{:.9n}'.format(ur.x), '12345,6789')
            self.assertEqual(to_string(ur.u, fmt),  '9876,5432')
            self.assertEqual('{:.8n}'.format(ur.u), '9876,5432')
        elif sys.version_info[:2] == (2, 7) or (sys.platform == 'win32' and
                                                sys.version_info[:2] == (3, 5)):
            if sys.platform == 'win32':
                self.assertEqual(to_string(ur, fmt),    '12\x92345.6789(9\x92876.5432)')
                self.assertEqual(to_string(ur.x, fmt),  '12\x92345.6789')
                self.assertEqual('{:.9n}'.format(ur.x), '12\x92345.6789')
                self.assertEqual(to_string(ur.u, fmt),  '9\x92876.5432')
                self.assertEqual('{:.8n}'.format(ur.u), '9\x92876.5432')
            else:
                self.assertEqual(to_string(ur, fmt),    '12\xe2\x80\x99345.6789(9\xe2\x80\x99876.5432)')
                self.assertEqual(to_string(ur.x, fmt),  '12\xe2\x80\x99345.6789')
                self.assertEqual('{:.9n}'.format(ur.x), '12\xe2\x80\x99345.6789')
                self.assertEqual(to_string(ur.u, fmt),  '9\xe2\x80\x99876.5432')
                self.assertEqual('{:.8n}'.format(ur.u), '9\xe2\x80\x99876.5432')
        else:
            self.assertEqual(to_string(ur, fmt),    '12’345.6789(9’876.5432)')
            self.assertEqual(to_string(ur.x, fmt),  '12’345.6789')
            self.assertEqual('{:.9n}'.format(ur.x), '12’345.6789')
            self.assertEqual(to_string(ur.u, fmt),  '9’876.5432')
            self.assertEqual('{:.8n}'.format(ur.u), '9’876.5432')

    def test_type_n_german(self):
        if sys.platform == 'win32':
            loc = 'German_Germany'
        elif sys.platform == 'darwin':
            loc = 'de_DE'
        else:
            loc = 'de_DE.utf8'
        locale.setlocale(locale.LC_NUMERIC, loc)

        ur = ureal(1.23456789, 0.987654321)
        fmt = create_format(ur, type='n')
        self.assertEqual(to_string(ur, fmt),    '1,23(99)')
        self.assertEqual(to_string(ur.x, fmt),  '1,23')
        self.assertEqual('{:.3n}'.format(ur.x), '1,23')
        self.assertEqual(to_string(ur.u, fmt),  '0,99')
        self.assertEqual('{:.2n}'.format(ur.u), '0,99')

        ur = ureal(1.2345678987e6, 0.987654321)
        fmt = create_format(ur, type='n', digits=4)
        if sys.platform == 'darwin':
            self.assertEqual(to_string(ur, fmt),     '1234567,8987(9877)')
            self.assertEqual(to_string(ur.x, fmt),   '1234567,8987')
            self.assertEqual('{:.11n}'.format(ur.x), '1234567,8987')
            self.assertEqual(to_string(ur.u, fmt),   '0,9877')
            self.assertEqual('{:.4n}'.format(ur.u),  '0,9877')
        else:
            self.assertEqual(to_string(ur, fmt),     '1.234.567,8987(9877)')
            self.assertEqual(to_string(ur.x, fmt),   '1.234.567,8987')
            self.assertEqual('{:.11n}'.format(ur.x), '1.234.567,8987')
            self.assertEqual(to_string(ur.u, fmt),   '0,9877')
            self.assertEqual('{:.4n}'.format(ur.u),  '0,9877')

        ur = ureal(12345.6789, 9876.54321)
        fmt = create_format(ur, type='n', digits=8)
        if sys.platform == 'darwin':
            self.assertEqual(to_string(ur, fmt),    '12345,6789(9876,5432)')
            self.assertEqual(to_string(ur.x, fmt),  '12345,6789')
            self.assertEqual('{:.9n}'.format(ur.x), '12345,6789')
            self.assertEqual(to_string(ur.u, fmt),  '9876,5432')
            self.assertEqual('{:.8n}'.format(ur.u), '9876,5432')
        else:
            self.assertEqual(to_string(ur, fmt),    '12.345,6789(9.876,5432)')
            self.assertEqual(to_string(ur.x, fmt),  '12.345,6789')
            self.assertEqual('{:.9n}'.format(ur.x), '12.345,6789')
            self.assertEqual(to_string(ur.u, fmt),  '9.876,5432')
            self.assertEqual('{:.8n}'.format(ur.u), '9.876,5432')

        if sys.version_info.major > 2:
            # Python 2.7 does not support the # symbol but Python 3.x does
            ur = ureal(2345, 1234)
            fmt = create_format(ur, type='n', digits=1, hash=True)
            self.assertEqual(to_string(ur, fmt),     '2,(1,)e+03')
            self.assertEqual(to_string(ur.x, fmt),   '2,e+03')
            self.assertEqual('{:#.1n}'.format(ur.x), '2,e+03')
            self.assertEqual(to_string(ur.u, fmt),   '1,e+03')
            self.assertEqual('{:#.1n}'.format(ur.u), '1,e+03')

        ur = ureal(12345, 9876)
        fmt = create_format(ur, type='n', sign=' ', hash=True)
        self.assertEqual(to_string(ur, fmt),      ' 1,23(99)e+04')
        self.assertEqual(to_string(ur.x, fmt),    ' 1,23e+04')
        self.assertEqual(to_string(ur.u, fmt),    ' 9,9e+03')
        if sys.version_info.major > 2:
            # Python 2.7 does not support the # symbol but Python 3.x does
            self.assertEqual('{: #.3n}'.format(ur.x), ' 1,23e+04')
            self.assertEqual('{: #.2n}'.format(ur.u), ' 9,9e+03')

    def test_type_n_india(self):
        # this locale is interesting because it can have a different
        # 'grouping' for the 'thousands_sep' key
        if sys.platform == 'win32':
            loc = 'English_India'
        elif sys.platform == 'darwin':
            loc = 'hi_IN.ISCII-DEV'
        else:
            loc = 'en_IN.utf8'
        locale.setlocale(locale.LC_NUMERIC, loc)

        ur = ureal(1.23456789, 0.987654321)
        fmt = create_format(ur, type='n')
        self.assertEqual(to_string(ur, fmt),    '1.23(99)')
        self.assertEqual(to_string(ur.x, fmt),  '1.23')
        self.assertEqual('{:.3n}'.format(ur.x), '1.23')
        self.assertEqual(to_string(ur.u, fmt),  '0.99')
        self.assertEqual('{:.2n}'.format(ur.u), '0.99')

        ur = ureal(1.2345678987e6, 0.987654321)
        fmt = create_format(ur, type='n', digits=4)
        if sys.platform == 'darwin':
            self.assertEqual(to_string(ur, fmt),     '12,345,67.8987(9877)')
            self.assertEqual(to_string(ur.x, fmt),   '12,345,67.8987')
            self.assertEqual('{:.11n}'.format(ur.x), '12,345,67.8987')
            self.assertEqual(to_string(ur.u, fmt),   '0.9877')
            self.assertEqual('{:.4n}'.format(ur.u),  '0.9877')
        else:
            self.assertEqual(to_string(ur, fmt),     '12,34,567.8987(9877)')
            self.assertEqual(to_string(ur.x, fmt),   '12,34,567.8987')
            self.assertEqual('{:.11n}'.format(ur.x), '12,34,567.8987')
            self.assertEqual(to_string(ur.u, fmt),   '0.9877')
            self.assertEqual('{:.4n}'.format(ur.u),  '0.9877')

        ur = ureal(12345.6789, 9876.54321)
        fmt = create_format(ur, type='n', digits=8)
        if sys.platform == 'darwin':
            self.assertEqual(to_string(ur, fmt),    '123,45.6789(98,76.5432)')
            self.assertEqual(to_string(ur.x, fmt),  '123,45.6789')
            self.assertEqual('{:.9n}'.format(ur.x), '123,45.6789')
            self.assertEqual(to_string(ur.u, fmt),  '98,76.5432')
            self.assertEqual('{:.8n}'.format(ur.u), '98,76.5432')
        else:
            self.assertEqual(to_string(ur, fmt),    '12,345.6789(9,876.5432)')
            self.assertEqual(to_string(ur.x, fmt),  '12,345.6789')
            self.assertEqual('{:.9n}'.format(ur.x), '12,345.6789')
            self.assertEqual(to_string(ur.u, fmt),  '9,876.5432')
            self.assertEqual('{:.8n}'.format(ur.u), '9,876.5432')

    def test_type_n_kiwi(self):
        # make sure the native locale for MSL is good
        if sys.platform == 'win32':
            loc = 'English_New Zealand'
        elif sys.platform == 'darwin':
            loc = 'en_NZ'
        else:
            loc = 'en_NZ.utf8'
        locale.setlocale(locale.LC_NUMERIC, loc)

        ur = ureal(1.23456789, 0.987654321)
        fmt = create_format(ur, type='n')
        self.assertEqual(to_string(ur, fmt),    '1.23(99)')
        self.assertEqual(to_string(ur.x, fmt),  '1.23')
        self.assertEqual('{:.3n}'.format(ur.x), '1.23')
        self.assertEqual(to_string(ur.u, fmt),  '0.99')
        self.assertEqual('{:.2n}'.format(ur.u), '0.99')

        ur = ureal(1.2345678987e6, 0.987654321)
        fmt = create_format(ur, type='n', digits=4)
        self.assertEqual(to_string(ur, fmt),     '1,234,567.8987(9877)')
        self.assertEqual(to_string(ur.x, fmt),   '1,234,567.8987')
        self.assertEqual('{:.11n}'.format(ur.x), '1,234,567.8987')
        self.assertEqual(to_string(ur.u, fmt),   '0.9877')
        self.assertEqual('{:.4n}'.format(ur.u),  '0.9877')

        ur = ureal(12345.6789, 9876.54321)
        fmt = create_format(ur, type='n', digits=8)
        self.assertEqual(to_string(ur, fmt),    '12,345.6789(9,876.5432)')
        self.assertEqual(to_string(ur.x, fmt),  '12,345.6789')
        self.assertEqual('{:.9n}'.format(ur.x), '12,345.6789')
        self.assertEqual(to_string(ur.u, fmt),  '9,876.5432')
        self.assertEqual('{:.8n}'.format(ur.u), '9,876.5432')

        ur = ureal(12345.6789, 9876.54321)
        fmt = create_format(ur, type='n', digits=8, sign='+')
        self.assertEqual(to_string(ur, fmt),     '+12,345.6789(9,876.5432)')
        self.assertEqual(to_string(ur.x, fmt),   '+12,345.6789')
        self.assertEqual('{:+.9n}'.format(ur.x), '+12,345.6789')
        self.assertEqual(to_string(ur.u, fmt),   '+9,876.5432')
        self.assertEqual('{:+.8n}'.format(ur.u), '+9,876.5432')

    def test_type_n_afrikaans(self):
        # this locale is interesting because it can have non-ascii characters
        if sys.platform == 'win32':
            loc = 'English_South Africa'
        elif sys.platform == 'darwin':
            loc = 'af_ZA'
        else:
            loc = 'en_ZA.utf8'
        locale.setlocale(locale.LC_NUMERIC, loc)

        ur = ureal(1.23456789, 0.987654321)
        fmt = create_format(ur, type='n')
        if sys.platform.startswith('linux'):
            self.assertEqual(to_string(ur, fmt),    '1.23(99)')
            self.assertEqual(to_string(ur.x, fmt),  '1.23')
            self.assertEqual('{:.3n}'.format(ur.x), '1.23')
            self.assertEqual(to_string(ur.u, fmt),  '0.99')
            self.assertEqual('{:.2n}'.format(ur.u), '0.99')
        else:
            self.assertEqual(to_string(ur, fmt),    '1,23(99)')
            self.assertEqual(to_string(ur.x, fmt),  '1,23')
            self.assertEqual('{:.3n}'.format(ur.x), '1,23')
            self.assertEqual(to_string(ur.u, fmt),  '0,99')
            self.assertEqual('{:.2n}'.format(ur.u), '0,99')

        ur = ureal(1.2345678987e6, 0.987654321)
        fmt = create_format(ur, type='n', digits=4)
        if sys.platform == 'win32':
            self.assertEqual(to_string(ur, fmt),     '1\xa0234\xa0567,8987(9877)')
            self.assertEqual(to_string(ur.x, fmt),   '1\xa0234\xa0567,8987')
            self.assertEqual('{:.11n}'.format(ur.x), '1\xa0234\xa0567,8987')
            self.assertEqual(to_string(ur.u, fmt),   '0,9877')
            self.assertEqual('{:.4n}'.format(ur.u),  '0,9877')
        elif sys.platform == 'darwin':
            self.assertEqual(to_string(ur, fmt),     '1.234.567,8987(9877)')
            self.assertEqual(to_string(ur.x, fmt),   '1.234.567,8987')
            self.assertEqual('{:.11n}'.format(ur.x), '1.234.567,8987')
            self.assertEqual(to_string(ur.u, fmt),   '0,9877')
            self.assertEqual('{:.4n}'.format(ur.u),  '0,9877')
        else:
            self.assertEqual(to_string(ur, fmt),     '1,234,567.8987(9877)')
            self.assertEqual(to_string(ur.x, fmt),   '1,234,567.8987')
            self.assertEqual('{:.11n}'.format(ur.x), '1,234,567.8987')
            self.assertEqual(to_string(ur.u, fmt),   '0.9877')
            self.assertEqual('{:.4n}'.format(ur.u),  '0.9877')

        ur = ureal(12345.6789, 9876.54321)
        fmt = create_format(ur, type='n', digits=8)
        if sys.platform == 'win32':
            self.assertEqual(to_string(ur, fmt),    '12\xa0345,6789(9\xa0876,5432)')
            self.assertEqual(to_string(ur.x, fmt),  '12\xa0345,6789')
            self.assertEqual('{:.9n}'.format(ur.x), '12\xa0345,6789')
            self.assertEqual(to_string(ur.u, fmt),  '9\xa0876,5432')
            self.assertEqual('{:.8n}'.format(ur.u), '9\xa0876,5432')
        elif sys.platform == 'darwin':
            self.assertEqual(to_string(ur, fmt),    '12.345,6789(9.876,5432)')
            self.assertEqual(to_string(ur.x, fmt),  '12.345,6789')
            self.assertEqual('{:.9n}'.format(ur.x), '12.345,6789')
            self.assertEqual(to_string(ur.u, fmt),  '9.876,5432')
            self.assertEqual('{:.8n}'.format(ur.u), '9.876,5432')
        else:
            self.assertEqual(to_string(ur, fmt),    '12,345.6789(9,876.5432)')
            self.assertEqual(to_string(ur.x, fmt),  '12,345.6789')
            self.assertEqual('{:.9n}'.format(ur.x), '12,345.6789')
            self.assertEqual(to_string(ur.u, fmt),  '9,876.5432')
            self.assertEqual('{:.8n}'.format(ur.u), '9,876.5432')

    def test_to_string_raises(self):
        ur = ureal(1, 1)
        fmt = create_format(ur)
        for obj in [None, '1.0', b'0', {}, [], (), object()]:
            self.assertRaises(RuntimeError, to_string, obj, fmt)
