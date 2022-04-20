import unittest

from GTC import ureal, ucomplex, inf_dof, inf, nan
from GTC import formatting
from GTC.lib import UncertainReal, UncertainComplex


class TestFormatting(unittest.TestCase):

    def test_parse_raises(self):
        # want the exception types raised by formatting.parse to
        # match what the builtin format(float, format_spec) would raise
        def check(exception, format_spec):
            self.assertRaises(exception, format, 1.0, format_spec)
            self.assertRaises(exception, formatting.parse, format_spec)

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
        check(ValueError, 'A')  # invalid <type> or <fill> without <align>
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
        check(ValueError, 'BR')  # two modes specified
        check(ValueError, 'LP')  # two styles specified

    def test_parse(self):
        # also call the builtin format(float, format_spec) to verify
        # that the formatting.parse function is okay
        def parse(format_spec, check=True):
            if check:  # must ignore for GTC-specific fields
                format(1.0, format_spec)
            return formatting.parse(format_spec)

        def expect(**kwargs):
            out = {
                'fill': None, 'align': None, 'sign': None, 'hash': None,
                'zero': None, 'width': None, 'grouping': None,
                'precision': None, 'type': None, 'df_decimals': None,
                'mode': None, 'style': None
            }
            out.update(**kwargs)
            return out

        # check the builtin-supported fields
        self.assertEqual(parse('G'),
                         expect(type='G'))
        self.assertEqual(parse('='),
                         expect(align='='))
        self.assertEqual(parse(' ='),
                         expect(fill=' ', align='='))
        self.assertEqual(parse('<<'),
                         expect(fill='<', align='<'))
        self.assertEqual(parse(' 10.1'),
                         expect(sign=' ', width=10, precision=1))
        self.assertEqual(parse('0'),
                         expect(zero='0'))
        self.assertEqual(parse('0.0'),
                         expect(zero='0', precision=0))
        self.assertEqual(parse('02'),
                         expect(zero='0', width=2))
        self.assertEqual(parse('02.0'),
                         expect(zero='0', width=2, precision=0))
        self.assertEqual(parse('.10'),
                         expect(precision=10))
        self.assertEqual(parse('07.2f'),
                         expect(zero='0', width=7, precision=2, type='f'))
        self.assertEqual(parse('*<-06,.4E'),
                         expect(fill='*', align='<', sign='-', zero='0',
                                width=6, grouping=',', precision=4, type='E'))

        # additional GTC-specific fields
        self.assertEqual(parse('B', False),
                         expect(mode='B'))
        self.assertEqual(parse('S', False),
                         expect(style='S'))
        self.assertEqual(parse('GB', False),
                         expect(type='G', mode='B'))
        self.assertEqual(parse('GBL', False),
                         expect(type='G', mode='B', style='L'))
        self.assertEqual(parse('.2P', False),
                         expect(precision=2, style='P'))
        self.assertEqual(parse('9R', False),
                         expect(width=9, mode='R'))
        self.assertEqual(parse('.7.5', False),
                         expect(precision=7, df_decimals=5))
        self.assertEqual(parse('e.11', False),
                         expect(type='e', df_decimals=11))
        self.assertEqual(parse('.2f.0', False),
                         expect(precision=2, type='f', df_decimals=0))
        self.assertEqual(parse('.2f.3R', False),
                         expect(precision=2, type='f', df_decimals=3,
                                mode='R'))
        self.assertEqual(parse(' ^16.4fL', False),
                         expect(fill=' ', align='^', width=16,
                                precision=4, type='f', style='L'))
        self.assertEqual(parse('^^03S', False),
                         expect(fill='^', align='^', zero='0', width=3,
                                style='S'))
        self.assertEqual(parse('^^03BS', False),
                         expect(fill='^', align='^', zero='0', width=3,
                                mode='B', style='S'))
        self.assertEqual(parse('^^03gBS', False),
                         expect(fill='^', align='^',zero='0', width=3,
                                type='g', mode='B', style='S'))
        self.assertEqual(parse('^^03gB', False),
                         expect(fill='^', align='^', zero='0', width=3,
                                type='g', mode='B'))
        self.assertEqual(parse('*> #011,.2g.8S', False),
                         expect(fill='*', align='>', sign=' ', hash='#',
                                zero='0', width=11, grouping=',', precision=2,
                                type='g', df_decimals=8, style='S'))

    def test_Format(self):
        f = formatting.Format()
        self.assertEqual(f.format_spec, '')
        self.assertEqual(repr(f), 'Format{}')
        self.assertEqual(str(f), 'Format{}')

        f = formatting.Format(fill='*', align='>', sign=' ', hash='#', zero='0',
                              width=20, grouping=',', precision=3, type='g',
                              df_decimals=0, mode='B', style='L')
        self.assertEqual(f.format_spec, '*> #020,.3g')
        self.assertEqual(repr(f), 'Format{*> #020,.3g.0BL}')
        self.assertEqual(str(f), 'Format{*> #020,.3g.0BL}')

        f = formatting.Format(width=10, type='f', mode='B')
        self.assertEqual(f.format_spec, '10f')
        self.assertEqual(repr(f), 'Format{10fB}')
        self.assertEqual(str(f), 'Format{10fB}')

        f = formatting.Format()
        number = -9.3+123.456789j
        self.assertEqual(f.format(number), '{}'.format(number))
        number = 123.456789
        self.assertEqual(f.format(number), '{}'.format(number))
        self.assertEqual(f.format(number, sign=' ', precision=4, type='f'),
                         '{: .4f}'.format(number))

        f = formatting.Format(precision=4, sign='', width=20)
        self.assertEqual(f.format_spec, '20.4')
        number = 123.456789
        self.assertEqual(f.format(number), '{:20.4}'.format(number))
        number = -9.3+123.456789j
        self.assertEqual(f.format(number), '{:20.4}'.format(number))

        f = formatting.Format(precision=4, sign='+', type='f')
        self.assertEqual(f.format_spec, '+.4f')
        number = 123.456789
        self.assertEqual(f.format(number), '{:+.4f}'.format(number))
        number = -9.3+123.456789j
        self.assertEqual(f.format(number), '{:+.4f}'.format(number))

        f = formatting.Format(fill='*', align='^', width=20,
                              grouping=',', precision=0, type='f')
        self.assertEqual(f.format_spec, '*^20,.0f')
        number = 123456789
        self.assertEqual(f.format(number), '{:*^20,.0f}'.format(number))

    def test_nan_or_inf(self):
        nan_or_inf = formatting._nan_or_inf
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
        # TODO was expecting label='MSL' not label=MSL
        check(ucomplex(1.23456789 + 0.12345j, 0.1, df=8, label='MSL'),
              'ucomplex((1.23456789+0.12345j), u=[0.1,0.1], r=0.0, df=8.0, label=MSL)')

    def test_str_ureal(self):
        def check(ur, expected):
            # different ways to get the same result
            self.assertEqual(str(ur), expected)
            self.assertEqual('{}'.format(ur), expected)
            self.assertEqual('{!s}'.format(ur), expected)
            self.assertEqual('{: .2f.0B}'.format(ur), expected)

        check(ureal(1.23456789, 1000), ' 0(1000)')
        check(ureal(1.23456789, 100), ' 0(100)')
        check(ureal(1.23456789, 10), ' 1(10)')
        check(ureal(1.23456789, 1), ' 1.2(1.0)')
        check(ureal(1.23456789, 0), ' 1.234568')
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
            self.assertEqual('{:+.2f.0B}'.format(uc), expected)

        check(ucomplex(1.23456789 + 9.87654321j, 1000),
              '(+0(1000)+0(1000)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 100),
              '(+0(100)+10(100)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 10),
              '(+1(10)+10(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 1),
              '(+1.2(1.0)+9.9(1.0)j)')

        # TODO Historically, the following would have been returned
        #      '(+1.234568(0)+9.876543(0)j)'
        #      For the case of print(ureal(1.23456789, 0)) the result is
        #      ' 1.234568'
        #      i.e., there is no '(0)' at the end.
        #      What do we want if the uncertainty=0?
        check(ucomplex(1.23456789 + 9.87654321j, 0),
              '(+1.234568+9.876543j)')

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
        create = formatting.create
        convert = formatting.convert

        ur = UncertainReal._elementary(inf, inf, inf, None, True)
        self.assertEqual(str(ur), ' inf(inf)')
        self.assertEqual('{}'.format(ur), ' inf(inf)')
        self.assertEqual('{!s}'.format(ur), ' inf(inf)')
        for t in ['f', 'g', 'e']:
            fmt = create(ur, type=t)
            self.assertEqual(convert(ur, fmt), 'inf(inf)')
            self.assertEqual(convert(ur.x, fmt), 'inf')
            self.assertEqual(convert(ur.u, fmt), 'inf')
        for t in ['F', 'G', 'E']:
            fmt = create(ur, type=t)
            self.assertEqual(convert(ur, fmt), 'INF(INF)')
            self.assertEqual(convert(ur.x, fmt), 'INF')
            self.assertEqual(convert(ur.u, fmt), 'INF')

        ur = UncertainReal._elementary(inf, nan, inf, None, True)
        self.assertEqual(str(ur), ' inf(nan)')
        self.assertEqual('{}'.format(ur), ' inf(nan)')
        self.assertEqual('{!s}'.format(ur), ' inf(nan)')
        for t in ['f', 'g', 'e']:
            fmt = create(ur, type=t)
            self.assertEqual(convert(ur, fmt), 'inf(nan)')
            self.assertEqual(convert(ur.x, fmt), 'inf')
            self.assertEqual(convert(ur.u, fmt), 'nan')
        for t in ['F', 'G', 'E']:
            fmt = create(ur, type=t)
            self.assertEqual(convert(ur, fmt), 'INF(NAN)')
            self.assertEqual(convert(ur.x, fmt), 'INF')
            self.assertEqual(convert(ur.u, fmt), 'NAN')

        ur = UncertainReal._elementary(-inf, nan, inf, None, True)
        self.assertEqual(str(ur), '-inf(nan)')
        self.assertEqual('{}'.format(ur), '-inf(nan)')
        self.assertEqual('{!s}'.format(ur), '-inf(nan)')
        for t in ['f', 'g', 'e']:
            fmt = create(ur, type=t)
            self.assertEqual(convert(ur, fmt), '-inf(nan)')
            self.assertEqual(convert(ur.x, fmt), '-inf')
            self.assertEqual(convert(ur.u, fmt), 'nan')
        for t in ['F', 'G', 'E']:
            fmt = create(ur, type=t)
            self.assertEqual(convert(ur, fmt), '-INF(NAN)')
            self.assertEqual(convert(ur.x, fmt), '-INF')
            self.assertEqual(convert(ur.u, fmt), 'NAN')

        ur = UncertainReal._elementary(nan, inf, inf, None, True)
        self.assertEqual(str(ur), ' nan(inf)')
        self.assertEqual('{}'.format(ur), ' nan(inf)')
        self.assertEqual('{!s}'.format(ur), ' nan(inf)')
        for t in ['f', 'g', 'e']:
            fmt = create(ur, type=t)
            self.assertEqual(convert(ur, fmt), 'nan(inf)')
            self.assertEqual(convert(ur.x, fmt), 'nan')
            self.assertEqual(convert(ur.u, fmt), 'inf')
        for t in ['F', 'G', 'E']:
            fmt = create(ur, type=t)
            self.assertEqual(convert(ur, fmt), 'NAN(INF)')
            self.assertEqual(convert(ur.x, fmt), 'NAN')
            self.assertEqual(convert(ur.u, fmt), 'INF')

        ur = UncertainReal._elementary(nan, nan, inf, None, True)
        self.assertEqual(str(ur), ' nan(nan)')
        self.assertEqual('{}'.format(ur), ' nan(nan)')
        self.assertEqual('{!s}'.format(ur), ' nan(nan)')
        for t in ['f', 'g', 'e']:
            fmt = create(ur, type=t)
            self.assertEqual(convert(ur, fmt), 'nan(nan)')
            self.assertEqual(convert(ur.x, fmt), 'nan')
            self.assertEqual(convert(ur.u, fmt), 'nan')
        for t in ['F', 'G', 'E']:
            fmt = create(ur, type=t)
            self.assertEqual(convert(ur, fmt), 'NAN(NAN)')
            self.assertEqual(convert(ur.x, fmt), 'NAN')
            self.assertEqual(convert(ur.u, fmt), 'NAN')

    def test_bracket_nan_inf_ucomplex(self):
        create = formatting.create
        convert = formatting.convert

        uc = UncertainComplex._elementary(
            complex(inf, inf), inf, inf,
            None, inf, None, True
        )

        self.assertEqual(str(uc), '(+inf(inf)+inf(inf)j)')
        self.assertEqual('{}'.format(uc), '(+inf(inf)+inf(inf)j)')
        self.assertEqual('{!s}'.format(uc), '(+inf(inf)+inf(inf)j)')
        for t in ['f', 'g', 'e']:
            fmt = create(uc, type=t, sign='+')
            self.assertEqual(convert(uc, fmt), '(+inf(inf)+inf(inf)j)')
            self.assertEqual(convert(uc.x, fmt), '+inf+infj')
            self.assertEqual(convert(uc.u, fmt), '+inf+infj')
            fmt = create(uc, type=t)
            self.assertEqual(convert(uc, fmt), '(inf(inf)+inf(inf)j)')
            self.assertEqual(convert(uc.x, fmt), 'inf+infj')
            self.assertEqual(convert(uc.u, fmt), 'inf+infj')
        for t in ['F', 'G', 'E']:
            fmt = create(uc, type=t, sign='+')
            self.assertEqual(convert(uc, fmt), '(+INF(INF)+INF(INF)j)')
            self.assertEqual(convert(uc.x, fmt), '+INF+INFj')
            self.assertEqual(convert(uc.u, fmt), '+INF+INFj')
            fmt = create(uc, type=t)
            self.assertEqual(convert(uc, fmt), '(INF(INF)+INF(INF)j)')
            self.assertEqual(convert(uc.x, fmt), 'INF+INFj')
            self.assertEqual(convert(uc.u, fmt), 'INF+INFj')

        uc = UncertainComplex._elementary(
            complex(nan, -inf), inf, nan,
            None, inf, None, True
        )

        self.assertEqual(str(uc), '(+nan(inf)-inf(nan)j)')
        self.assertEqual('{}'.format(uc), '(+nan(inf)-inf(nan)j)')
        self.assertEqual('{!s}'.format(uc), '(+nan(inf)-inf(nan)j)')
        for t in ['f', 'g', 'e']:
            fmt = create(uc, type=t, sign='+')
            self.assertEqual(convert(uc, fmt), '(+nan(inf)-inf(nan)j)')
            self.assertEqual(convert(uc.x, fmt), '+nan-infj')
            self.assertEqual(convert(uc.u, fmt), '+inf+nanj')
            fmt = create(uc, type=t)
            self.assertEqual(convert(uc, fmt), '(nan(inf)-inf(nan)j)')
            self.assertEqual(convert(uc.x, fmt), 'nan-infj')
            self.assertEqual(convert(uc.u, fmt), 'inf+nanj')
        for t in ['F', 'G', 'E']:
            fmt = create(uc, type=t, sign='+')
            self.assertEqual(convert(uc, fmt), '(+NAN(INF)-INF(NAN)j)')
            self.assertEqual(convert(uc.x, fmt), '+NAN-INFj')
            self.assertEqual(convert(uc.u, fmt), '+INF+NANj')
            fmt = create(uc, type=t)
            self.assertEqual(convert(uc, fmt), '(NAN(INF)-INF(NAN)j)')
            self.assertEqual(convert(uc.x, fmt), 'NAN-INFj')
            self.assertEqual(convert(uc.u, fmt), 'INF+NANj')

    def test_bracket_type_f_ureal(self):
        create = formatting.create
        convert = formatting.convert

        ur = ureal(1.23456789, 0.0123456789)

        fmt = create(ur, precision=1)
        self.assertEqual(convert(ur, fmt),     '1.23(1)')
        self.assertEqual('{:.1}'.format(ur),   '1.23(1)')  # f and B are defaults
        self.assertEqual('{:.1f}'.format(ur),  '1.23(1)')
        self.assertEqual('{:.1fB}'.format(ur), '1.23(1)')
        self.assertEqual(convert(ur.x, fmt),   '1.23')
        self.assertEqual(convert(ur.u, fmt),   '0.01')

        fmt = create(ur, precision=2)
        self.assertEqual(convert(ur, fmt),    '1.235(12)')
        self.assertEqual('{:.2f}'.format(ur), '1.235(12)')
        self.assertEqual(convert(ur.x, fmt),  '1.235')
        self.assertEqual(convert(ur.u, fmt),  '0.012')

        fmt = create(ur, precision=3)
        self.assertEqual(convert(ur, fmt),    '1.2346(123)')
        self.assertEqual('{:.3f}'.format(ur), '1.2346(123)')
        self.assertEqual(convert(ur.x, fmt),  '1.2346')
        self.assertEqual(convert(ur.u, fmt),  '0.0123')

        fmt = create(ur, precision=9)
        self.assertEqual(convert(ur, fmt),    '1.2345678900(123456789)')
        self.assertEqual('{:.9f}'.format(ur), '1.2345678900(123456789)')
        self.assertEqual(convert(ur.x, fmt),  '1.2345678900')
        self.assertEqual(convert(ur.u, fmt),  '0.0123456789')

        fmt = create(ur, precision=14)
        self.assertEqual(convert(ur, fmt),     '1.234567890000000(12345678900000)')
        self.assertEqual('{:.14f}'.format(ur), '1.234567890000000(12345678900000)')
        self.assertEqual(convert(ur.x, fmt),   '1.234567890000000')
        self.assertEqual(convert(ur.u, fmt),   '0.012345678900000')

        u = ur * (10 ** -20)
        fmt = create(u, type='f', precision=4, width=39, fill=' ', align='>')
        self.assertEqual(convert(u, fmt),   '      0.0000000000000000000123457(1235)')
        self.assertEqual(convert(u.x, fmt), '            0.0000000000000000000123457')
        self.assertEqual(convert(u.u, fmt), '            0.0000000000000000000001235')

        u = ur * (10 ** -19)
        fmt = create(u, type='f', precision=4, width=39, fill=' ', align='>')
        self.assertEqual(convert(u, fmt),   '       0.000000000000000000123457(1235)')
        self.assertEqual(convert(u.x, fmt), '             0.000000000000000000123457')
        self.assertEqual(convert(u.u, fmt), '             0.000000000000000000001235')

        u = ur * (10 ** -18)
        fmt = create(u, type='f', precision=4, width=39, fill=' ', align='<')
        self.assertEqual(convert(u, fmt),   '0.00000000000000000123457(1235)        ')
        self.assertEqual(convert(u.x, fmt), '0.00000000000000000123457              ')
        self.assertEqual(convert(u.u, fmt), '0.00000000000000000001235              ')

        u = ur * (10 ** -12)
        fmt = create(u, type='f', precision=4, width=39, fill='-', align='^')
        self.assertEqual(convert(u, fmt),   '-------0.00000000000123457(1235)-------')
        self.assertEqual(convert(u.x, fmt), '----------0.00000000000123457----------')
        self.assertEqual(convert(u.u, fmt), '----------0.00000000000001235----------')

        u = ur * (10 ** -6)
        fmt = create(u, type='f', precision=4, width=19)
        self.assertEqual(convert(u, fmt),   '0.00000123457(1235)')
        self.assertEqual(convert(u.x, fmt), '      0.00000123457')
        self.assertEqual(convert(u.u, fmt), '      0.00000001235')

        u = ur * (10 ** 0)
        fmt = create(u, type='f', precision=4, width=15, fill=' ', align='>')
        self.assertEqual(convert(u, fmt),   '  1.23457(1235)')
        self.assertEqual(convert(u.x, fmt), '        1.23457')
        self.assertEqual(convert(u.u, fmt), '        0.01235')

        u = ur * (10 ** 1)
        fmt = create(u, type='f', precision=4, width=15, fill=' ', align='>')
        self.assertEqual(convert(u, fmt),   '  12.3457(1235)')
        self.assertEqual(convert(u.x, fmt), '        12.3457')
        self.assertEqual(convert(u.u, fmt), '         0.1235')

        u = ur * (10 ** 2)
        fmt = create(u, type='f', precision=4, width=15, fill=' ', align='>')
        self.assertEqual(convert(u, fmt),   ' 123.457(1.235)')
        self.assertEqual(convert(u.x, fmt), '        123.457')
        self.assertEqual(convert(u.u, fmt), '          1.235')

        u = ur * (10 ** 3)
        fmt = create(u, type='f', precision=4, width=15, fill=' ', align='>')
        self.assertEqual(convert(u, fmt),   ' 1234.57(12.35)')
        self.assertEqual(convert(u.x, fmt), '        1234.57')
        self.assertEqual(convert(u.u, fmt), '          12.35')

        u = ur * (10 ** 4)
        fmt = create(u, type='f', precision=4, width=15, fill=' ', align='>')
        self.assertEqual(convert(u, fmt),   ' 12345.7(123.5)')
        self.assertEqual(convert(u.x, fmt), '        12345.7')
        self.assertEqual(convert(u.u, fmt), '          123.5')

        u = ur * (10 ** 5)
        fmt = create(u, type='f', precision=4, width=15, fill=' ', align='>')
        self.assertEqual(convert(u, fmt),   '   123457(1235)')
        self.assertEqual(convert(u.x, fmt), '         123457')
        self.assertEqual(convert(u.u, fmt), '           1235')

        u = ur * (10 ** 6)
        fmt = create(u, type='f', precision=4, sign='+', width=20, fill=' ', align='>')
        self.assertEqual(convert(u, fmt),   '     +1234570(12350)')
        self.assertEqual(convert(u.x, fmt), '            +1234568')
        self.assertEqual(convert(u.u, fmt), '              +12346')

        u = ur * (10 ** 7)
        fmt = create(u, type='f', precision=4, width=16, fill=' ', align='>')
        self.assertEqual(convert(u, fmt),   '12345700(123500)')
        self.assertEqual(convert(u.x, fmt), '        12345679')
        self.assertEqual(convert(u.u, fmt), '          123457')

        u = ur * (10 ** 8)
        fmt = create(u, type='f', precision=4)
        self.assertEqual(convert(u, fmt),   '123457000(1235000)')
        self.assertEqual(convert(u.x, fmt), '123456789')
        self.assertEqual(convert(u.u, fmt),   '1234568')

        u = ur * (10 ** 18)
        fmt = create(u, type='f', precision=4)
        self.assertEqual(convert(u, fmt),   '1234570000000000000(12350000000000000)')
        self.assertEqual(convert(u.x, fmt), '1234567890000000000')
        self.assertEqual(convert(u.u, fmt),   '12345678900000000')

        ur = ureal(1.23456789, 1234.56789)
        fmt = create(ur, sign=' ', precision=2, type='f', mode='B')
        self.assertEqual(convert(ur, fmt), ' 0(1200)')
        self.assertEqual(convert(ur.x, fmt),     ' 1')
        self.assertEqual(convert(ur.u, fmt),  ' 1235')

        ur = ureal(1.23456789, 123.456789)
        fmt = create(ur, sign=' ', precision=2, type='f', mode='B')
        self.assertEqual(convert(ur, fmt),     ' 0(120)')
        self.assertEqual(convert(ur.x, fmt),   ' 1')
        self.assertEqual(convert(ur.u, fmt), ' 123')

        ur = ureal(1.23456789, 12.3456789)
        fmt = create(ur, sign=' ', precision=2, type='f', mode='B')
        self.assertEqual(convert(ur, fmt),    ' 1(12)')
        self.assertEqual(convert(ur.x, fmt),  ' 1')
        self.assertEqual(convert(ur.u, fmt), ' 12')

        ur = ureal(1.23456789, 1.23456789)
        fmt = create(ur, sign=' ', precision=2, type='f', mode='B')
        self.assertEqual(convert(ur, fmt),   ' 1.2(1.2)')
        self.assertEqual(convert(ur.x, fmt), ' 1.2')
        self.assertEqual(convert(ur.u, fmt), ' 1.2')

        ur = ureal(1.23456789, 0.123456789)
        fmt = create(ur, sign=' ', precision=2, type='f', mode='B')
        self.assertEqual(convert(ur, fmt),   ' 1.23(12)')
        self.assertEqual(convert(ur.x, fmt), ' 1.23')
        self.assertEqual(convert(ur.u, fmt), ' 0.12')

        ur = ureal(1.23456789, 0.0123456789)
        fmt = create(ur, sign=' ', precision=2, type='f', mode='B')
        self.assertEqual(convert(ur, fmt),   ' 1.235(12)')
        self.assertEqual(convert(ur.x, fmt), ' 1.235')
        self.assertEqual(convert(ur.u, fmt), ' 0.012')

        ur = ureal(1.23456789, 0.00123456789)
        fmt = create(ur, sign=' ', precision=2, type='f', mode='B')
        self.assertEqual(convert(ur, fmt),   ' 1.2346(12)')
        self.assertEqual(convert(ur.x, fmt), ' 1.2346')
        self.assertEqual(convert(ur.u, fmt), ' 0.0012')

        ur = ureal(1.23456789, 0.000123456789)
        fmt = create(ur, sign=' ', precision=2, type='f', mode='B')
        self.assertEqual(convert(ur, fmt),   ' 1.23457(12)')
        self.assertEqual(convert(ur.x, fmt), ' 1.23457')
        self.assertEqual(convert(ur.u, fmt), ' 0.00012')

        ur = ureal(1.23456789, 0.000000123456789)
        fmt = create(ur, sign=' ', precision=2, type='f', mode='B')
        self.assertEqual(convert(ur, fmt),   ' 1.23456789(12)')
        self.assertEqual(convert(ur.x, fmt), ' 1.23456789')
        self.assertEqual(convert(ur.u, fmt), ' 0.00000012')

        ur = ureal(1.23456789, 0.000000000123456789)
        fmt = create(ur, sign=' ', precision=2, type='f', mode='B')
        self.assertEqual(convert(ur, fmt),   ' 1.23456789000(12)')
        self.assertEqual(convert(ur.x, fmt), ' 1.23456789000')
        self.assertEqual(convert(ur.u, fmt), ' 0.00000000012')

        ur = ureal(1.23456789e-4, 0.000000000123456789)
        fmt = create(ur, sign=' ', precision=2, type='f', mode='B')
        self.assertEqual(convert(ur, fmt),   ' 0.00012345679(12)')
        self.assertEqual(convert(ur.x, fmt), ' 0.00012345679')
        self.assertEqual(convert(ur.u, fmt), ' 0.00000000012')

        ur = ureal(1.23456789e4, 0.000000123456789)
        fmt = create(ur, sign=' ', precision=2, type='f', mode='B')
        self.assertEqual(convert(ur, fmt),   ' 12345.67890000(12)')
        self.assertEqual(convert(ur.x, fmt), ' 12345.67890000')
        self.assertEqual(convert(ur.u, fmt),     ' 0.00000012')

        ur = ureal(1.23456789, 0.0123456789)
        fmt = create(ur.x)  # use the value and the default kwargs
        self.assertEqual(convert(ur, fmt),   '1.2(0.0)')
        self.assertEqual(convert(ur.x, fmt), '1.2')
        self.assertEqual(convert(ur.u, fmt), '0.0')

        ur = ureal(1.23456789, 0.0123456789)
        fmt = create(ur.u)  # use the uncertainty and the default kwargs
        self.assertEqual(convert(ur, fmt),   '1.235(12)')
        self.assertEqual(convert(ur.x, fmt), '1.235')
        self.assertEqual(convert(ur.u, fmt), '0.012')

        ur = ureal(1.23456789, 0)
        fmt = create(ur, type='f')  # precision defaults to 6
        self.assertEqual(convert(ur, fmt),   '1.234568')
        self.assertEqual(convert(ur.x, fmt), '1.234568')
        self.assertEqual(convert(ur.u, fmt), '0.000000')

        ur = ureal(1.23456789, 0)
        fmt = create(ur, type='g')  # precision defaults to 6
        self.assertEqual(convert(ur, fmt),   '1.23457')
        self.assertEqual(convert(ur.x, fmt), '1.23457')
        self.assertEqual(convert(ur.u, fmt), '0')

        ur = ureal(1.23456789, 0)
        fmt = create(ur, type='E')  # precision defaults to 6
        self.assertEqual(convert(ur, fmt),   '1.234568E+00')
        self.assertEqual(convert(ur.x, fmt), '1.234568E+00')
        self.assertEqual(convert(ur.u, fmt), '0.000000E+00')

        # TODO a precision of 6 is forced if u=0, do we still want this?
        ur = ureal(1.23456789, 0)
        fmt = create(ur, type='E', precision=1)
        self.assertEqual(convert(ur, fmt),   '1.234568E+00')
        self.assertEqual(convert(ur.x, fmt), '1.234568E+00')
        self.assertEqual(convert(ur.u, fmt), '0.000000E+00')

    def test_bracket_type_f_ucomplex(self):
        create = formatting.create
        convert = formatting.convert

        uc = ucomplex(1.23456789e6 + 9.87654321j, [1234.56789, 0.342567])
        fmt = create(uc, sign='+')
        self.assertEqual(convert(uc, fmt),  '(+1234567.89(123457)+9.88(34)j)')
        self.assertEqual(convert(uc.x, fmt), '+1234567.89+9.88j')
        self.assertEqual(convert(uc.u, fmt), '+1234.57+0.34j')

        uc = ucomplex(1.23456789e6 + 9.87654321j, [0.342567, 13.56789])
        fmt = create(uc, sign='+')
        self.assertEqual(convert(uc, fmt),   '(+1234567.89(34)+9.88(1357)j)')
        self.assertEqual(convert(uc.x, fmt), '+1234567.89+9.88j')
        self.assertEqual(convert(uc.u, fmt), '+0.34+13.57j')

        uc = ucomplex(12.3456789 + 0.87654321j, [0.342567, 0.00056789])

        fmt = create(uc, sign='+', precision=4)
        self.assertEqual(convert(uc, fmt),   '(+12.3456789(3425670)+0.8765432(5679)j)')
        self.assertEqual(convert(uc.x, fmt), '+12.3456789+0.8765432j')
        self.assertEqual(convert(uc.u, fmt), '+0.3425670+0.0005679j')

        fmt = create(uc, sign=' ', fill=' ', align='>', width=40)
        self.assertEqual(convert(uc, fmt),   '         ( 12.34568(34257)+0.87654(57)j)')
        self.assertEqual(convert(uc.x, fmt), '                       12.34568+0.87654j')
        self.assertEqual(convert(uc.u, fmt), '                        0.34257+0.00057j')

        fmt = create(uc, fill=' ', align='>', width=40)
        self.assertEqual(convert(uc, fmt),   '          (12.34568(34257)+0.87654(57)j)')
        self.assertEqual(convert(uc.x, fmt), '                       12.34568+0.87654j')
        self.assertEqual(convert(uc.u, fmt), '                        0.34257+0.00057j')

        uc = ucomplex(12.3456789 - 0.87654321j, [0.342567, 0.00056789])
        fmt = create(uc, sign='+', fill=' ', align='>', width=40)
        self.assertEqual(convert(uc, fmt),   '         (+12.34568(34257)-0.87654(57)j)')
        self.assertEqual(convert(uc.x, fmt), '                      +12.34568-0.87654j')
        self.assertEqual(convert(uc.u, fmt), '                       +0.34257+0.00057j')

        fmt = create(uc, fill='*', align='<', width=35)
        self.assertEqual(convert(uc, fmt),   '(12.34568(34257)-0.87654(57)j)*****')
        self.assertEqual(convert(uc.x, fmt), '12.34568-0.87654j******************')
        self.assertEqual(convert(uc.u, fmt), '0.34257+0.00057j*******************')

        uc = ucomplex(12.3456789 + 0.87654321j, 0)
        fmt = create(uc, type='f')  # precision defaults to 6
        self.assertEqual(convert(uc, fmt),   '(12.345679+0.876543j)')
        self.assertEqual(convert(uc.x, fmt), '12.345679+0.876543j')
        self.assertEqual(convert(uc.u, fmt), '0.000000+0.000000j')

        uc = ucomplex(12.3456789 + 0.87654321j, 0)
        fmt = create(uc, type='E')  # precision defaults to 6
        self.assertEqual(convert(uc, fmt),   '(1.234568E+01+8.765432E-01j)')
        self.assertEqual(convert(uc.x, fmt), '1.234568E+01+8.765432E-01j')
        self.assertEqual(convert(uc.u, fmt), '0.000000E+00+0.000000E+00j')
