import unittest

from GTC import ureal, ucomplex, inf_dof
from GTC import formatting


class TestFormatting(unittest.TestCase):

    def test_parse_raises(self):
        # want the exception types raised by formatting.parse to
        # match what the builtin format(float, format_spec) would raise
        def check(exception, format_spec):
            self.assertRaises(exception, format, 1.0, format_spec)
            self.assertRaises(exception, formatting.parse, format_spec)

        # format_spec must be a str
        check(TypeError, 1.2)
        check(TypeError, ureal(1, 0.1))
        check(TypeError, ucomplex(1+1j, 0.1))

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
                'zero': None, 'width': None, 'grouping': None, 'precision': None,
                'type': None, 'df_decimals': None, 'mode': None
            }
            out.update(**kwargs)
            return out

        # only check the builtin-supported fields
        self.assertEqual(parse('G'), expect(type='G'))
        self.assertEqual(parse('='), expect(align='='))
        self.assertEqual(parse(' ='), expect(fill=' ', align='='))
        self.assertEqual(parse('<<'), expect(fill='<', align='<'))
        self.assertEqual(parse(' 10.1'), expect(sign=' ', width=10, precision=1))
        self.assertEqual(parse('02'), expect(zero='0', width=2))
        self.assertEqual(parse('02.0'), expect(zero='0', width=2, precision=0))
        self.assertEqual(parse('.10'), expect(precision=10))

        self.assertEqual(parse('07.2f'),
                         expect(zero='0', width=7, precision=2, type='f'))

        self.assertEqual(parse('*<-06,.4E'),
                         expect(fill='*', align='<', sign='-', zero='0', width=6,
                                grouping=',', precision=4, type='E'))

        # additional GTC-specific fields
        self.assertEqual(parse('B', False), expect(mode='B'))
        self.assertEqual(parse('GB', False), expect(type='G', mode='B'))
        self.assertEqual(parse('.2P', False), expect(precision=2, mode='P'))
        self.assertEqual(parse('.7.5', False), expect(precision=7, df_decimals=5))
        self.assertEqual(parse('e.11', False), expect(type='e', df_decimals=11))

        self.assertEqual(parse('.2f.0', False),
                         expect(precision=2, type='f', df_decimals=0))

        self.assertEqual(parse('.2f.3R', False),
                         expect(precision=2, type='f', df_decimals=3, mode='R'))

        self.assertEqual(parse(' ^16.4fL', False),
                         expect(fill=' ', align='^', width=16, precision=4,
                                type='f', mode='L'))

        self.assertEqual(parse('*> #011,.2g.8S', False),
                         expect(fill='*', align='>', sign=' ', hash='#', zero='0', width=11,
                                grouping=',', precision=2, type='g', df_decimals=8, mode='S'))

    def test_Format(self):
        f = formatting.Format()
        self.assertEqual(f.format_spec, '')

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

    def test_ureal_repr(self):
        def check(ur, expected):
            # different ways to get the same result
            self.assertEqual(repr(ur), expected)
            self.assertEqual('{!r}'.format(ur), expected)

        check(ureal(1.23456789, 0.001), 'ureal(1.23456789,0.001,inf)')
        check(ureal(-1, 1.23456789e-7, df=7), 'ureal(-1.0,1.23456789e-07,7.0)')
        check(ureal(3, 0.01, df=inf_dof+1), 'ureal(3.0,0.01,inf)')
        check(ureal(1.23456789e10, 10), "ureal(12345678900.0,10.0,inf)")

        check(ureal(1.23456789e18, 10, label='numbers'),
              "ureal(1.23456789e+18,10.0,inf, label='numbers')")

        check(ureal(1.23456789e-9, 2.1e-11),
              'ureal(1.23456789e-09,2.1e-11,inf)')

        check(ureal(3.141592653589793, 0.01, df=3, label='PI'),
              "ureal(3.141592653589793,0.01,3.0, label='PI')")

    def test_ureal_str(self):
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

    def test_ucomplex_repr(self):
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

    def test_ucomplex_str(self):
        def check(uc, expected):
            # different ways to get the same result
            self.assertEqual(str(uc), expected)
            self.assertEqual('{}'.format(uc), expected)
            self.assertEqual('{!s}'.format(uc), expected)
            self.assertEqual('{:+.2f.0B}'.format(uc), expected)

        check(ucomplex(1.23456789 + 9.87654321j, 1000), '(+0(1000)+0(1000)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 100), '(+0(100)+10(100)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 10), '(+1(10)+10(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 1), '(+1.2(1.0)+9.9(1.0)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0), '(+1.234568(0)+9.876543(0)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.1), '(+1.23(10)+9.88(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.01), '(+1.235(10)+9.877(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.001), '(+1.2346(10)+9.8765(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.0001), '(+1.23457(10)+9.87654(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.00001), '(+1.234568(10)+9.876543(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.000001), '(+1.2345679(10)+9.8765432(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.0000001), '(+1.23456789(10)+9.87654321(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.00000001), '(+1.234567890(10)+9.876543210(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.000000001), '(+1.2345678900(10)+9.8765432100(10)j)')
        check(ucomplex(1.23456789 + 9.87654321j, 0.0000000001), '(+1.23456789000(10)+9.87654321000(10)j)')

        check(ucomplex(1.23456789e16 + 9.87654321e14j, 1e13),
              '(+12346000000000000(10000000000000)+988000000000000(10000000000000)j)')

        check(ucomplex(1.23456789e-16 + 9.87654321e-14j, 1e-18),
              '(+0.0000000000000001235(10)+0.0000000000000987654(10)j)')
