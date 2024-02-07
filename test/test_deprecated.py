import unittest
import warnings

from GTC.deprecated import GTCDeprecationWarning
from GTC.deprecated import _append_sphinx_directive
from GTC.deprecated import deprecated


class TestDeprecated(unittest.TestCase):

    def test_function_no_args_no_kwargs(self):

        @deprecated
        def foo():
            """Docstring for foo"""
            return 'Called foo'

        self.assertEqual(foo.__name__, 'foo')
        self.assertEqual(
            foo.__doc__,
            'Docstring for foo\n\n.. warning::\n   The function `foo` is deprecated.\n'
        )

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            self.assertEqual(foo(), 'Called foo')
            self.assertEqual(len(warn), 1)
            self.assertTrue(issubclass(warn[0].category, GTCDeprecationWarning))
            self.assertEqual(str(warn[0].message), 'The function `foo` is deprecated.')
            self.assertEqual(warn[0].lineno, 26)  # where foo() is actually called

    def test_nested_function(self):
        @deprecated
        def foo():
            """Docstring for foo"""
            return 'Called foo'

        def bar():
            return foo()

        def baz():
            return bar()

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            self.assertEqual(baz(), 'Called foo')
            self.assertEqual(len(warn), 1)
            self.assertTrue(issubclass(warn[0].category, GTCDeprecationWarning))
            self.assertEqual(str(warn[0].message), 'The function `foo` is deprecated.')
            self.assertEqual(warn[0].lineno, 39)  # where foo() is called in bar()

    def test_class_no_args_no_kwargs(self):

        @deprecated
        class Foo:
            """Docstring for class foo"""
            def __init__(self): pass
            def bar(self): pass

        self.assertEqual(Foo.__name__, 'Foo')
        self.assertEqual(
            Foo.__doc__,
            'Docstring for class foo\n\n.. warning::\n   The class `Foo` is deprecated.\n'
        )

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            f = Foo()
            f.bar()
            self.assertEqual(len(warn), 1)  # f.bar() does not issue a warning
            self.assertTrue(issubclass(warn[0].category, GTCDeprecationWarning))
            self.assertEqual(str(warn[0].message), 'The class `Foo` is deprecated.')
            self.assertEqual(warn[0].lineno, 68)  # where Foo() is actually instantiated

            Foo()
            self.assertEqual(len(warn), 2)

    def test_method_no_args_no_kwargs(self):

        class Foo:
            """Docstring for class foo"""
            def __init__(self): pass

            @deprecated()
            def bar(self):
                """Docstring for method bar"""
                pass

        self.assertEqual(Foo.bar.__name__, 'bar')
        self.assertEqual(
            Foo.bar.__doc__,
            'Docstring for method bar\n\n.. warning::\n   The method `bar` is deprecated.\n'
        )

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            f = Foo()
            self.assertEqual(len(warn), 0)

            f.bar()
            self.assertEqual(len(warn), 1)
            self.assertTrue(issubclass(warn[0].category, GTCDeprecationWarning))
            self.assertEqual(str(warn[0].message), 'The method `bar` is deprecated.')
            self.assertEqual(warn[0].lineno, 100)  # where f.bar() is actually called

            Foo()
            self.assertEqual(len(warn), 1)

    def test_reason_as_arg(self):
        @deprecated('Do not use anymore')
        def function(x=0):
            return x

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            self.assertEqual(function(), 0)
            self.assertEqual(len(warn), 1)
            self.assertTrue(issubclass(warn[0].category, GTCDeprecationWarning))
            self.assertEqual(
                str(warn[0].message),
                'The function `function` is deprecated. Do not use anymore'
            )
            self.assertEqual(function(1), 1)
            self.assertEqual(len(warn), 2)
            self.assertEqual(function(x=21), 21)
            self.assertEqual(len(warn), 3)

    def test_reason_as_kwarg(self):
        @deprecated(reason='Do not use anymore')
        def function(x=0):
            return x

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            self.assertEqual(function(), 0)
            self.assertEqual(len(warn), 1)
            self.assertTrue(issubclass(warn[0].category, GTCDeprecationWarning))
            self.assertEqual(
                str(warn[0].message),
                'The function `function` is deprecated. Do not use anymore'
            )
            self.assertEqual(function(1), 1)
            self.assertEqual(len(warn), 2)
            self.assertEqual(function(x=21), 21)
            self.assertEqual(len(warn), 3)

    def test_deprecated_in(self):
        @deprecated('\nMessage\t\n', deprecated_in='1.2')
        def function(x, y=0):
            return x+y

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            self.assertEqual(function(0), 0)
            self.assertEqual(len(warn), 1)
            self.assertEqual(function(1, y=5), 6)
            self.assertEqual(len(warn), 2)
            self.assertEqual(function(x=21, y=-21), 0)
            self.assertEqual(len(warn), 3)
            for i in range(3):
                self.assertTrue(issubclass(warn[i].category, GTCDeprecationWarning))
                self.assertEqual(
                    str(warn[1].message),
                    'The function `function` is deprecated since version 1.2. \nMessage'
                )

    def test_deprecated_in_remove_in(self):
        @deprecated('Message to\nshow', deprecated_in='1.2', remove_in='9999.9999')
        def function():
            return

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            self.assertIsNone(function())
            self.assertEqual(len(warn), 1)
            self.assertTrue(issubclass(warn[0].category, GTCDeprecationWarning))
            self.assertEqual(
                str(warn[0].message),
                'The function `function` is deprecated since version 1.2 and is '
                'planned for removal in version 9999.9999. Message to\nshow'
            )

    def test_too_many_args(self):
        with self.assertRaises(SyntaxError) as err:
            @deprecated('A', '1.5')
            def function(): pass
        self.assertEqual(
            str(err.exception),
            "@deprecated('A', '1.5') has too many arguments, "
            "only the reason (as a string) is allowed"
        )

    def test_arg_wrong_type(self):
        with self.assertRaises(TypeError) as err:
            @deprecated(1.2)
            def function(): pass
        self.assertEqual(
            str(err.exception),
            "Cannot use @deprecated on an object of type 'float'"
        )

    def test_action_once(self):
        @deprecated(action='once')
        def function(): return

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            for _ in range(10):
                function()
            self.assertEqual(
                str(warn[0].message),
                'The function `function` is deprecated.'
            )

    def test_action_ignore(self):
        @deprecated(action='ignore')
        def function(): return

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            for _ in range(10):
                function()
            self.assertEqual(len(warn), 0)

    def test_action_error(self):
        @deprecated(action='error')
        def stop_using(): return

        with self.assertRaises(GTCDeprecationWarning) as err:
            stop_using()

        self.assertEqual(
            str(err.exception),
            "The function `stop_using` is deprecated."
        )

    def test_category(self):
        @deprecated(category=RuntimeWarning)
        def fcn(): return

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            self.assertIsNone(fcn())
            self.assertEqual(len(warn), 1)
            self.assertTrue(issubclass(warn[0].category, RuntimeWarning))
            self.assertEqual(
                str(warn[0].message),
                'The function `fcn` is deprecated.'
            )

    def test_stacklevel(self):
        @deprecated(stacklevel=2)
        def fcn(): return

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            fcn()
            # Not sure what lineno will be, but it won't be 247
            self.assertNotEqual(warn[0].lineno, 247)

    def test_docstring_line_width(self):
        @deprecated(deprecated_in='1.5', docstring_line_width=20)
        def fcn():
            """
            Docstring.
            """
            return

        self.assertEqual(
            fcn.__doc__,
            '\nDocstring.\n\n.. warning::\n   The function\n   `fcn` is'
            '\n   deprecated\n   since version\n   1.5.\n'
        )

    def test_remove_in_raises(self):
        # This must raise an exception because GTC version > remove_in value.
        # The `stop_using` function does not even need to be called.
        # Decorating it, is sufficient.
        # This does require that GTC.deprecated._running_tests = True
        # which should be True when the tests are run.
        with self.assertRaises(RuntimeError) as err:
            @deprecated(remove_in='0.1')
            def stop_using(): return

        self.assertTrue(str(err.exception).startswith('Dear GTC developer'))

    def test_static_method_1(self):
        # The inspect module cannot differential between a function
        # and a staticmethod, the default that chosen is a function

        class Foo:
            def __init__(self): pass

            @staticmethod
            @deprecated
            def bar(x=0):
                """Docstring."""
                return x

        self.assertEqual(Foo.bar.__name__, 'bar')
        self.assertEqual(
            Foo.bar.__doc__,
            'Docstring.\n\n.. warning::\n   The function `bar` is deprecated.\n'
        )

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            f = Foo()
            self.assertEqual(len(warn), 0)
            self.assertEqual(f.bar(8), 8)
            self.assertEqual(len(warn), 1)
            self.assertTrue(issubclass(warn[0].category, GTCDeprecationWarning))
            self.assertEqual(str(warn[0].message), 'The function `bar` is deprecated.')
            self.assertEqual(warn[0].lineno, 310)  # where f.bar() is actually called

    def test_static_method_2(self):
        # The inspect module cannot differential between a function
        # and a staticmethod, explicitly define the "kind"

        class Foo:
            def __init__(self): pass

            @staticmethod
            @deprecated(kind='staticmethod')
            def bar(x=0):
                """Docstring."""
                return x

        self.assertEqual(Foo.bar.__name__, 'bar')
        self.assertEqual(
            Foo.bar.__doc__,
            'Docstring.\n\n.. warning::\n   The staticmethod `bar` is deprecated.\n'
        )

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            f = Foo()
            self.assertEqual(len(warn), 0)
            self.assertEqual(f.bar(8), 8)
            self.assertEqual(len(warn), 1)
            self.assertTrue(issubclass(warn[0].category, GTCDeprecationWarning))
            self.assertEqual(str(warn[0].message), 'The staticmethod `bar` is deprecated.')
            self.assertEqual(warn[0].lineno, 339)  # where f.bar() is actually called

    def test_class_method(self):
        # The inspect module cannot differential between a function
        # and a staticmethod, the default that chosen is a function

        class Foo:
            def __init__(self, x=0):
                self.x = x

            @classmethod
            @deprecated
            def bar(cls, x):
                """Docstring.

                :param x: The x value.
                :type x: int
                """
                return Foo(x=x)

        self.assertEqual(Foo.bar.__name__, 'bar')
        self.assertEqual(
            Foo.bar.__doc__,
            'Docstring.\n\n'
            ':param x: The x value.\n'
            ':type x: int\n\n'
            '.. warning::\n'
            '   The classmethod `bar` is deprecated.\n'
        )

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            f = Foo()
            self.assertEqual(len(warn), 0)
            f2 = f.bar(8)
            self.assertEqual(len(warn), 1)
            self.assertTrue(issubclass(warn[0].category, GTCDeprecationWarning))
            self.assertEqual(str(warn[0].message), 'The classmethod `bar` is deprecated.')
            self.assertEqual(warn[0].lineno, 377)  # where f.bar() is actually called
            self.assertEqual(f2.x, 8)

    def test_update_docstring(self):

        @deprecated(update_docstring=False)
        class Foo:
            """
            Docstring for class foo
            """
            def __init__(self): pass
            def bar(self): pass

        self.assertEqual(Foo.__name__, 'Foo')
        self.assertEqual(
            Foo.__doc__,
            '\n            Docstring for class foo\n            '
        )

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            f = Foo()
            f.bar()
            f.bar()
            f.bar()
            self.assertEqual(len(warn), 1)  # f.bar() does not issue a warning
            self.assertTrue(issubclass(warn[0].category, GTCDeprecationWarning))
            self.assertEqual(str(warn[0].message), 'The class `Foo` is deprecated.')
            self.assertEqual(warn[0].lineno, 402)  # where Foo() is actually instantiated

    def test_multiple(self):
        @deprecated
        def foo():
            """foo"""
            return 'foo'

        @deprecated('Bar', deprecated_in='1.1')
        def bar():
            """bar"""
            return 'bar'

        @deprecated(reason='Baz', deprecated_in='0.8', remove_in='9999.9999')
        def baz():
            """baz"""
            return 'baz'

        def hello():
            return 'world'

        self.assertEqual(foo.__name__, 'foo')
        self.assertEqual(
            foo.__doc__,
            'foo\n\n'
            '.. warning::\n'
            '   The function `foo` is deprecated.\n'
        )

        self.assertEqual(bar.__name__, 'bar')
        self.assertEqual(
            bar.__doc__,
            'bar\n\n.. warning::\n   The function `bar` is deprecated '
            'since version 1.1. Bar\n'
        )

        self.assertEqual(baz.__name__, 'baz')
        self.assertEqual(
            baz.__doc__,
            'baz\n\n.. warning::\n   The function `baz` is deprecated since '
            'version 0.8 and is planned for\n   removal in version 9999.9999. Baz\n'
        )

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            self.assertEqual(foo(), 'foo')
            self.assertEqual(hello(), 'world')
            self.assertEqual(foo(), 'foo')
            self.assertEqual(bar(), 'bar')
            self.assertEqual(foo(), 'foo')
            self.assertEqual(baz(), 'baz')
            self.assertEqual(baz(), 'baz')
            self.assertEqual(hello(), 'world')

            self.assertEqual(len(warn), 6)
            self.assertEqual(
                str(warn[0].message),
                'The function `foo` is deprecated.'
            )
            self.assertEqual(
                str(warn[1].message),
                'The function `foo` is deprecated.'
            )
            self.assertEqual(
                str(warn[2].message),
                'The function `bar` is deprecated since version 1.1. Bar'
            )
            self.assertEqual(
                str(warn[3].message),
                'The function `foo` is deprecated.'
            )
            self.assertEqual(
                str(warn[4].message),
                'The function `baz` is deprecated since version 0.8 and '
                'is planned for removal in version 9999.9999. Baz')
            self.assertEqual(
                str(warn[5].message),
                'The function `baz` is deprecated since version 0.8 and '
                'is planned for removal in version 9999.9999. Baz')

    def test_reason_with_indents(self):

        class Foo:
            def __init__(self): pass

            @deprecated("""Do not use this anymore.

        Paragraph 1
            Indent 1

        Paragraph 2

            Indent 1
            Same indent
                Indent 2            
            """)
            def bar(self):
                """
                Docstring for bar.
                """
                return 'bar'

        self.assertEqual(Foo.bar.__name__, 'bar')
        self.assertEqual(
            Foo.bar.__doc__,
            '\nDocstring for bar.\n\n'
            '.. warning::\n'
            '   The method `bar` is deprecated. Do not use this anymore.\n\n'
            '           Paragraph 1\n'
            '               Indent 1\n\n'
            '           Paragraph 2\n\n'
            '               Indent 1\n'
            '               Same indent\n'
            '                   Indent 2\n'
        )

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            self.assertEqual(Foo().bar(), 'bar')
            self.assertEqual(len(warn), 1)
            self.assertTrue(issubclass(warn[0].category, GTCDeprecationWarning))
            self.assertEqual(
                str(warn[0].message),
                'The method `bar` is deprecated. Do not use this anymore.\n\n'
                '        Paragraph 1\n'
                '            Indent 1\n\n'
                '        Paragraph 2\n\n'
                '            Indent 1\n'
                '            Same indent\n'
                '                Indent 2'
            )
            self.assertEqual(warn[0].lineno, 527)  # where Foo().bar() is actually called

    def test_remove_sphinx_role(self):

        @deprecated('Do not use anymore, use :func:`.dump_xml`, '
                    ':py:func:`~GTC.persistence.dump_json` or '
                    ':meth:`Archive.dump` instead')
        class Foo:
            """Docstring for class foo."""
            def __init__(self): pass
            def bar(self): pass

        self.assertEqual(Foo.__name__, 'Foo')
        self.assertEqual(
            Foo.__doc__,
            'Docstring for class foo.\n\n'
            '.. warning::\n'
            '   The class `Foo` is deprecated. Do not use anymore, use :func:`.dump_xml`,\n'
            '   :py:func:`~GTC.persistence.dump_json` or :meth:`Archive.dump` instead\n'
        )

        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            Foo()
            self.assertEqual(len(warn), 1)
            self.assertTrue(issubclass(warn[0].category, GTCDeprecationWarning))
            self.assertEqual(
                str(warn[0].message),
                'The class `Foo` is deprecated. Do not use anymore, use `.dump_xml`, '
                '`~GTC.persistence.dump_json` or `Archive.dump` instead'
            )
            self.assertEqual(warn[0].lineno, 563)  # where Foo() is actually instantiated

    def test_docstring_1(self):
        s = _append_sphinx_directive(None, 'Message')
        self.assertEqual(
            s,
            '\n\n.. warning::\n   Message\n'
        )

    def test_docstring_2(self):
        s = _append_sphinx_directive(None, 'Message', docstring_line_width=4)
        self.assertEqual(
            s,
            '\n\n.. warning::\n   M\n   e\n   s\n   s\n   a\n   g\n   e\n'
        )

    def test_docstring_3(self):
        s = _append_sphinx_directive(' \n\n \n \t', 'This is now deprecated')
        self.assertEqual(
            s,
            '\n\n.. warning::\n   This is now deprecated\n'
        )

    def test_docstring_4(self):
        s = _append_sphinx_directive('Hello', 'This is now deprecated')
        self.assertEqual(
            s,
            'Hello\n\n.. warning::\n   This is now deprecated\n'
        )

    def test_docstring_5(self):
        docstring = """
            Hello world
            """

        s = _append_sphinx_directive(docstring, 'This is now deprecated')
        self.assertEqual(
            s,
            '\nHello world\n\n.. warning::\n   This is now deprecated\n'
        )

    def test_docstring_6(self):
        docstring = """Hello world

                    A big indentation.
                    Is occurring.

                    :param x: The x value.
                        Must be > 0.


                    """

        s = _append_sphinx_directive(
            docstring, 'This is now deprecated', docstring_line_width=13)

        self.assertEqual(
            s,
            'Hello world\n\nA big indentation.\nIs occurring.\n\n'
            ':param x: The x value.\n    Must be > 0.\n\n.. warning::'
            '\n   This is\n   now dep\n   recated\n'
        )

    def test_docstring_7(self):
        docstring = """
                    Hello world

                    A big indentation.
                    Is occurring.

                    :param x: The x value.
                        Must be > 0.
                    """

        s = _append_sphinx_directive(
            docstring, 'This is now deprecated', docstring_line_width=13)

        self.assertEqual(
            s,
            '\nHello world\n\nA big indentation.\nIs occurring.\n\n'
            ':param x: The x value.\n    Must be > 0.\n\n.. warning::'
            '\n   This is\n   now dep\n   recated\n'
        )

    def test_docstring_8(self):
        docstring = """
\tHello world
\t
\tThis docstring uses tabs.
"""

        s = _append_sphinx_directive(
            docstring, 'This is now deprecated')

        self.assertEqual(
            s,
            '\nHello world\n\nThis docstring uses tabs.'
            '\n\n.. warning::\n   This is now deprecated\n'
        )

    def test_docstring_9(self):
        docstring = """Hello world
\t
\tThis docstring uses tabs.
"""

        s = _append_sphinx_directive(
            docstring, 'This is now deprecated')

        self.assertEqual(
            s,
            'Hello world\n\nThis docstring uses tabs.'
            '\n\n.. warning::\n   This is now deprecated\n'
        )


if __name__ == '__main__':
    unittest.main()
