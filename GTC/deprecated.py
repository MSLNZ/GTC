"""
This module contains a `deprecated` function that may be used
to mark a function, class or method as deprecated.
"""
import functools
import inspect
import re
import sys
import textwrap
import warnings

from GTC import version

PY2 = sys.version_info.major == 2

_running_tests = 'unittest' in sys.modules

# Strip Sphinx cross-reference syntax (like ":function:" and ":py:func:")
# from warning messages that are written to stdout. The format of the syntax
# are ":role:`foo`", ":domain:role:`foo`", where ``role`` and ``domain``
# match "[a-zA-Z]+"
_regex_remove_role = re.compile(
    r'(?: : [a-zA-Z]+ )? : [a-zA-Z]+ : (`[^`]*`)', flags=re.X)


__all__ = (
    'GTCDeprecationWarning',
    'deprecated',
)


class GTCDeprecationWarning(UserWarning):
    """
    By default, Python will not show a DeprecationWarning because it
    is ignored by the default warning-filter settings. Inheriting from
    UserWarning, instead of DeprecationWarning, allows for the warning
    to be displayed more reliably.

    DeprecationWarning is meant for internal use by the Python developers.
    From the Python documentation:

        Base category for warnings about deprecated features when those
        warnings are intended for other Python developers (ignored by
        default, unless triggered by code in __main__).

    """


def deprecated(*args, **kwargs):
    """Use as a decorator to mark a function, class or method as deprecated.

    :param args: The number of positional arguments may be either zero or one.
        If one, then it must be the reason (as a string) why the object is
        marked as deprecated.

        Examples,

            @deprecated
            def foo():

            @deprecated()
            def foo():

            @deprecated('Use :func:`bar` instead')
            def foo():

    :param kwargs: The following keyword arguments may be specified.

        action : str or None
            The type of simple-warning filter to use. One of "error",
            "ignore", "always", "default", "module", or "once". If None,
            then the global-warning-filter setting is used. Default is None.

        category : Type[Warning]
            The type of Warning class to use. Default is GTCDeprecationWarning.

        deprecated_in : str or None
            The version that the wrapped object became deprecated in.
            Default is None.

        docstring_line_width: int
            The maximum line width of the deprecation message that gets
            appended to the docstring of the wrapped object. In order for
            the message to be appended to the docstring, `update_docstring`
            must be True. Default is 80.

        kind : str or None
            The kind of object that is wrapped (e.g., "function", "class"
            "staticmethod"). The `inspect` module automatically tries to
            determine the value of `kind`, but `inspect` may sometimes fail
            to identify the object appropriately, in which case, you can
            explicitly specify the kind of object that is wrapped (which
            will skip inspection). Default is None.

        reason : str or None
            The reason for issuing the warning. Default is None.

        remove_in : str or None
            The version that the wrapped object is planned to be removed in.
            If specified, and the version of GTC is greater than or equal
            to the `remove_in` value, an exception is raised when
            @decorated() is invoked. Default is None.

        stacklevel : int
            Number of frames up the stack that issued the warning. Default is 3.

        update_docstring : bool
            Whether to append the deprecation message to the docstring of the
            wrapped object (for rendering with Sphinx). Default is True.

        Examples,

            @deprecated(reason='Stop using')
            def foo():

            @deprecated('Stop using', deprecated_in='1.5', remove_in='2.0')
            def foo():

            @deprecated('Use :func:`bar` instead', action='error')
            def foo():

    """
    def wrapper(obj):
        message, warn_kw = _prepare_warning(obj, **kwargs)
        message = _regex_remove_role.sub(r'\1', message)

        @functools.wraps(obj)
        def _wrapper(*a, **kw):
            # *a and **kw are the arguments and keyword arguments
            # that the wrapped object takes
            _warn(message, **warn_kw)
            return obj(*a, **kw)
        return _wrapper

    if not (args or kwargs):
        # Handles @deprecated()
        return deprecated

    if args:
        if len(args) > 1:
            raise SyntaxError(
                '@deprecated{} has too many arguments, '
                'only the reason (as a string) is allowed'.format(args))

        arg0 = args[0]

        # Handles if a string is passed in as the first argument, e.g.,
        #   @deprecated('A message')
        #   @deprecated('A message', deprecated_in='1.5')
        if isinstance(arg0, str):
            return deprecated(reason=arg0, **kwargs)

        if not (inspect.isfunction(arg0) or inspect.isclass(arg0)):
            raise TypeError(
                'Cannot use @deprecated on an object of type {!r}'.format(
                    type(arg0).__name__))

        # Handles @deprecated
        return wrapper(arg0)

    # Handles keyword arguments, e.g.,
    #   @deprecated(reason='A message')
    #   @deprecated(deprecated_in='1.3', remove_in='2.0')
    def _decorated(func):
        return wrapper(func)
    return _decorated


def _prepare_warning(wrapped,
                     action=None,
                     category=GTCDeprecationWarning,
                     deprecated_in=None,
                     docstring_line_width=80,
                     kind=None,
                     reason=None,
                     remove_in=None,
                     stacklevel=3,
                     update_docstring=True):
    """Prepare the deprecation message.

    :param wrapped: A callable object that is marked as deprecated.

    All keyword arguments as defined in :func:`decorated`.

    :return: The warning message and the keyword argument for :func:`_warn`.
    :rtype: tuple(str, dict)
    """
    if not kind:
        if inspect.isfunction(wrapped):
            if PY2:
                args = inspect.getargspec(wrapped).args
            else:
                args = inspect.getfullargspec(wrapped).args

            if args and args[0] == 'self':
                kind = 'method'
            elif args and args[0] == 'cls':
                kind = 'classmethod'
            else:  # Could be a function or staticmethod, assume function
                kind = 'function'
        elif inspect.isclass(wrapped):
            kind = 'class'
        else:
            kind = 'callable object'

    # Builds the deprecation message
    msg = ['The {} `{}` is deprecated'.format(kind, wrapped.__name__)]
    if deprecated_in:
        msg.append(' since version {}'.format(deprecated_in))
    if remove_in:
        msg.append(' and is planned for removal in version {}'.format(remove_in))
    msg.append('. ')
    if reason:
        msg.append(reason)

    message = ''.join(msg).rstrip()

    if update_docstring:
        wrapped.__doc__ = _append_sphinx_directive(
            wrapped.__doc__,
            message,
            docstring_line_width=docstring_line_width,
        )

    if _running_tests and (
            remove_in and
            tuple(map(int, version.split('.')[:3])) >=
            tuple(map(int, remove_in.split('.')))):
        raise RuntimeError(
            'Dear GTC developer, you are still using a {} that '
            'should be removed:\n{}'.format(kind, message)
        )

    warn_kw = {'action': action, 'category': category, 'stacklevel': stacklevel}
    return message, warn_kw


def _append_sphinx_directive(docstring,
                             message,
                             docstring_line_width=80):
    """Append the ".. warning::" Sphinx directive to a docstring.

    :param docstring: The docstring of the wrapped object.
    :type docstring: str or None

    :param message: The warning message.
    :type message: str

    All other keyword arguments are defined in :func:`decorated`.

    :return: The docstring with the Sphinx directive appended.
    :rtype: str
    """
    # A docstring can be None
    docstring = docstring or ''

    # The docstring may either start on the same line as the triple
    # quotes or on the next line. Ensure that the dedent() function
    # handles either case correctly.
    lines = docstring.rstrip().splitlines(True) or ['']
    docstring = lines[0] + textwrap.dedent(''.join(lines[1:]))

    indent = '   '
    width = max(1, docstring_line_width - len(indent))

    # there must be at least one empty line before the Sphinx directive
    directive = ['\n\n.. warning::']
    for line in message.splitlines():
        if line:
            directive.append(
                textwrap.fill(
                    line,
                    width=width,
                    initial_indent=indent,
                    subsequent_indent=indent,
                ))
        else:
            directive.append('')

    # an empty line should follow the Sphinx directive
    directive.append('')

    return docstring + '\n'.join(directive)


def _warn(message,
          action=None,
          category=GTCDeprecationWarning,
          stacklevel=3):
    """Issue a warning, or maybe ignore it or raise an exception.

    :param message: The warning message.
    :type message: str

    All other keyword arguments are defined in :func:`decorated`.
    """
    if action:
        with warnings.catch_warnings():
            warnings.simplefilter(action, category)
            warnings.warn(message, category=category, stacklevel=stacklevel)
    else:
        warnings.warn(message, category=category, stacklevel=stacklevel)
