"""
This module contains a `deprecated` function that may be used
as a decorator to mark a function, class or method as deprecated.
"""
import functools
import inspect
import os
import re
import sys
import textwrap
import warnings

from GTC import version

PY2 = sys.version_info.major == 2

# This environment variable is defined in conftest.py when pytest runs the tests
_running_tests = os.getenv('GTC_RUNNING_TESTS', 'false') == 'true'

# Strip Sphinx cross-reference syntax (like ":class:" and ":py:func:") from
# warning messages that are passed to warnings.warn(). The format of the syntax
# are ":role:`foo`" and ":domain:role:`foo`", where ``role`` and ``domain``
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

    See https://docs.python.org/3/library/warnings.html#default-warning-filter
    """


def deprecated(*args, **kwargs):
    """Use as a decorator to mark a function, class or method as deprecated.

    :param args: The number of positional arguments may either be zero or one.
        If one, then it must be the reason (as a string) why the object is
        deprecated.

        Examples,

            @deprecated
            def foo():

            @deprecated()
            def foo():

            @deprecated("The reason why `foo` is deprecated")
            def foo():

    :param kwargs: The following keyword arguments may be specified.

        action : str or None
            The type of filter to use when the warning is issued. One of
            "always", "default", "error", "ignore" or "once".
            If None, the global warning-filter setting is used. Default is None.
            See https://docs.python.org/3/library/warnings.html#the-warnings-filter

        category : Type[Warning]
            The type of Warning class to use. Default is GTCDeprecationWarning.

        deprecated_in : str or None
            The version that the wrapped object became deprecated in.
            Default is None.

        docstring_line_width : int
            The maximum line width of the deprecation message that gets
            appended to the docstring of the wrapped object. In order for
            the message to be appended to the docstring, `update_docstring`
            must be True. Default is 79.

        kind : str or None
            The kind of object that is wrapped (e.g., "function", "class",
            "staticmethod"). The `inspect` module determines the value of
            `kind`, but `inspect` may sometimes fail to identify the object
            appropriately, in which case, you can explicitly specify the
            kind of object that is wrapped (which will skip inspection).
            Default is None.

        prefix : str or None
            The text to insert before the deprecation message that is
            constructed from other keyword arguments. For example, you may
            want to insert the newline character so that when warnings.warn()
            is called, the deprecation message appears on a new line.
            Default is None.

        reason : str or None
            The reason for issuing the warning. Default is None.

        remove_in : str or None
            The version that the wrapped object is planned to be removed in.
            If the tests are running (i.e., a GTC_RUNNING_TESTS environment
            variable is set to be "true") and the version of GTC is greater
            than or equal to the `remove_in` value, an exception is raised
            when @deprecated` is called (not when the wrapped object is
            called), which occurs when the module that contains the deprecated
            object gets imported. Default is None.

        stacklevel : int
            Number of frames up the stack that issued the warning. Default is 3.

        update_docstring : bool
            Whether to append the deprecation message (as a Sphinx "warning::"
            directive) to the docstring of the wrapped object. Default is True.

        Examples,

            @deprecated(reason="Stop using")
            def foo():

            @deprecated("Stop using", deprecated_in="1.5", remove_in="2.0")
            def foo():

            @deprecated("Use :func:`bar` instead", action="error")
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
                     prefix=None,
                     reason=None,
                     remove_in=None,
                     stacklevel=3,
                     update_docstring=True):
    """Prepare the deprecation warning.

    :param wrapped: A callable object that is deprecated.

    All other keyword arguments are defined in :func:`deprecated`.

    :return: The warning message and the keyword arguments for :func:`_warn`.
    :rtype: tuple[str, dict]
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
    msg = []
    if prefix:
        msg.append(prefix)
    msg.append('The {} `{}` is deprecated'.format(kind, wrapped.__name__))
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

    if action == 'module':
        raise ValueError('Using action="module" is not supported')

    warn_kw = {'action': action, 'category': category, 'stacklevel': stacklevel}
    return message, warn_kw


def _append_sphinx_directive(docstring,
                             message,
                             docstring_line_width=79):
    """Append the "warning::" Sphinx directive to a docstring.

    :param docstring: The docstring of an object.
    :type docstring: str or None

    :param message: The warning message, without the "warning::" directive.
    :type message: str

    All other keyword arguments are defined in :func:`deprecated`.

    :return: The docstring with the directive appended.
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

    All other keyword arguments are defined in :func:`deprecated`.
    """
    # When specifying the action, the catch_warnings() context manager allows
    # the warning behavior to be modified when entering the context and is
    # restored when exiting. Using the context manager is required so that the
    # application's global context is not modified. In the case of "once", this
    # has the effect of restoring the internal call counter, so a warning message
    # is issued every time the line issuing the warning is called. Therefore,
    # for action="once" (and "default") we treat it equivalent to not specifying
    # the action.
    if action and action not in ('once', 'default'):
        with warnings.catch_warnings():
            warnings.simplefilter(action, category)
            warnings.warn(message, category=category, stacklevel=stacklevel)
    else:
        warnings.warn(message, category=category, stacklevel=stacklevel)
