"""
Injects GTC in to the doctest namespace for pytest.
"""
import sys

import pytest

from GTC import *


@pytest.fixture(autouse=True)
def add_gtc(doctest_namespace):
    for key, val in globals().items():
        if key.startswith('_'):
            continue
        doctest_namespace[key] = val

    if sys.version_info.major > 2:
        doctest_namespace['xrange'] = range

    if sys.version_info.major == 2:
        PY27 = lambda: pytest.skip(msg="Skip Python 2.7 since unicode strings require u''")
    else:
        PY27 = lambda: None

    doctest_namespace['SKIP_IF_PYTHON_27'] = PY27
