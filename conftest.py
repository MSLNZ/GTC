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
