"""
Injects GTC in to the doctest namespace for pytest.
"""
import os
import sys

import pytest

# This environment variable must be defined before GTC is imported
os.environ['GTC_RUNNING_TESTS'] = 'true'

from GTC import *


@pytest.fixture(autouse=True)
def add_gtc(doctest_namespace):
    for key, val in globals().items():
        if key.startswith('_'):
            continue
        doctest_namespace[key] = val

    if sys.version_info.major > 2:
        doctest_namespace['xrange'] = range
