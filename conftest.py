"""
Injects GTC in to the doctest namespace for pytest.
"""
import pytest

from GTC import *


@pytest.fixture(scope='session', autouse=True)
def add_gtc(doctest_namespace):
    for key, value in globals().items():
        if key.startswith('_'):
            continue
        doctest_namespace[key] = value
