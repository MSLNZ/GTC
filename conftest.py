"""
Injects GTC in to the doctest namespace for pytest.
"""
import pytest


@pytest.fixture(scope='session', autouse=True)
def add_gtc(doctest_namespace):
    from GTC import *
    for key, value in locals().items():
        if key.startswith('_'):
            continue
        doctest_namespace[key] = value
