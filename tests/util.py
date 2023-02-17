import pytest


def suite_A(test_method):
    return pytest.mark.suite_A(test_method)


def suite_B(test_method):
    return pytest.mark.suite_B(test_method)


def suite_C(test_method):
    return pytest.mark.suite_C(test_method)
