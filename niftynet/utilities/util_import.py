# -*- coding: utf-8 -*-
""" check module to be imported"""
import importlib

import tensorflow as tf


def require_module(name,
                   min_version=None,
                   descriptor='Optional',
                   mandatory=False):
    """
    Check if the module exists, and
    satisfies the minimum version requirement.

    Returns the imported module if it satisfies requirements.

    Raises ImportError and AssertionError.

    :param name:
    :param min_version:
    :param descriptor:
    :param mandatory:
    :return: the imported module
    """

    name = '{}'.format(name)

    try:
        the_module = importlib.import_module(name)
    except ImportError:
        if mandatory:
            raise

    try:
        if min_version is not None:
            if isinstance(min_version, tuple):
                version_number = the_module.__version__.split('.')
                min_version = tuple(int(v) for v in min_version)
                mod_version = tuple(int(v) for v in version_number)
            else:
                mod_version = the_module.__version__
                min_version = '{}'.format(min_version)

            assert mod_version >= min_version
        return the_module
    except AttributeError:
        pass
    except (AssertionError, NameError):
        if mandatory:
            raise
