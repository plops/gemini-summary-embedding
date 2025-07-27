# filepath: tests/test_smoke.py
import pytest

def test_imports():
    """
    A simple test to ensure all major libraries can be imported.
    """
    try:
        import numpy
        import pandas
        import seaborn
        import umap
        import sklearn
        import matplotlib
        import google.genai
        import sqlite_minutils
    except ImportError as e:
        pytest.fail(f"Failed to import a required library: {e}")
