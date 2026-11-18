'''
@author: Michael

Shared pytest configuration for all tests of the project.
'''

import matplotlib
import pytest

# Never open plot windows during the tests.
matplotlib.use("Agg")


@pytest.fixture(autouse = True)
def close_figures():
    yield
    import matplotlib.pyplot as mpl
    mpl.close("all")


# END OF FILE
