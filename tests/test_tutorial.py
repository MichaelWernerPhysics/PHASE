'''
@author: Michael

Smoke test: The tutorial (examples/phase_tutorial.py) has to run through with the current version of the package.
'''

import runpy
from pathlib import Path

import pytest

TUTORIAL = Path(__file__).resolve().parents[1] / "examples" / "phase_tutorial.py"


@pytest.mark.slow
def test_tutorial_runs(capsys):
    runpy.run_path(str(TUTORIAL), run_name = "__main__")

    output = capsys.readouterr().out
    assert "11. Gravity gradient noise" in output


# END OF FILE
