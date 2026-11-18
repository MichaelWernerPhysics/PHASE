'''
@author: Michael

Tests of the package metadata.
'''

import re
from pathlib import Path

import phase


def test_citation_version_matches_package_version():
    citation = (Path(__file__).parents[1] / "CITATION.cff").read_text()
    match = re.search(r"^version:\s*(\S+)\s*$", citation, flags = re.MULTILINE)

    assert match is not None, "CITATION.cff has no version entry."
    assert match.group(1) == phase.__version__


# END OF FILE
