import gixstools
import gixstools.align
from pathlib import Path


def test_macro():
    dir = Path(__file__).parent
    angles = gixstools.align.macro.arange_list(start=-1, finish=1, step=.2)
    positions = gixstools.align.macro.arange_list(start=-1, finish=1, step=.2)
    gixstools.align.create_om_macro(angles, dir / "macro-test-dump")
    gixstools.align.create_z_macro(positions, dir / "macro-test-dump")


if __name__ == "__main__":
    test_macro()