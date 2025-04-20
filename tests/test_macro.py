import gixstools
import gixstools.align
from pathlib import Path
from datetime import datetime


def test_macro():
    print("--------")
    print("Testing macro writing tools")
    print("--------")
    dir = Path(__file__).parent
    angles = gixstools.align.macro.create_list(start=-1, finish=1, step=.2)
    positions = gixstools.align.macro.create_list(start=-1, finish=1, step=.2)
    date = datetime.now()
    date_stamp = f"{date.year:02d}{date.month:02d}{date.day:02d}-{date.hour:02d}"
    dump_folder = dir / "macro-test-dump"
    gixstools.align.create_om_macro(angles, dump_folder)
    gixstools.align.create_z_macro(positions, dump_folder)
    
    specular_format = str(dump_folder / "Specular_om_macro-{}.txt")
    new_file = Path(specular_format.format(date_stamp))
    with open(specular_format.format("master"), "r") as master, open(new_file, "r") as test:
        assert master.read() == test.read()
    new_file.unlink()
    print(" - omega-scan macro written successfully")

    specular_format = str(dump_folder / "Specular_z_macro-{}.txt")
    new_file = Path(specular_format.format(date_stamp))
    with open(specular_format.format("master"), "r") as master, open(new_file, "r") as test:
        assert master.read() == test.read()
    new_file.unlink()
    print(" - z-scan macro written successfully")
    
    print("--------")
    print("PASSED: macro writing test")
    print("--------")

if __name__ == "__main__":
    test_macro()