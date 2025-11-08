from pathlib import Path
import matplotlib.pylab as plt

from gixstools.align import DirectBeam


def test_beam_finder():
    data_path = Path("tests/test-data/")
    filename_om_db = data_path / "ex-om-scan/om_scan_direct_beam.tif"
    filename_z_db = data_path / "ex-z-scan/z_scan_direct_beam.tif"
    assert filename_om_db.exists()
    assert filename_z_db.exists()
    db1 = DirectBeam(filename_om_db)
    db1.find_center()
    db1.show_beam()
    plt.show()


if __name__ == "__main__":
    test_beam_finder()
    # plt.show()