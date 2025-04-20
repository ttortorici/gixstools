from unittest import TestCase
from gixstools.align import DirectBeam
from pathlib import Path
import matplotlib.pylab as plt


class TestBeamFinder(TestCase):
    def test_beam_finder(self):
        data_path = Path("tests/test-data/")
        filename_om_db = data_path / "ex-om-scan/om_scan_direct_beam.tif"
        filename_z_db = data_path / "ex-z-scan/z_scan_direct_beam.tif"
        db1 = DirectBeam(filename_om_db)
        db1.show_beam()



if __name__ == "__main__":
    from unittest import main
    main()
    plt.show()