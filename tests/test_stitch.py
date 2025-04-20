from unittest import TestCase
import gixstools.detector
import gixstools.config
from pathlib import Path
import numpy as np
import pyFAI.detectors

import gixstools.detector


class TestStitch(TestCase):
    def test_stitching(self):
        path = Path("tests/raw-images-to-stitch")

        config = gixstools.config.load("stitch")

        rows, columns, compensators = float('-inf'), float('-inf'), float('-inf')

        for file in path.glob("*"):
            r, c, s = map(int, file.stem.split(config["separator"]))  # Split the string and convert to integers
            rows = max(rows, r)
            columns = max(columns, c)
            compensators = max(compensators, s)
        
        raw_detector = gixstools.detector.Detector()

        expected_rows = raw_detector.get_rows() * rows + int(config["compensate"]) * config["missing_band_rows"] - config["image_overlap"] * (rows - 1)
        expected_columns = raw_detector.get_columns() * columns - config["image_overlap"] * (columns - 1)
        
        data, flat, stitched_detector = gixstools.detector.stitch(path)
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(flat, np.ndarray)
        self.assertIsInstance(stitched_detector, pyFAI.detectors.Detector)
        self.assertEqual(expected_columns, stitched_detector.shape[1],
                         "Resulting detector does not have the correct number of columns.")
        self.assertEqual(expected_rows, stitched_detector.shape[0],
                         "Resulting detector does not have the correct number of rows.")
        self.assertEqual((expected_rows, expected_columns), data.shape,
                         "Resulting stitch is not the right size.")
        self.assertEqual(data.shape, flat.shape,
                         "resulting flat field is not the right size.")




if __name__ == "__main__":
    from unittest import main
    main()
