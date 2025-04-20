from gixstools.detector import Detector
from pathlib import Path
import fabio
import numpy as np


def load_raw(filename: Path):
    detector = Detector()
    image = fabio.open(filename).data
    mask = detector.calc_mask()

    mask[file_data == detector.MAX_INT] = 1
    file_data *= np.logical_not(mask)
    return file_data


class RawLoader:
    def __init__(self):
        detector = Detector()
        self.base_mask = np.logical_not(detector.calc_mask())
        self.max_int = detector.MAX_INT
    
    def load(self, filename: Path):
        file_data= fabio.open(filename).data
        mask = self.base_mask.copy()
        mask[file_data == self.max_int] = 0
        return file_data * mask