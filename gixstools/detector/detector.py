import numpy as np
import gixstools.config
from scipy import ndimage
import pyFAI.detectors


def get_detector(detector_name: str) -> pyFAI.detectors._common.Detector:
    """
    Load the detector class from a string: i.e. "Eiger1M" -> pyFAI.detectors.Eiger1M
    """
    DetectorClass = getattr(pyFAI.detectors, detector_name, None)
    if DetectorClass is None:
        raise ValueError(f"Detector '{detector_name}' is not a valid detector type. Visit https://pyfai.readthedocs.io/en/stable/api/detectors/index.html#module-pyFAI.detectors")
    return DetectorClass

detector_config = gixstools.config.load("detector")

DetectorFromConfig = get_detector(detector_config["name"])

detector_shape = DetectorFromConfig().shape


class Detector(DetectorFromConfig):

    BAD_PIXELS = (np.array(detector_config['bad_pixels']['rows'], dtype=np.int64),   # rows
                  np.array(detector_config['bad_pixels']['columns'], dtype=np.int64))   # columns
    
    ISSUE_QUADS = (tuple(detector_config['issue_quads']['start_rows']),
                   tuple(detector_config['issue_quads']['start_columns']))
    
    ROWS = detector_shape[0]
    COLS = detector_shape[1]

    MAX_INT = 1 << detector_config["bit_size"] - 1

    def __init__(self):
        super().__init__(orientation=detector_config["orientation"])
    
    @classmethod
    def calc_mask_cls(cls) -> np.ndarray:
        mask = super().calc_mask()
        mask[cls.BAD_PIXELS] = 1
        for row_start_quad in cls.ISSUE_QUADS[0]:
            for col_start_quad in cls.ISSUE_QUADS[1]:
                mask[row_start_quad:(row_start_quad + 4), col_start_quad:(col_start_quad + 4)] = 1
        return mask
    
    def calc_mask(self) -> np.ndarray:
        mask = super().calc_mask()
        mask[self.BAD_PIXELS] = 1
        for row_start_quad in self.ISSUE_QUADS[0]:
            for col_start_quad in self.ISSUE_QUADS[1]:
                mask[row_start_quad:(row_start_quad + 4), col_start_quad:(col_start_quad + 4)] = 1
        return mask
    
    def calc_mask_and_outliers(self, image: np.ndarray, threshold: float = 100) -> np.ndarray:
        mask = self.calc_mask()
        outliers = self.find_outliers(image, threshold)
        return np.logical_and(mask, outliers)
    
    def calc_mask_dezinger(self, image: np.ndarray, cut_off: float=None,
                           gaussian_standard_deviation: float=None) -> np.ndarray:
        return np.logical_or(self.find_zingers(image, cut_off, gaussian_standard_deviation), self.calc_mask())
    
    @classmethod
    def get_size(cls) -> tuple:
        return (cls.ROWS, cls.COLS)
    
    @classmethod
    def get_rows(cls) -> int:
        return cls.ROWS
    
    @classmethod
    def get_columns(cls) -> int:
        return cls.COLS
    
    @staticmethod
    def find_outliers(image: np.ndarray, threshold: int):
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        neighbor_sums = ndimage.convolve(image, kernel, mode='constant', cval=0)
        above_mask = image > threshold
        low_neighbor_sum = neighbor_sums < 3
        outliers = np.logical_and(above_mask, low_neighbor_sum)
        return outliers
    
    @staticmethod
    def find_zingers(image: np.ndarray, cut_off: float=5, gaussian_standard_deviation: float=2) -> np.ndarray:
        """
        Finds abnormally hot pixels by applying a gaussian filter to smooth the image and locate pixels and returns a mask for them
        :params image: image to find zingers in
        :params cut_off: 
        :params gaussian_standard_deviations:
        :return: return mask of zinger locations
        """
        if cut_off is None:
            cut_off = 1000.
        if gaussian_standard_deviation is None:
            gaussian_standard_deviation = .5
        smoothed_img = ndimage.gaussian_filter(image, gaussian_standard_deviation)
        dif_img = image - smoothed_img

        zinger_chart = dif_img / (smoothed_img + 1)
        anomalies = zinger_chart > cut_off
        print(f'Found {np.sum(anomalies)} zingers')
        return anomalies.astype(bool)
        


if __name__ == "__main__":
    detector = Detector()
    print(detector)