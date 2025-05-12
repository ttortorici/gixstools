import numpy as np
import gixstools.config
from gixstools.detector import Detector
from pathlib import Path
import fabio
import pyFAI.detectors

stitch_config = gixstools.config.load("stitch")

def stitch(directory: Path) -> tuple:
    """
    Stitch images in a directory. Determined by config file.

    :param directory: path to raw images.
    :return: data:       stitched data.
             flat_field: relative exposure time of each pixel in stitched image.
             detector:   detector object representing stitched data information.
    """
    stitcher = Stitcher()
    data, flat_field, detector = stitcher.stitch(directory)
    return data, flat_field, detector


class Stitcher:

    OVERLAP = stitch_config["image_overlap"]
    MISSING_BAND_PIXELS = stitch_config["missing_band_rows"]
    WEIGHT_PER = stitch_config["exposure_per_image"]
    SEP = stitch_config["separator"]
    LABEL = stitch_config["raw_label"]
    EXT = stitch_config["file_extension"]
    
    def __init__(self, rows: int = 0, columns: int = 0, compensate: bool = stitch_config["compensate"]):
        """
        Define the stitch by the rows and columns and if the detector will be moved to compensate for the missing band.
        If rows=0 and columns=0 is given, it will look for all the raw images present in the directory to stitch.

        :params rows: number of stitching rows.
        :params columns: number of stitching columns.
        :params compensate: Will the detector be moved to compensate for a missing band of pixels?
        """
        self.detector = Detector()
        self.stitch_rows = rows
        self.stitch_columns = columns
        self.compensate = compensate
        if compensate:
            self.format = f"{self.LABEL}*{self.SEP}*{self.SEP}*{self.EXT}" # * are the numbers
        else:
            self.format = f"{self.LABEL}*{self.SEP}*{self.EXT}" # * are the numbers
        self.shape = (0, 0)
        
    def determine_shape(self):
        """
        Determine the shape of the stitched image.
        """
        rows = self.stitch_rows * (self.detector.get_rows() + Stitcher.MISSING_BAND_PIXELS - Stitcher.OVERLAP) + Stitcher.OVERLAP
        columns = self.stitch_columns * (self.detector.get_columns() - Stitcher.OVERLAP) + Stitcher.OVERLAP
        self.shape = (rows, columns)
        print(f"Stitched image will have shape: ({self.shape[0]}, {self.shape[1]}).")

    def stitch(self, directory: Path, dezinger: bool=False, cut_off: float=None, gaussian_standard_deviation: float=None):
        """
        Load data from directory based on self.rows and self.columns
        """
        print(f"Loading raw images from: {directory.as_posix()}\n")
        if not (self.stitch_rows and self.stitch_columns):  # if either of these are 0
            print("Will look for images to determine stitching size\n")
            # Find stitch size by using all files present
            for file in directory.glob(self.format):
                print(f"found: {file.name}")
                try:
                    name = file.stem.lstrip(self.LABEL)
                    stitch_row, stitch_col, _ = name.split(self.SEP)
                    stitch_row = int(stitch_row)
                    stitch_col = int(stitch_col)
                    print(f" - row: {stitch_row}; col: {stitch_col}")
                    if stitch_row > self.stitch_rows:
                        self.stitch_rows = stitch_row
                    if stitch_col > self.stitch_columns:
                        self.stitch_columns = stitch_col
                except ValueError:
                    print(" - not a valid raw image for stitching")
        stitch_description = f"Stitching {self.stitch_rows} rows and {self.stitch_columns} columns"
        if self.compensate:
            stitch_description += ", with compensation for empty band,"
        stitch_description += " for new image."
        print(stitch_description)
        self.determine_shape()

        data = np.zeros(self.shape, dtype=np.float64)
        flat_field = np.zeros(self.shape, dtype=np.float64)
        base_mask = np.logical_not(self.detector.calc_mask())

        comp_range = int(self.compensate) + 1
        
        for stitch_row in range(1, self.stitch_rows + 1):
            for stitch_col in range(1, self.stitch_columns + 1):
                for stitch_offset in range(1, comp_range + 1):
                    raw_image_file_name = f"{self.LABEL}{stitch_row}{self.SEP}{stitch_col}"
                    if self.compensate:
                        raw_image_file_name += f"{self.SEP}{stitch_offset}"
                    raw_image_file_name += self.EXT
                    file_data = fabio.open(directory / raw_image_file_name).data
                    if dezinger:
                        mask = np.logical_not(self.detector.calc_mask_dezinger(file_data, cut_off, gaussian_standard_deviation))
                    else:
                        mask = base_mask.copy()                
                    mask = np.logical_not(self.detector.calc_mask())
                    mask[file_data >= self.detector.MAX_INT] = 0
                    file_data *= mask
                    start_row = (self.stitch_rows - stitch_row) * (self.detector.ROWS - Stitcher.OVERLAP)
                    if self.compensate:
                        start_row += (2 - stitch_offset) * Stitcher.MISSING_BAND_PIXELS
                    start_column = (stitch_col - 1) * (self.detector.COLS - Stitcher.OVERLAP)
                    data[start_row:(start_row + self.detector.ROWS), start_column:(start_column + self.detector.COLS)] += file_data
                    flat_field[start_row:(start_row + self.detector.ROWS), start_column:(start_column + self.detector.COLS)] += mask
        flat_field *= self.WEIGHT_PER

        detector = pyFAI.detectors.Detector(pixel1=self.detector.pixel1,
                                            pixel2=self.detector.pixel2,
                                            max_shape=data.shape,
                                            orientation=int(self.detector.orientation))
        return data, flat_field, detector
    