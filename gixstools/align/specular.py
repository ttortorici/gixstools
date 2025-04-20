import matplotlib.offsetbox
import numpy as np
import fabio
import re
from gixstools.detector import Detector, RawLoader, load_raw
import gixstools.config
from scipy.optimize import curve_fit, root_scalar
from scipy.special import erf
from pathlib import Path
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
plt.style.use("gixstools.style")

_config = gixstools.config.load("align")
WHERE_INFO = _config["info"]
IM_TYPE = _config["im_filetype"]
BEAM_POS = _config["beam_position"]
UNIT = _config["unit"]
if UNIT == "m":
    UNIT_CONV = 1
elif UNIT == "mm":
    UNIT_CONV = 1e3
elif UNIT == "um":
    UNIT_CONV = 1e6


class SpecularScan:

    compressed_filename = "compressed_scan.npz"
    detector = Detector()

    def __init__(self, directory: Path, approximate_detector_distance_meters: float, beam_cut_in_widths: float = 1., max_angle_degrees: float = 1.):
        self.directory = directory
        self.dist_guess = approximate_detector_distance_meters

        self.z0 = 0
        self.data_remove = None # revisit
        self.type = None
        self.motor_positions = None
        self.z_positions = None
        self.specular_intensities = None
        self.all_images = None
        self.direct_beam = None

        self.beam_width = None
        self.beam_center = None

        self.beam_cut = beam_cut_in_widths

        """Determine if there is a compression"""
        if (directory / self.compressed_filename).is_file():
            self.load_compressed()
        else:
            raw = self.load_raw()
            self.process_data(raw, max_angle_degrees)
    
    def load_compressed(self):
        file_list = list(self.directory.glob("*" + IM_TYPE))
        db_ind = DirectBeam.find_file_index(file_list)
        try:
            db_file = file_list[db_ind]
            self.direct_beam = DirectBeam(db_file)
        except TypeError:
            pass
        data = np.load(self.directory / "compressed_scan.npz")
        self.motor_positions = data["motor_positions"]
        self.z_positions = data["z_positions"]
        self.specular_intensities = data["intensities"]
    
    def load_raw(self):
        loader = RawLoader()
        file_list = list(self.directory.glob("*" + IM_TYPE))
        db_ind = DirectBeam.find_file_index(file_list)
        db_file = file_list[db_ind]
        del(file_list[db_ind])
        self.direct_beam = DirectBeam(db_file)
        self.beam_center = self.direct_beam.center[0] * self.detector.pixel1 * UNIT_CONV
        self.beam_width = self.direct_beam.width[0] * self.detector.pixel1 * UNIT_CONV
        motor = np.empty_like(file_list, dtype=np.float64)
        full_data = np.empty((len(file_list), *self.detector.shape), dtype=np.float64)
        scan_types = [""] * len(file_list)

        for ii, file in enumerate(file_list):
            if "om" in file.name:
                scan_types[ii] = "om"
            elif "z" in file.name:
                scan_types[ii] = "z"
            motor[ii] = self.get_motor_position(file)
            full_data[ii] = loader.load(file)

        self.type = max(set(scan_types), key=scan_types.count)  # either 'om' or 'z'
        
        sorting_args = np.argsort(motor)
        self.motor_positions = motor = motor[sorting_args]
        intensity_data = full_data[sorting_args]
        return intensity_data

    def process_data(self, raw_data: np.ndarray, max_angle: float = 3.):
        self.z_positions = self.direct_beam.center[0] - np.arange(self.detector.shape[0])  # now positive direction is up

        crop_above = int(self.direct_beam.center[0] - self.dist_guess / self.detector.pixel1 * np.tan(2. * np.radians(max_angle)))
        crop_below = int(self.direct_beam.center[0] + self.direct_beam.width[0] * 5)
        crop_left  = int(self.direct_beam.center[1] - self.direct_beam.width[1] * 1.5)
        crop_right = int(self.direct_beam.center[1] + self.direct_beam.width[1] * 1.5)

        self.z_positions = self.z_positions[crop_above:crop_below] * self.detector.pixel1 * UNIT_CONV

        self.all_images = raw_data[:, crop_above:crop_below, crop_left:crop_right]
        
        self.intensity_specular = np.sum(self.all_images, axis=2).T

    def show_crops(self, image_indices: list, figsize: tuple = None):
        pos = [None] * len(image_indices)

        max_value = self.all_images[image_indices, :, :].max()

        fig, axes = plt.subplots(1, len(image_indices), figsize=figsize)
        for ii, (ax, ind) in enumerate(zip(axes, image_indices)):
            pos[ii] = self.plot_crop(ax, ind, max_value)
        cb = fig.colorbar(pos[-1], ax=axes.ravel().tolist(), fraction=0.0315, pad=.05, anchor=(1000,.5))
        cb.ax.tick_params(which="both", direction="out")
        if self.type == "om":
            # axes[int(len(image_indices) * 0.5)].set_title("$\\omega$-motor position")
            fig.suptitle("$\\omega$-motor position", fontsize=11, y=0.9)
        elif self.type == "z":
            # axes[int(len(image_indices) * 0.5)].set_title("$z$-motor position")
            fig.suptitle("$z$-motor position", fontsize=11, y=.9)
        return fig, axes

    def plot_crop(self, ax: matplotlib.axes._axes.Axes, im_ind: int, max_value: float = None):
        ax.set_facecolor("k")
        image = self.all_images[im_ind]
        if max_value is None:
            max_value = image.max()
        pos = ax.imshow(image, norm=LogNorm(1, max_value), aspect='equal')
        motor_position = self.motor_positions[im_ind]
        if self.type == "om":
            title = f"${motor_position:.2f}\\degree$"
        elif self.type == "z":
            title = f"{motor_position:.2f} mm"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return pos

    def get_motor_position(self, filename: Path) -> float:
        if WHERE_INFO.lower() == "title":
            angle = self.motor_angle_from_file_info(filename.name)
        elif WHERE_INFO.lower() == "header":
            comment = fabio.open(filename).header["Comment"]
            angle = float(re.search(r"[-+]?\d*\.\d+|[-+]?\d+", comment).group())
        else:
            raise ValueError("align.info must be 'header' or 'title' in config.toml")
        return angle

    @staticmethod
    def motor_angle_from_file_info(filename: str):
        left_of_decimal = filename.split("_")[-3]
        angle = float(left_of_decimal)
        right_of_decimal = filename.split("_")[-2].replace(".tif", "")
        if left_of_decimal[0] == "-":
            angle -= float(right_of_decimal) / 10. ** len(right_of_decimal)
        else:
            angle += float(right_of_decimal) / 10. ** len(right_of_decimal)
        angle = round(angle, 3)
        return angle

    def show_direct_beam(self, figsize=None, column=False):
        if column:
            fig, axes = self.direct_beam.plot_column(figsize=figsize)
        else:
            fig, axes = self.direct_beam.plot_quad(figsize=figsize)
        return fig, axes
    
    def save(self):
        data = {
            "motor_positions": self.motor_positions,
            "z_positions": self.z_positions,
            "intensities": self.specular_intensities
        }
        np.savez_compressed(self.directory / self.compressed_filename, **data)

    #############################

    def fit(self, z0=None, max_angle=1, pixel_cut=None, standard_deviations=None):
        if self.type == "om":
            self.fit_om(z0, max_angle, pixel_cut, standard_deviations)
            
            # if self.data_remove:
            #     self.refit_om()

        elif self.type == "z":
            self.fit_z(standard_deviations)
        else:
            raise AttributeError("Scan type was not established.")
        
    def fit_om(self, z0: float=None, max_angle: float = 1., pixel_cut: int=None, cut_in_widths: float=None):
        if z0 is not None:
            self.z0 = z0
        if cut_in_widths is not None:
            self.beam_cut = cut_in_widths

        self.counts_total = np.sum(self.intensity_specular, axis=0)
        self.counts_specular = np.sum(
            self.intensity_specular[np.where(self.z_positions > self.beam_width * self.beam_cut)],
            axis=0
        )
        self.counts_dbeam = np.sum(
            self.intensity_specular[np.where(self.z_positions < self.beam_width * self.beam_cut)],
            axis=0
        )
        # print(f"z\u2080 = {self.z0}")
        max_ind = np.argmax(self.intensity_specular, axis=1)
        where_max_angle = self.motor_positions[max_ind]
        
        print(self.z_positions)
        print(self.z0)
        print(self.dist_guess * np.radians(max_angle) * UNIT_CONV + self.z0)
        # Remove data below the beam center
        valid = np.where(np.logical_and(
            self.z_positions > self.z0,
            self.z_positions < self.dist_guess * np.radians(max_angle) * UNIT_CONV + self.z0
            ))
        self.z_valid = self.z_positions[valid]
        self.where_max_angle = where_max_angle[valid]
        if pixel_cut is not None:
            self.z_valid = self.z_valid[:-pixel_cut]
            self.where_max_angle = self.where_max_angle[:-pixel_cut]
        
        (self.omega0, self.det_dist_fit), pcov = curve_fit(self.specular_om_fit, self.where_max_angle, self.z_valid, p0=[0, self.dist_guess])
        self.perr = np.sqrt(np.diag(pcov))
        print("Fit results:")
        print(f"    \u03C9\u2080 = ({self.omega0} \u00B1 {self.perr[0]})\u00B0")
        print(f"    d\u209B = ({self.det_dist_fit} \u00B1 {self.perr[1]}) mm")
        
    def plot_om(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7))
        if self.plot_title:
            fig.suptitle(self.plot_title, fontsize=12)

    def plot_specular_fit(self, ax):
        """FIT THROUGH MAX COUNT IN EACH ROW"""
        ax1.scatter(self.where_max_angle, self.z_valid, s=10, marker='o',
                    edgecolors='k', lw=.75, facecolor='w')
        omega = np.linspace(self.where_max_angle[-1] - 0.02, self.where_max_angle[0] + 0.02, 100)
        ax1.plot(omega, self.specular_om_fit(omega, self.omega0, self.det_dist_fit), "r")
        
        ax1.set_ylabel("$z$ (mm)", fontsize=12)
        ax1.set_title("where max pixel occurs", fontsize=12)
        # ax.legend(title="pixel")
        annotation_text = f"$\\omega_0 = {self.omega0:.4f} \\pm {self.perr[0]:.4f}^\\circ$\n$d_{{sd}} = {self.det_dist_fit:.2f} \\pm {self.perr[1]:.2f}$ mm"
        ax1.text(omega.min(), 0.95 * self.max_angle, annotation_text, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        
    def plot_trad_om_scan(self, ax):
        """TRADITIONAL OM SCAN"""
        ax2.scatter(self.angles, self.counts_total,
                    s=10, marker='o', edgecolors='k', lw=.75, facecolor='w')
        ax2.set_title("Total Counts")

        ax3.scatter(self.angles, self.counts_specular,
                    s=10, marker='o', edgecolors='k', lw=.75, facecolor='w')
        ax3.set_title("Counts above beam")

        ax4.scatter(self.angles, self.counts_dbeam,
                    s=10, marker='o', edgecolors='k', lw=.75, facecolor='w')
        ax4.set_title("Counts in Direct Beam")
        
        for ax in (ax1, ax2, ax3, ax4):
            ax.tick_params(axis='both', which='both', direction='in', right=True, top=True)
            ax.set_xlabel("$\\omega$ motor position $(\\degree)$", fontsize=12)
            ax.grid(linestyle='dotted')
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        for ax in (ax2, ax3, ax4):
            ax.set_ylabel("Counts")
        fig.tight_layout()
        
        self.save_fig(fig)
    # def refit_om(self):
    #     difference = np.abs(self.z_valid - self.specular_om_fit(self.where_max_angle, self.omega0, self.det_dist_fit))
    #     keep_inds = np.sort(np.argsort(difference)[:len(difference) - self.data_remove])
    #     (self.omega0, self.det_dist_fit), pcov = curve_fit(self.specular_om_fit,
    #                                                        self.where_max_angle[keep_inds],
    #                                                        self.z_valid[keep_inds],
    #                                                        p0=[0, self.det_dist_fit])
    #     self.perr = np.sqrt(np.diag(pcov))
    #     print("Fit results:")
    #     print(f"    \u03C9\u2080 = ({self.omega0} \u00B1 {self.perr[0]})\u00B0")
    #     print(f"    d\u209B = ({self.det_dist_fit} \u00B1 {self.perr[1]}) mm")

    def fit_z(self, standard_deviations=None):
        if standard_deviations is not None:
            self.standard_deviations = standard_deviations
        self.counts_total = np.sum(self.intensity_specular.T, axis=0)
        self.counts_specular = np.sum(
            self.intensity_specular.T[np.where(self.z > self.standard_deviations * self.bc_sigma)],
            axis=0
        )
        self.counts_dbeam = np.sum(
            self.intensity_specular.T[np.where(self.z < self.standard_deviations * self.bc_sigma)],
            axis=0
        )

        print("Fit results:")
        fit_names = ["max", "z\u2080", "\u03C3\u2080"]
        unit_names = ["counts", "mm", "mm"]
        try:
            self.total_fit, pcov_total = curve_fit(self.occlusion_fit_single, self.z_motor, self.counts_total,
                                                   # p0=(1e6, 0.5, self.bc_sigma))
                                                   p0=(self.counts_total.max(), 0.5 * (self.z_motor.min() + self.z_motor.max()), self.bc_sigma))
            self.perr_total = np.sqrt(np.diag(pcov_total))
            print("  Total counts:")
            for name, fit_res, err, unit in zip(fit_names, self.total_fit, self.perr_total, unit_names):
                print(f"    {name} = ({fit_res:.5f} \u00B1 {err:.5f}) {unit}")
            self.success_total = True
        except RuntimeError:
            print("Failed to fit total counts")
            self.success_total = False

        try:
            self.dbeam_fit, pcov_dbeam = curve_fit(self.occlusion_fit_single, self.z_motor, self.counts_dbeam,
                                                   # p0=(1e6, 0.5, self.bc_sigma)
                                                   p0=(self.counts_dbeam.max(), 0.5 * (self.z_motor.min() + self.z_motor.max()), self.bc_sigma))
            self.perr_dbeam = np.sqrt(np.diag(pcov_dbeam))
            print("  Primary beam counts:")
            for name, fit_res, err, unit in zip(fit_names, self.dbeam_fit, self.perr_dbeam, unit_names):
                print(f"    {name} = ({fit_res:.5f} \u00B1 {err:.5f}) {unit}")
            self.success_dbeam = True
        except RuntimeError:
            print("Failed to fit direct beam counts")
            self.success_dbeam = False

        try:
            self.specz_fit, pcov_specz = curve_fit(self.specular_z_fit, self.z_motor, self.counts_specular,
                                                   # p0=(1e5, .4, .7, 0.5 * self.bc_sigma, 0.5 * self.bc_sigma))
                                                   p0=(self.counts_specular.max(), self.z_motor.min(), self.z_motor.max(), 0.5 * self.bc_sigma, 0.5 * self.bc_sigma))
            self.perr_specz = np.sqrt(np.diag(pcov_specz))
            fit_names = ["max", "z\u2081", "z\u2082", "\u03C3\u2081", "\u03C3\u2082"]
            unit_names = ["counts", "counts", "mm", "mm", "mm", "mm"]
            print("  Specular counts:")
            for name, fit_res, err, unit in zip(fit_names, self.specz_fit, self.perr_specz, unit_names):
                print(f"    {name} = ({fit_res:.5f} \u00B1 {err:.5f}) {unit}")
            self.success_specz = True
        except RuntimeError:
            print("Failed to fit specular counts")
            self.success_specz = False

        self.plot()

    def specular_om_fit(self, omega, omega0, det_dist):
        return det_dist * np.tan(2. * np.radians(omega - omega0)) + self.z0
    
    def specular_om_fit2(self, omega, omega0, det_dist, radial_offset, rotation_offset):
        alpha = np.radians(omega - omega0)
        vertical_offset = radial_offset * (np.sin(rotation_offset + alpha) - np.sin(rotation_offset))
        return (det_dist - radial_offset * np.cos(rotation_offset + alpha)) * np.tan(2. * alpha) + vertical_offset

    
    @staticmethod
    def specular_om_error(omega, omega0, det_dist, omega_err, dist_err):
        omega_center = omega - omega0
        dist_deriv = np.tan(2. * omega_center)
        sec_2omega = 1. / np.cos(2. * omega_center)
        omega_deriv = 2. * det_dist * sec_2omega * sec_2omega
        dist_term = dist_err * dist_deriv
        omega_term = omega_err * omega_deriv
        return np.sqrt(dist_term * dist_term + omega_term * omega_term)

    
    def specular_z_fit(self, z_motor, max_counts, z_lo, z_hi, sigma_lo, sigma_hi):
        return 0.5 * max_counts * (erf((z_motor - z_lo) / sigma_lo) - erf((z_motor - z_hi) / sigma_hi))
    
    def zero_angle(self, omega, omega0, det_dist):
        return det_dist * np.tan(np.radians(omega - omega0)) + self.z0
    
    def yoneda(self, omega, omega0, det_dist, critical_angle):
        return det_dist * np.tan(np.radians(omega - omega0 + critical_angle)) + self.z0
    
    def transmission(self, omega, omega0, det_dist, critical_angle):
        alpha = np.radians(omega - omega0)
        alpha_sq = alpha * alpha
        critical_angle = np.radians(critical_angle)
        crit_sq = critical_angle * critical_angle
        refraction_angle_sq = (alpha_sq - crit_sq) / (1 - 0.5 * crit_sq)
        return self.z0 + det_dist * np.tan(alpha - np.sqrt(refraction_angle_sq))
    
    def exiting_refraction(self, omega, omega0, det_dist, critical_angle):
        alpha = np.radians(omega - omega0)
        alpha_sq = alpha * alpha
        critical_angle = np.radians(critical_angle)
        crit_sq = critical_angle * critical_angle
        refraction_angle_sq = (alpha_sq - crit_sq) / (1 - 0.5 * crit_sq)
        return self.z0 + det_dist * np.tan(alpha + np.sqrt(refraction_angle_sq))
    



    


class SpecularScanOld:

    def __init__(self, data_directory: Path, det_dist: float=150, anglular_range: float=1.5,
                 beam_width: float=1, standard_deviations: float=4, data_remove: int = None,
                 beam_location: str = "LC", plot_name: str = "", plot_dpi: int = None, plot_title: str = None):
        self.plot_name = plot_name
        if plot_dpi is None:
            self.save_plots = False
        else:
            self.save_plots = True
        self.data_remove = data_remove
        self.beam_loc = beam_location.upper()
        self.plot_title = plot_title
        self.plot_dpi = plot_dpi
        self.type = None
        self.z0 = 0
        self.det_dist=det_dist
        self.beam_width = beam_width
        self.data_directory = data_directory
        self.instrument = None
        self.detector = None
        self.file_type = None
        self.beam_center = None
        self.bc_sigma = None
        self.bc_amp = None
        self.where_max_angle = None
        self.perr = None
        self.max_angle = None
        self.standard_deviations = standard_deviations
        self.z_specular = None
        self.counts_total = None
        self.success_total = False
        self.success_dbeam = False
        self.success_specz = False

        self.file_list = list(data_directory.glob("*" + self.file_type))

        self.pixel_size = self.detector.get_pixel1() * 1e3  # mm/pixel
        self.pixels_above = int(det_dist * np.radians(anglular_range) / self.pixel_size)   # number of pixels above the beam to keep (for crop)
        self.pixels_below = 8   # number of pixels below the beam to keep (for crop)
        self.pixel_horiontal_offset = 0
        self.z = np.arange(self.detector.shape[0])[::-1] * self.pixel_size
        
        self.base_mask = np.logical_not(self.detector.calc_mask())  # 1 keep, 0 mask (opposite of pyFAI standard, so it can be simply multiplied)

        self.angles = None
        self.z_motor = None
        self.intensity_data = None
        self.intensity_specular = None
        self.run()

    def save(self, directory: Path):
        if self.type == "om":
            data_to_save = {"angles": self.angles}
        elif self.type == "z":
            data_to_save = {"z_motor": self.z_motor}
        else:
            raise AttributeError("Has not determined type of specular scan.")
        data_to_save["z_pos"] = self.z
        data_to_save["specular_data"] = self.intensity_specular
        np.savez(directory / "compressed_scan.npz", **data_to_save)
    

    def run(self):
        direct_beam_file_index = self.find_direct_beam_file_index()

        if direct_beam_file_index is None:
            raise FileExistsError("Did not find a direct beam exposure")
        x_pos = self.beam_loc[1]
        y_pos = self.beam_loc[0]
        self.beam_center = self.find_beam_center(direct_beam_file_index, x_pos, y_pos)
        print("Direct beam is at ({}, {})".format(*self.beam_center))
        del(self.file_list[direct_beam_file_index])
        motor, self.intensity_data = self.load_raw()
            
        if self.type == "om":
            self.angles = motor
        elif self.type == "z":
            self.z_motor = motor
        else:
            raise AttributeError("Scan type not established")

        self.process_data()
        self.fit()

    def fit(self, z0=None, max_angle=1, pixel_cut=None, standard_deviations=None):
        if self.type == "om":
            self.fit_om(z0, max_angle, pixel_cut, standard_deviations)
            
            if self.data_remove:
                self.refit_om()

        elif self.type == "z":
            self.fit_z(standard_deviations)
        else:
            raise AttributeError("Scan type was not established.")
        
    def refit_om(self):
        difference = np.abs(self.z_valid - self.specular_om_fit(self.where_max_angle, self.omega0, self.det_dist_fit))
        keep_inds = np.sort(np.argsort(difference)[:len(difference) - self.data_remove])
        (self.omega0, self.det_dist_fit), pcov = curve_fit(self.specular_om_fit,
                                                           self.where_max_angle[keep_inds],
                                                           self.z_valid[keep_inds],
                                                           p0=[0, self.det_dist_fit])
        self.perr = np.sqrt(np.diag(pcov))
        print("Fit results:")
        print(f"    \u03C9\u2080 = ({self.omega0} \u00B1 {self.perr[0]})\u00B0")
        print(f"    d\u209B = ({self.det_dist_fit} \u00B1 {self.perr[1]}) mm")


    def fit_om(self, z0: float=None, max_angle: float=1, pixel_cut: int=None,
               standard_deviations: float=None):
        self.max_angle = max_angle
        if z0 is not None:
            self.z0 = z0
        if standard_deviations is not None:
            self.standard_deviations = standard_deviations

        self.counts_total = np.sum(self.intensity_specular.T, axis=0)
        self.counts_specular = np.sum(
            self.intensity_specular.T[np.where(self.z > self.standard_deviations * self.bc_sigma)],
            axis=0
        )
        self.counts_dbeam = np.sum(
            self.intensity_specular.T[np.where(self.z < self.standard_deviations * self.bc_sigma)],
            axis=0
        )
        
        print(f"z\u2080 = {self.z0}")
        max_ind = np.argmax(self.intensity_specular, axis=0)
        where_max_angle = self.angles[max_ind]

        # Remove data below the beam center
        valid = np.where(np.logical_and(
            self.z > self.z0,
            self.z < self.det_dist * np.radians(max_angle) + self.z0
            ))
        self.z_valid = self.z[valid]
        self.where_max_angle = where_max_angle[valid]
        if pixel_cut is not None:
            self.z_valid = self.z_valid[:-pixel_cut]
            self.where_max_angle = self.where_max_angle[:-pixel_cut]
        
        (self.omega0, self.det_dist_fit), pcov = curve_fit(self.specular_om_fit, self.where_max_angle, self.z_valid, p0=[0, self.det_dist])
        self.perr = np.sqrt(np.diag(pcov))
        print("Fit results:")
        print(f"    \u03C9\u2080 = ({self.omega0} \u00B1 {self.perr[0]})\u00B0")
        print(f"    d\u209B = ({self.det_dist_fit} \u00B1 {self.perr[1]}) mm")

    def fit_z(self, standard_deviations=None):
        if standard_deviations is not None:
            self.standard_deviations = standard_deviations
        self.counts_total = np.sum(self.intensity_specular.T, axis=0)
        self.counts_specular = np.sum(
            self.intensity_specular.T[np.where(self.z > self.standard_deviations * self.bc_sigma)],
            axis=0
        )
        self.counts_dbeam = np.sum(
            self.intensity_specular.T[np.where(self.z < self.standard_deviations * self.bc_sigma)],
            axis=0
        )

        print("Fit results:")
        fit_names = ["max", "z\u2080", "\u03C3\u2080"]
        unit_names = ["counts", "mm", "mm"]
        try:
            self.total_fit, pcov_total = curve_fit(self.occlusion_fit_single, self.z_motor, self.counts_total,
                                                   # p0=(1e6, 0.5, self.bc_sigma))
                                                   p0=(self.counts_total.max(), 0.5 * (self.z_motor.min() + self.z_motor.max()), self.bc_sigma))
            self.perr_total = np.sqrt(np.diag(pcov_total))
            print("  Total counts:")
            for name, fit_res, err, unit in zip(fit_names, self.total_fit, self.perr_total, unit_names):
                print(f"    {name} = ({fit_res:.5f} \u00B1 {err:.5f}) {unit}")
            self.success_total = True
        except RuntimeError:
            print("Failed to fit total counts")
            self.success_total = False

        try:
            self.dbeam_fit, pcov_dbeam = curve_fit(self.occlusion_fit_single, self.z_motor, self.counts_dbeam,
                                                   # p0=(1e6, 0.5, self.bc_sigma)
                                                   p0=(self.counts_dbeam.max(), 0.5 * (self.z_motor.min() + self.z_motor.max()), self.bc_sigma))
            self.perr_dbeam = np.sqrt(np.diag(pcov_dbeam))
            print("  Primary beam counts:")
            for name, fit_res, err, unit in zip(fit_names, self.dbeam_fit, self.perr_dbeam, unit_names):
                print(f"    {name} = ({fit_res:.5f} \u00B1 {err:.5f}) {unit}")
            self.success_dbeam = True
        except RuntimeError:
            print("Failed to fit direct beam counts")
            self.success_dbeam = False

        try:
            self.specz_fit, pcov_specz = curve_fit(self.specular_z_fit, self.z_motor, self.counts_specular,
                                                   # p0=(1e5, .4, .7, 0.5 * self.bc_sigma, 0.5 * self.bc_sigma))
                                                   p0=(self.counts_specular.max(), self.z_motor.min(), self.z_motor.max(), 0.5 * self.bc_sigma, 0.5 * self.bc_sigma))
            self.perr_specz = np.sqrt(np.diag(pcov_specz))
            fit_names = ["max", "z\u2081", "z\u2082", "\u03C3\u2081", "\u03C3\u2082"]
            unit_names = ["counts", "counts", "mm", "mm", "mm", "mm"]
            print("  Specular counts:")
            for name, fit_res, err, unit in zip(fit_names, self.specz_fit, self.perr_specz, unit_names):
                print(f"    {name} = ({fit_res:.5f} \u00B1 {err:.5f}) {unit}")
            self.success_specz = True
        except RuntimeError:
            print("Failed to fit specular counts")
            self.success_specz = False

        self.plot()

    def specular_om_fit(self, omega, omega0, det_dist):
        return det_dist * np.tan(2. * np.radians(omega - omega0)) + self.z0
    
    def specular_om_fit2(self, omega, omega0, det_dist, radial_offset, rotation_offset):
        alpha = np.radians(omega - omega0)
        vertical_offset = radial_offset * (np.sin(rotation_offset + alpha) - np.sin(rotation_offset))
        return (det_dist - radial_offset * np.cos(rotation_offset + alpha)) * np.tan(2. * alpha) + vertical_offset

    
    @staticmethod
    def specular_om_error(omega, omega0, det_dist, omega_err, dist_err):
        omega_center = omega - omega0
        dist_deriv = np.tan(2. * omega_center)
        sec_2omega = 1. / np.cos(2. * omega_center)
        omega_deriv = 2. * det_dist * sec_2omega * sec_2omega
        dist_term = dist_err * dist_deriv
        omega_term = omega_err * omega_deriv
        return np.sqrt(dist_term * dist_term + omega_term * omega_term)

    
    def specular_z_fit(self, z_motor, max_counts, z_lo, z_hi, sigma_lo, sigma_hi):
        return 0.5 * max_counts * (erf((z_motor - z_lo) / sigma_lo) - erf((z_motor - z_hi) / sigma_hi))
    
    def zero_angle(self, omega, omega0, det_dist):
        return det_dist * np.tan(np.radians(omega - omega0)) + self.z0
    
    def yoneda(self, omega, omega0, det_dist, critical_angle):
        return det_dist * np.tan(np.radians(omega - omega0 + critical_angle)) + self.z0
    
    def transmission(self, omega, omega0, det_dist, critical_angle):
        alpha = np.radians(omega - omega0)
        alpha_sq = alpha * alpha
        critical_angle = np.radians(critical_angle)
        crit_sq = critical_angle * critical_angle
        refraction_angle_sq = (alpha_sq - crit_sq) / (1 - 0.5 * crit_sq)
        return self.z0 + det_dist * np.tan(alpha - np.sqrt(refraction_angle_sq))
    
    def exiting_refraction(self, omega, omega0, det_dist, critical_angle):
        alpha = np.radians(omega - omega0)
        alpha_sq = alpha * alpha
        critical_angle = np.radians(critical_angle)
        crit_sq = critical_angle * critical_angle
        refraction_angle_sq = (alpha_sq - crit_sq) / (1 - 0.5 * crit_sq)
        return self.z0 + det_dist * np.tan(alpha + np.sqrt(refraction_angle_sq))
    
    def plot(self, title="", critical_angle=None, horizon=False, det_dist=None, omega0=None):
        if self.type == "om":
            self.plot_specular(title, critical_angle, horizon, det_dist, omega0)
        elif self.type == "z":
            self.plot_z()
        else:
            raise AttributeError("Type of specular not determined")
        
    def plot_om(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7))
        if self.plot_title:
            fig.suptitle(self.plot_title, fontsize=12)

        """FIT THROUGH MAX COUNT IN EACH ROW"""
        ax1.scatter(self.where_max_angle, self.z_valid, s=10, marker='o',
                    edgecolors='k', lw=.75, facecolor='w')
        omega = np.linspace(self.where_max_angle[-1] - 0.02, self.where_max_angle[0] + 0.02, 100)
        ax1.plot(omega, self.specular_om_fit(omega, self.omega0, self.det_dist_fit), "r")
        
        ax1.set_ylabel("$z$ (mm)", fontsize=12)
        ax1.set_title("where max pixel occurs", fontsize=12)
        # ax.legend(title="pixel")
        annotation_text = f"$\\omega_0 = {self.omega0:.4f} \\pm {self.perr[0]:.4f}^\\circ$\n$d_{{sd}} = {self.det_dist_fit:.2f} \\pm {self.perr[1]:.2f}$ mm"
        ax1.text(omega.min(), 0.95 * self.max_angle, annotation_text, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        
        """TRADITIONAL OM SCAN"""
        ax2.scatter(self.angles, self.counts_total,
                    s=10, marker='o', edgecolors='k', lw=.75, facecolor='w')
        ax2.set_title("Total Counts")

        ax3.scatter(self.angles, self.counts_specular,
                    s=10, marker='o', edgecolors='k', lw=.75, facecolor='w')
        ax3.set_title("Counts above beam")

        ax4.scatter(self.angles, self.counts_dbeam,
                    s=10, marker='o', edgecolors='k', lw=.75, facecolor='w')
        ax4.set_title("Counts in Direct Beam")
        
        for ax in (ax1, ax2, ax3, ax4):
            ax.tick_params(axis='both', which='both', direction='in', right=True, top=True)
            ax.set_xlabel("$\\omega$ motor position $(\\degree)$", fontsize=12)
            ax.grid(linestyle='dotted')
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        for ax in (ax2, ax3, ax4):
            ax.set_ylabel("Counts")
        fig.tight_layout()
        
        self.save_fig(fig)

    def plot_specular(self, title="", critical_angle=None, horizon=False, det_dist=None, omega0=None):
        if det_dist is None:
            det_dist = self.det_dist_fit
        if omega0 is None:
            omega0 = self.omega0

        """SPECULAR"""

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.set_facecolor('k')
        if self.plot_title:
            fig.suptitle(self.plot_title, fontsize=12)

        # z = np.arange(intensity_x_total.shape[1])[::-1] * self.pixel_size
        ax.set_ylabel("$z$ (mm)", fontsize=12)

        color_map = ax.pcolormesh(self.angles, self.z, self.intensity_specular.T,
                                  norm=LogNorm(1, self.intensity_specular.max()), cmap="plasma")
        
        ax.axhline(self.z0, linestyle="--", color="#FF5349", linewidth=0.7)
        ax.axhline(self.standard_deviations * self.bc_sigma,
                   color="w", linewidth=0.5, linestyle="--")

        omega2 = np.linspace(omega0, omega0 + .75, 1000)
        ax.plot(omega2, self.specular_om_fit(omega2, omega0, det_dist), "white", linewidth=1, alpha=0.5)
        if horizon:
            ax.plot(omega2, self.zero_angle(omega2, omega0, det_dist), "white", linewidth=1, alpha=0.5)
        if critical_angle is not None:
            if isinstance(critical_angle, float):
                critical_angle = [critical_angle]
            last = None
            for crit in critical_angle:
                if crit == last:
                    omega1 = np.linspace(omega0 + crit, omega0 + crit + .05, 100)
                    ax.plot(omega1, self.exiting_refraction(omega1, omega0, det_dist, crit), "white", linewidth=1, alpha=0.5)
                else:
                    ax.plot(omega2, self.yoneda(omega2, omega0, det_dist, crit), "white", linewidth=1, alpha=0.5)
                    omega1 = np.linspace(omega0 + crit, self.angles[-1], 1000)
                    ax.plot(omega1, self.transmission(omega1, omega0, det_dist, crit), "white", linewidth=1, alpha=0.5)
                    omega1 = np.linspace(self.angles[0], omega0 + crit, 1000)
                    ax.plot(omega1, self.exiting_refraction(omega1, omega0, det_dist, crit), "white", linewidth=1, alpha=0.5)
                    
                    # spec_at = self.specular_fit(crit + self.omega0, self.omega0, self.det_dist_fit)
                    # ax.plot([crit + self.omega0, crit + self.omega0], [spec_at + 0.6, spec_at + 1], "white", linewidth=.5, alpha=0.5)
                last = crit
        color_bar = fig.colorbar(color_map, ax=ax)

        ax.set_xlim(self.angles.min(), self.angles.max())

        ax.set_xlabel("$\\omega\\ (\\degree)$", fontsize=12)
        ax.set_ylabel("$z$ (mm)", fontsize=12)
        ax.set_title(title)
        fig.tight_layout()
        self.save_fig(fig)
        return fig, ax
    
    def plot_z(self, figsize=(10, 7)):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        ax1.set_facecolor('k')
        if self.plot_title:
            fig.suptitle(self.plot_title, fontsize=12)

        z_m_fit = np.linspace(self.z_motor.min(), self.z_motor.max(), 500)

        annotation_texts = []
        locs = []
        fitted_axes = []

        # z = np.arange(intensity_x_total.shape[1])[::-1] * self.pixel_size
        ax1.set_ylabel("$z$ (mm)", fontsize=12)
        ax1.set_ylabel("$z$-motor (mm)", fontsize=12)
        #ax1.set_xticks(list(self.z_motor))
        color_map = ax1.pcolormesh(self.z_motor, self.z, self.intensity_specular.T,
                                   norm=LogNorm(1, self.intensity_specular.max()), cmap="plasma")
        color_bar = fig.colorbar(color_map, ax=ax1)
        ax1.axhline(self.standard_deviations * self.bc_sigma,
                    color="w", linewidth=0.5)

        # fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3))
        ax2.scatter(self.z_motor, self.counts_total,
                   s=50,  # marker size
                   marker="o",  # marker shape
                   edgecolors="black",  # marker edge color
                   lw=2,  # marker edge width
                   alpha=1,  # transparency
                   facecolor='w')
        if self.success_total:
            ax2.plot(z_m_fit, self.occlusion_fit_single(z_m_fit, *self.total_fit), "r")
            annotation_texts.append(f"$z_0 = ({self.total_fit[1]:.3f} \\pm {self.perr_total[1]:.3f})$ mm")
            locs.append("lower left")
            fitted_axes.append(ax2)
        ax2.set_title("Total", fontsize=12)

        ax3.scatter(self.z_motor, self.counts_specular,
                   s=50,  # marker size
                   marker="o",  # marker shape
                   edgecolors="black",  # marker edge color
                   lw=2,  # marker edge width
                   alpha=1,  # transparency
                   facecolor='w')
        if self.success_specz:
            ax3.plot(z_m_fit, self.specular_z_fit(z_m_fit, *self.specz_fit), "r")
            annotation_texts.append(f"$z_2 = ({self.specz_fit[1]:.3f} \\pm {self.perr_specz[1]:.3f})$ mm\n$z_1 = ({self.specz_fit[2]:.3f} \\pm {self.perr_specz[2]:.3f})$ mm\n$z_{{ave}}={0.5*(self.specz_fit[1]+self.specz_fit[2]):.3f}$ mm")
            locs.append("lower center")
            fitted_axes.append(ax3)
        ax3.set_title("Above Beam", fontsize=12)
        
        ax4.scatter(self.z_motor, self.counts_dbeam,
                   s=50,  # marker size
                   marker="o",  # marker shape
                   edgecolors="black",  # marker edge color
                   lw=2,  # marker edge width
                   alpha=1,  # transparency
                   facecolor='w')
        if self.success_dbeam:
            ax4.plot(z_m_fit, self.occlusion_fit_single(z_m_fit, *self.dbeam_fit), "r")
            annotation_texts.append(f"$z_0 = ({self.dbeam_fit[1]:.3f} \\pm {self.perr_dbeam[1]:.3f})$ mm")
            locs.append("lower left")
            fitted_axes.append(ax4)
        ax4.set_title("Direct Beam", fontsize=12)
        
        for ax, ann, loc in zip(fitted_axes, annotation_texts, locs):
            ax.tick_params(axis='both', which='both', direction='in', right=True, top=True)
            ax.set_xlabel("$z$-motor (mm)", fontsize=12)
            ax.set_ylabel("Counts", fontsize=12)
            ax.grid(linestyle='dotted')
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            anchored_text = matplotlib.offsetbox.AnchoredText(ann, loc=loc, prop=dict(size=12), frameon=True)
            anchored_text.patch.set_boxstyle("round,pad=0.5")
            anchored_text.patch.set_facecolor('white')
            anchored_text.patch.set_alpha(0.8)
            ax.add_artist(anchored_text)
        fig.suptitle("Specular z-scan")
        fig.tight_layout()
        self.save_fig(fig)
        return fig, ((ax1, ax2), (ax3, ax4))
    
    def animate_tiffs(self, fps: float = 6.0, scale: float = 1):
        scale *= 0.2
        bc_z = round(self.beam_center[0])
        z_lo = bc_z - self.pixels_above
        z_hi = bc_z + self.pixels_below
        self.z = self.z[z_lo:z_hi]
        if self.beam_width is None:
            data = self.intensity_data[:, z_lo:z_hi, :]
        else:
            half_width = 0.5 * self.beam_width / self.pixel_size
            x_lo = round(self.beam_center[1] - half_width)
            x_hi = round(self.beam_center[1] + half_width)
            x_lo += self.pixel_horiontal_offset
            x_hi += self.pixel_horiontal_offset
            data = self.intensity_data[:, z_lo:z_hi, x_lo:x_hi]

        fig, ax = plt.subplots(1, 1, figsize=(scale * (x_hi - x_lo), scale * 0.5 * (z_hi - z_lo)))
        fig.tight_layout()
        ax.set_facecolor("k")

        frame_delay = 1000. / fps

        im = ax.imshow(data[0], norm=LogNorm(1, data.max()))
        cbar = fig.colorbar(im, ax=ax)

        def _init():
            im.set_data(data[0])
            return im,

        def _update(ii):
            im.set_array(data[ii])
            return im,

        ani = FuncAnimation(fig, _update, frames=data.shape[0],
                            init_func=_init, interval=frame_delay, blit=True)

        return fig, ani

    def occlusion_fit_single(self, z_motor, max_counts, z0, sigma):
        return 0.5 * (max_counts - max_counts * erf((z_motor - z0) / sigma))

    def occlusion_fit_double(self, z_motor, max_counts, z_lo, z_hi, sigma_lo, sigma_hi):
        return 0.5 * max_counts * (erf((z_motor - z_hi) / sigma_hi) - erf((z_motor - z_lo) / sigma_lo)) + max_counts




class DirectBeam:
    """
    The DirectBeam class processes a direct beam image to locate the center of the beam in millimeters.

    Attributes:
        detector (Detector): An instance of the Detector class used for processing the image.
        data (np.ndarray): The processed image data with masking applied.
        center (tuple): The (y, x) coordinates of the beam center in millimeters.
        width (tuple): The standard deviation (sigma) of the beam profile in the vertical and horizontal directions.
        amplitude (tuple): The amplitude of the beam profile in the vertical and horizontal directions.
    """

    detector = Detector()

    def __init__(self, filename: Path):
        """
        Initializes the DirectBeam instance by loading and processing the direct beam image.

        Args:
            filename (Path): The path to the direct beam image file.
        """
        image = fabio.open(filename).data
        mask = self.detector.calc_mask()
        mask[image == self.detector.MAX_INT] = 1

        self.data = image * np.logical_not(mask)
        
        self.x = np.arange(self.detector.shape[1])
        self.y = np.arange(self.detector.shape[0])
        self.counts_x = np.sum(self.data, axis=0)
        self.counts_y = np.sum(self.data, axis=1)
        self.center = None
        self.width = None
        self.amplitude = None
        
        self.find_center()
    
    def find_center(self) -> tuple:
        x_pos, y_pos = self.find_start_guess(self.x.max(), self.y.max())
        
        (x0, sigma_x, a_x), _, = curve_fit(self.gaussian, self.x, self.counts_x,
                                           p0=(x_pos, 100, self.counts_x.max()),
                                           nan_policy="omit")
        self.double_result_x, _ = curve_fit(self.double_erf, self.x, self.counts_x,
                                            p0=(x0, sigma_x, sigma_x * 0.2, a_x), nan_policy="omit")
        (y0, sigma_y, a_y), _ = curve_fit(self.gaussian, self.y, self.counts_y,
                                           p0=(y_pos, 100, self.counts_y.max()),
                                           nan_policy="omit")
        self.double_result_y, _ = curve_fit(self.double_erf, self.y, self.counts_y,
                                            p0=(y0, sigma_y, sigma_y * 0.2, a_y), nan_policy="omit")
        self.center = (self.double_result_y[0], self.double_result_x[0])
        self.width = (abs(self.double_result_y[1]), abs(self.double_result_x[1]))

    def plot_profile(self, ax, index, xrange):
        ax.scatter(self.x, self.counts_x,
                   s=10, marker='o', edgecolors='k', lw=.75, facecolor='w')
        ax.plot(self.x , self.gaussian(self.x, self.center[index], self.width[index], self.amplitude[index]), "r")
        ax.set_xlim(self.center[index] - xrange, self.center[index] + xrange)
        ax.set_ylabel("Counts")
        ax.grid()

    def plot_horizontal_profile(self, ax):
        xrange = int(self.width[1] * 1.5)
        ax.set_title("Horizontal beam profile")
        ax.set_xlabel("column")
        ax.scatter(self.x, self.counts_x,
                   s=10, marker='o', edgecolors='k', lw=.75, facecolor='w')
        x_fit = np.linspace(self.center[1] - xrange, self.center[1] + xrange, 1000)
        ax.plot(x_fit, self.double_erf(x_fit, *self.double_result_x), "r")
        ax.set_xlim(self.center[1] - xrange, self.center[1] + xrange)
        ax.set_ylabel("Counts")
        ax.grid()

    def plot_vertical_profile(self, ax, flip=True):
        yrange = int(self.width[1] * 1.5)
        ax.set_title("Vertical beam profile")
        if flip:
            ax.set_xlabel("Counts")
            ax.set_ylabel("row")
            y = self.y
            y_fit = np.linspace(self.center[0] - yrange, self.center[0] + yrange, 1000)
            x = self.counts_y
            x_fit = self.double_erf(y_fit, *self.double_result_y)
            ax.set_ylim(self.center[0] - yrange, self.center[0] + yrange)
        else:
            x = self.y
            x_fit = np.linspace(self.center[0] - yrange, self.center[0] + yrange, 1000)
            y = self.counts_y
            y_fit = self.double_erf(x_fit, *self.double_result_y)
            ax.set_ylabel("Counts")
            ax.set_xlabel("row")
            ax.set_xlim(self.center[0] - yrange, self.center[0] + yrange)
        ax.scatter(x, y, s=10, marker='o', edgecolors='k', lw=.75, facecolor='w')
        ax.plot(x_fit , y_fit, "r")
        ax.grid()

    def plot_beam(self, ax):
        ax.set_facecolor('k')
        pos = ax.imshow(self.data, norm=LogNorm(1, self.data.max()))
        xyrange = int(self.width[1] * 1.5)
        ax.set_xlim(self.center[1] - xyrange, self.center[1] + xyrange)
        ax.set_ylim(self.center[0] + xyrange, self.center[0] - xyrange)
        ax.axvline(self.center[1], color='r', linewidth=0.5)
        ax.axhline(self.center[0], color='r', linewidth=0.5)
        ax.tick_params(which='both', axis='both', colors='white', labelcolor='black')
        left = (self.center[1] - 0.5 * np.ones(2) * self.width[1], self.center[0] - 0.5 * np.array([-self.width[0], self.width[0]]))
        right = (self.center[1] + 0.5 * np.ones(2) * self.width[1], self.center[0] - 0.5 * np.array([-self.width[0], self.width[0]]))
        top = (self.center[1] - 0.5 * np.array([-self.width[1], self.width[1]]), self.center[0] + 0.5 * np.ones(2) * self.width[0])
        bottom = (self.center[1] - 0.5 * np.array([-self.width[1], self.width[1]]), self.center[0] - 0.5 * np.ones(2) * self.width[0])
        lw = 0.5
        ax.plot(*left, "r", lw=lw)
        ax.plot(*right, "r", lw=lw)
        ax.plot(*top, "r", lw=lw)
        ax.plot(*bottom, "r", lw=lw)
        return pos
        
    def plot_quad(self, figsize=None):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.delaxes(ax4)
        fig.suptitle("Direct Beam", fontsize=12)

        self.plot_horizontal_profile(ax3)
        self.plot_vertical_profile(ax2, flip=True)
        
        pos = self.plot_beam(ax1)
        
        cb = fig.colorbar(pos, ax=ax1)
        cb.ax.tick_params(which="both", direction="out")
        fig.tight_layout()
        return fig, ((ax1, ax2), (ax3, ax4))
    
    def plot_column(self, figsize=(3.37, 10)):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
        pos = self.plot_beam(ax1)
        cb = fig.colorbar(pos, ax=ax1)
        cb.ax.tick_params(which="both", direction="out")
        self.plot_horizontal_profile(ax2)
        self.plot_vertical_profile(ax3, flip=False)
        fig.tight_layout()
        return fig, (ax1, ax2, ax3)

    def show_beam(self, figsize=(3.37, 2.75)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        pos = self.plot_beam(ax)
        fig.colorbar(pos, ax=ax)
        fig.tight_layout()
        return fig, ax

    
    @staticmethod
    def find_start_guess(width: float, height: float):
        if BEAM_POS["x"] not in ["L", "left", "C", "center", "R", "right"]:
            raise ValueError("align.beam_position.x must be 'L', 'C', 'R', 'left', 'center', or 'right' in config.")
        if BEAM_POS["y"] not in ["U", "upper", "C", "center", "L", "lower"]:
            raise ValueError("align.beam_position.y must be 'U', 'C', 'L', 'upper', 'center', or 'lower' in config.")
        if BEAM_POS["x"] in ["L", "left"]:
            x_pos = 0.25 * width
        elif BEAM_POS["x"] in ["C", "center"]:
            x_pos = 0.5 * width
        elif BEAM_POS["x"] in ["R", "right"]:
            x_pos = 0.75 * width
        if BEAM_POS["y"] in ["U", "upper"]:
            y_pos = 0.25 * height
        elif BEAM_POS["y"] in ["C", "center"]:
            y_pos = 0.5 * height
        elif BEAM_POS["y"] in ["L", "lower"]:
            y_pos = 0.75 * height
        return x_pos, y_pos
    
    @staticmethod
    def gaussian(x, x0, sigma, amplitude):
        arg = (x - x0) / sigma
        return amplitude * np.exp(-0.5 * arg * arg)
    
    @staticmethod
    def double_erf(x, center, width, sigma, amplitude):
        half_width = 0.5 * width
        return amplitude * 0.5 * (erf((x - center + half_width) / sigma) - erf((x - center - half_width) / sigma))

    @classmethod
    def find_file_index(cls, file_list):
        if WHERE_INFO.lower() == "title":
            for ii, file in enumerate(file_list):
                if "direct_beam" in file.stem:
                    return ii
        elif WHERE_INFO.lower() == "header":
            for ii, file in enumerate(file_list):
                if "direct_beam" in fabio.open(file).header["Comment"] or "direct beam" in fabio.open(file).header["Comment"]:
                    return ii
        print("Did not find a direct beam file")
        return None
    

if __name__ == "__main__":
    data_path = Path("tests/test-data/")
    om_path = data_path / "ex-om-scan"
    z_path = data_path / "ex-z-scan"
    spec = SpecularScan(om_path, .150)
    spec.fit()
    plt.show()