from matplotlib.axes._axes import Axes
import numpy as np
import fabio
import re
from gixstools.detector import Detector, RawLoader, load_raw
import gixstools.config
from scipy.optimize import curve_fit, root_scalar
from scipy.special import erf
from pathlib import Path
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
# from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker
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

cmap = cmap = LinearSegmentedColormap.from_list("my_cmap", [
    "#2C0066", "#830B87", "#0031B8", "#3DCEDA", "#00FF0D", "#207800",
    #"#E0D900",
    "#C86B00", "#EB0800", "#FF5CEC", "#FFFF75", "#FFFFFF"
]) # 'terrain' # 'gist_ncar' # 'nipy_spectral'


class SpatiallyResolvedScan:

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
        self.intensity_srs = None
        self.all_images = None
        self.direct_beam = None

        self.z_valid = None
        self.popt = None
        self.perr = None

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
            self.direct_beam.find_center()
        except TypeError:
            pass
        data = np.load(self.directory / "compressed_scan.npz")
        self.motor_positions = data["motor_positions"]
        self.z_positions = data["z_positions"]
        self.intensity_srs = data["intensities"]
    
    def load_raw(self, zinger: bool = False):
        loader = RawLoader()
        file_list = list(self.directory.glob("*" + IM_TYPE))
        db_ind = DirectBeam.find_file_index(file_list)
        db_file = file_list[db_ind]
        del(file_list[db_ind])
        self.direct_beam = DirectBeam(db_file, zinger=zinger)
        self.direct_beam.find_center()
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
        
        self.intensity_srs = np.sum(self.all_images, axis=2).T

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
            fig.suptitle("$\\omega$-motor position", y=0.9)
        elif self.type == "z":
            # axes[int(len(image_indices) * 0.5)].set_title("$z$-motor position")
            fig.suptitle("$z$-motor position", y=.9)
        return fig, axes

    def plot_crop(self, ax: Axes, im_ind: int, max_value: float = None):
        ax.set_facecolor("k")
        image = self.all_images[im_ind]
        if max_value is None:
            max_value = image.max()
        pos = ax.imshow(image, norm=LogNorm(1, max_value), aspect='equal', cmap=cmap)
        motor_position = self.motor_positions[im_ind]
        if self.type == "om":
            title = f"${motor_position:.2f}\\degree$"
        elif self.type == "z":
            title = f"{motor_position:.2f} mm"
        ax.set_title(title)
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
            "intensities": self.intensity_srs
        }
        np.savez_compressed(self.directory / self.compressed_filename, **data)

    #############################

    def fit(self, z0: float=None, max_angle: float=1, pixel_cut: int=None,
            standard_deviations: float=None, motor_range: tuple=None):
        if self.type == "om":
            self.fit_om(z0, max_angle, pixel_cut, standard_deviations, motor_range=motor_range)
            
            # if self.data_remove:
            #     self.refit_om()

        elif self.type == "z":
            self.fit_z(standard_deviations)
        else:
            raise AttributeError("Scan type was not established.")
        
    def fit_om(self, z0: float=None, max_angle: float = 1., pixel_cut: int=None,
               cut_in_widths: float=None, motor_range=None):
        if z0 is not None:
            self.z0 = z0
        if cut_in_widths is not None:
            self.beam_cut = cut_in_widths

        if motor_range is not None:
            if isinstance(motor_range, (int, float)):
                motor_range = [motor_range, None]
            if motor_range[0] is None:
                mask = self.motor_positions < motor_range[1]
            if motor_range[1] is None:
                mask = self.motor_positions > motor_range[0]
            else:
                mask = np.logical_and(self.motor_positions > motor_range[0],
                                      self.motor_positions < motor_range[1])
            intensity_srs = self.intensity_srs[:, mask]
            motor_positions = self.motor_positions[mask]
        else:
            intensity_srs = self.intensity_srs
            motor_positions = self.motor_positions

        max_ind = np.argmax(intensity_srs, axis=1)
        where_max_angle = motor_positions[max_ind]
        
        valid = np.where(np.logical_and(
            self.z_positions > self.z0,
            self.z_positions < self.dist_guess * np.radians(max_angle) * UNIT_CONV + self.z0
            ))
        self.z_valid = self.z_positions[valid]
        self.where_max_angle = where_max_angle[valid]
        if pixel_cut is not None:
            self.z_valid = self.z_valid[:-pixel_cut]
            self.where_max_angle = self.where_max_angle[:-pixel_cut]
        
        self.popt, pcov = curve_fit(self.srs_om_fit, self.where_max_angle, self.z_valid, p0=[0, self.dist_guess])
        self.omega0, self.det_dist_fit = self.popt
        self.perr = np.sqrt(np.diag(pcov))
        print("Fit results:")
        print(f"    \u03C9\u2080 = ({self.omega0} \u00B1 {self.perr[0]})\u00B0")
        print(f"    d\u209B = ({self.det_dist_fit} \u00B1 {self.perr[1]}) mm")

    def fit_z(self, cut_in_widths=None):
        if cut_in_widths is not None:
            self.beam_cut = cut_in_widths

        print("Fit results:")
        fit_names = ["max", "z\u2080", "\u03C3\u2080"]
        unit_names = ["counts", "mm", "mm"]

        self.popt = {"total": None, "above": None, "beam": None}
        self.perr = {"total": None, "above": None, "beam": None}

        try:
            counts = np.sum(self.intensity_srs, axis=0)
            self.popt["total"], pcov = curve_fit(
                self.occlusion_fit_single, self.motor_positions, counts,
                p0=(counts.max(),
                    0.5 * (self.motor_positions.min() + self.motor_positions.max()),
                    self.beam_width)
                )
            self.perr["total"] = np.sqrt(np.diag(pcov))
            print("  Total counts:")
            for name, fit_res, err, unit in zip(fit_names, self.popt["total"], self.perr["total"], unit_names):
                print(f"    {name} = ({fit_res:.5f} \u00B1 {err:.5f}) {unit}")
        except RuntimeError:
            print("Failed to fit total counts")

        try:
            counts = np.sum(
                self.intensity_srs[np.where(self.z_positions < self.beam_width * self.beam_cut)],
                axis=0
            )
            self.popt["beam"], pcov = curve_fit(
                self.occlusion_fit_single, self.motor_positions, counts,
                p0=(counts.max(),
                    0.5 * (self.motor_positions.min() + self.motor_positions.max()),
                    self.beam_width)
            )
            self.perr["beam"] = np.sqrt(np.diag(pcov))
            print("  Primary beam counts:")
            for name, fit_res, err, unit in zip(fit_names, self.popt["beam"], self.perr["beam"], unit_names):
                print(f"    {name} = ({fit_res:.5f} \u00B1 {err:.5f}) {unit}")
            self.success_dbeam = True
        except RuntimeError:
            print("Failed to fit direct beam counts")

        try:
            counts = np.sum(
                self.intensity_srs[np.where(self.z_positions > self.beam_width * self.beam_cut)],
                axis=0
            )
            self.popt["above"], pcov = curve_fit(
                self.srs_z_fit, self.motor_positions, counts,
                p0=(counts.max(),
                    self.motor_positions.min(),
                    self.motor_positions.max(),
                    0.5 * self.beam_width,
                    0.5 * self.beam_width)
            )
            self.perr["above"] = np.sqrt(np.diag(pcov))
            fit_names = ["max", "z\u2081", "z\u2082", "\u03C3\u2081", "\u03C3\u2082"]
            unit_names = ["counts", "counts", "mm", "mm", "mm", "mm"]
            print("  Specular counts:")
            for name, fit_res, err, unit in zip(fit_names, self.popt["above"], self.perr["above"], unit_names):
                print(f"    {name} = ({fit_res:.5f} \u00B1 {err:.5f}) {unit}")
            self.success_specz = True
        except RuntimeError:
            print("Failed to fit specular counts")
        
    def show_omega_scan(self, style="quad", figsize=None, title=None, loc="upper left"):
        style = style.lower()
        if style == "quad":
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        elif "col" in style:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)
        if title is not None:
            fig.suptitle(title)
        self.plot_omega_fit(ax1)
        if isinstance(loc, (list, tuple)):
            self.plot_trad_scan(ax2, "total", loc=loc[0])
            self.plot_trad_scan(ax3, "above", loc=loc[1])
            self.plot_trad_scan(ax4, "beam", loc=loc[2])
        else:
            self.plot_trad_scan(ax2, "total", loc=loc)
            self.plot_trad_scan(ax3, "above", loc=loc)
            self.plot_trad_scan(ax4, "beam", loc=loc)
        fig.tight_layout()
        if style == "quad":
            return fig, ((ax1, ax2), (ax3, ax4))
        elif "col" in style:
            return fig, (ax1, ax2, ax3, ax4)

    def plot_omega_fit(self, ax):
        """FIT THROUGH MAX COUNT IN EACH ROW"""
        ax.scatter(self.where_max_angle, self.z_valid, s=10, marker='o',
                    edgecolors='k', lw=.75, facecolor='w')
        omega = np.linspace(self.where_max_angle[-1] - 0.02, self.where_max_angle[0] + 0.02, 100)
        ax.plot(omega, self.srs_om_fit(omega, self.omega0, self.det_dist_fit), "r")
        ax.set_xlabel("$\\omega$-motor position")
        ax.set_ylabel("$z$ (mm)")
        ax.set_title("What row brightest pixel occurs")
        annotation_text = f"$\\omega_0 = {self.omega0:.4f} \\pm {self.perr[0]:.4f}^\\circ$\n$d_{{sd}} = {self.det_dist_fit:.1f} \\pm {self.perr[1]:.1f}$ mm"
        ax.text(0.05, 0.93, annotation_text, transform=ax.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        ax.grid()

    def show_z_scan(self, style="quad", figsize=None, title=None, loc="upper left"):
        style = style.lower()
        if style == "quad":
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        elif "col" in style:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)
        if title is not None:
            fig.suptitle(title)
        self.plot_srs_plot(ax1)
        if isinstance(loc, (list, tuple)):
            self.plot_trad_scan(ax2, "total", loc=loc[0])
            self.plot_trad_scan(ax3, "above", loc=loc[1])
            self.plot_trad_scan(ax4, "beam", loc=loc[2])
        else:
            self.plot_trad_scan(ax2, "total", loc=loc)
            self.plot_trad_scan(ax3, "above", loc=loc)
            self.plot_trad_scan(ax4, "beam", loc=loc)
        fig.tight_layout()
        if style == "quad":
            return fig, ((ax1, ax2), (ax3, ax4))
        elif "col" in style:
            return fig, (ax1, ax2, ax3, ax4)

    def plot_trad_scan(self, ax, which: str = "total", loc="upper left"):
        loc_split = loc.split(" ")
        if len(loc_split) == 1:
            if loc == "center":
                text_x = 0.5
                halign = "center"
                text_y = 0.5
                valign = "center"
            elif loc == "left":
                text_x = 0.05
                halign = "left"
                text_y = 0.5
                valign = "center"
            elif loc == "right":
                text_x = 0.95
                halign = "right"
                text_y = 0.5
                valign = "center"
            elif loc == "upper" or loc == "top":
                text_x = 0.5
                halign = "center"
                text_y = 0.93
                valign = "top"
            elif loc == "lower" or loc == "bottom":
                text_x = 0.5
                halign = "center"
                text_y = 0.07
                valign = "bottom"
            else:
                raise ValueError("Invalid text location")
        else:
            if loc_split[0] == "upper" or loc_split[0] == "top":
                text_y = 0.93
                valign = "top"
            elif loc_split[0] == "center" or loc_split[0] == "middle":
                text_y = 0.5
                valign = "center"
            elif loc_split[0] == "lower" or loc_split[0] == "bottom":
                text_y = 0.07
                valign = "bottom"
            else:
                raise ValueError("Invalid text location")
            if loc_split[1] == "right":
                text_x = 0.95
                halign = "right"
            elif loc_split[1] == "center" or loc_split[1] == "middle":
                text_x = 0.5
                halign = "center"
            elif loc_split[1] == "left":
                text_x = 0.05
                halign = "left"
            else:
                raise ValueError("Invalid text location")
        which = which.lower()
        if which == "total":
            counts = np.sum(self.intensity_srs, axis=0)
            title = "Total Counts"
        elif "spec" in which or which == "above":
            counts = np.sum(
                self.intensity_srs[np.where(self.z_positions > self.beam_width * self.beam_cut)],
                axis=0
            )
            title = "Counts above beam"
        elif "beam" in which or which == "below":
            counts = np.sum(
                self.intensity_srs[np.where(self.z_positions < self.beam_width * self.beam_cut)],
                axis=0
            )
            title = "Counts in direct beam"
        ax.scatter(self.motor_positions, counts, s=10, edgecolors='k', lw=.75, facecolor='w')
        ax.set_title(title)
        ax.grid()
        ax.set_ylabel("Counts")
        if self.type == "om":
            motor = "\\omega"
            at_max = self.motor_positions[counts.argmax()]
            annotation_text = f"max at $\\omega={at_max:.2f}\\degree$"
            ax.text(text_x, text_y, annotation_text, transform=ax.transAxes,
                    verticalalignment=valign, horizontalalignment=halign,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.axvline(at_max, color="r", lw=.5)
        elif self.type == "z":
            func = {"total": self.occlusion_fit_single,
                    "beam": self.occlusion_fit_single,
                    "above": self.srs_z_fit}
            if which == "above":
                z0 = 0.5 * (self.popt[which][1] + self.popt[which][2])
                dz0 = np.sqrt(self.perr[which][1] * self.perr[which][1] + self.perr[which][2] * self.perr[which][2])
            else:
                z0 = self.popt[which][1]
                dz0 = self.perr[which][1]
            # annotation_text = f"$z_0={z0:.3f}\pm{dz0:.3f}$ mm"
            annotation_text = f"$z_0={z0:.2f}$ mm"
            ax.text(text_x, text_y, annotation_text, transform=ax.transAxes,
                    verticalalignment=valign, horizontalalignment=halign,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            motor = "z"
            x = np.linspace(self.motor_positions.min(), self.motor_positions.max(), 500)
            y = func[which](x, *self.popt[which])
            ax.plot(x, y, "r")

        ax.set_xlabel(f"${motor}$-motor position")
        if counts.max() > 1e4:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 1000:.0f}'))
            ax.set_ylabel("Counts per column $(\\times 1000)$")

    def srs_om_fit(self, omega, omega0, det_dist):
        return det_dist * np.tan(2. * np.radians(omega - omega0)) + self.z0
    
    def srs_om_fit2(self, omega, omega0, det_dist, radial_offset, rotation_offset):
        alpha = np.radians(omega - omega0)
        vertical_offset = radial_offset * (np.sin(rotation_offset + alpha) - np.sin(rotation_offset))
        return (det_dist - radial_offset * np.cos(rotation_offset + alpha)) * np.tan(2. * alpha) + vertical_offset

    @staticmethod
    def srs_om_error(omega, omega0, det_dist, omega_err, dist_err):
        omega_center = omega - omega0
        dist_deriv = np.tan(2. * omega_center)
        sec_2omega = 1. / np.cos(2. * omega_center)
        omega_deriv = 2. * det_dist * sec_2omega * sec_2omega
        dist_term = dist_err * dist_deriv
        omega_term = omega_err * omega_deriv
        return np.sqrt(dist_term * dist_term + omega_term * omega_term)

    @staticmethod
    def occlusion_fit_single(z_motor, max_counts, z0, sigma):
        return 0.5 * (max_counts - max_counts * erf((z_motor - z0) / sigma))

    @staticmethod
    def occlusion_fit_double(z_motor, max_counts, z_lo, z_hi, sigma_lo, sigma_hi):
        return 0.5 * max_counts * (erf((z_motor - z_hi) / sigma_hi) - erf((z_motor - z_lo) / sigma_lo)) + max_counts
    
    @staticmethod
    def srs_z_fit(z_motor, max_counts, z_lo, z_hi, sigma_lo, sigma_hi):
        return 0.5 * max_counts * (erf((z_motor - z_lo) / sigma_lo) - erf((z_motor - z_hi) / sigma_hi))
    
    def horizon_func(self, omega, omega0, det_dist):
        return det_dist * np.tan(np.radians(omega - omega0)) + self.z0
    
    def yoneda_func(self, omega, omega0, det_dist, critical_angle):
        return det_dist * np.tan(np.radians(omega - omega0 + critical_angle)) + self.z0
    
    def transmission_func(self, omega, omega0, det_dist, critical_angle):
        alpha = np.radians(omega - omega0)
        alpha_sq = alpha * alpha
        critical_angle = np.radians(critical_angle)
        crit_sq = critical_angle * critical_angle
        refraction_angle_sq = (alpha_sq - crit_sq) / (1 - 0.5 * crit_sq)
        return self.z0 + det_dist * np.tan(alpha - np.sqrt(refraction_angle_sq))
    
    def transmission_reflection_func(self, omega, omega0, det_dist, critical_angle):
        alpha = np.radians(omega - omega0)
        alpha_sq = alpha * alpha
        critical_angle = np.radians(critical_angle)
        crit_sq = critical_angle * critical_angle
        refraction_angle_sq = (alpha_sq - crit_sq) / (1 - 0.5 * crit_sq)
        return self.z0 + det_dist * np.tan(alpha + np.sqrt(refraction_angle_sq))
    
    def refraction_exiting_func(self, omega, omega0, det_dist, critical_angle):
        alpha = np.radians(omega - omega0)
        critical_angle = np.radians(critical_angle)
        crit_sq = critical_angle * critical_angle
        refraction_angle = np.sqrt(alpha * alpha * (1 - 0.5 * crit_sq) + crit_sq)
        return self.z0 + det_dist * np.tan(alpha + refraction_angle)
    
    def show_srs_plot(self, critical_angle=None, title=None, figsize=None, horizon=True, max_counts=None):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if title is None:
            if self.type == "om":
                title = "SRS-$\\omega$"
            else:
                title = "SRS-$z$"
        color_map = self.plot_srs_plot(ax, critical_angle=critical_angle, title=title, horizon=horizon, max_counts=max_counts)
        color_bar = fig.colorbar(color_map, ax=ax)
        color_bar.set_label("Counts per row")
        color_bar.ax.tick_params(which="both", direction="out")
        fig.tight_layout()
        return fig, ax

    def plot_srs_plot(self, ax, critical_angle=None, title=None, horizon=False, max_counts=None):
        """SPECULAR"""
        ax.set_facecolor('k')
        if title is not None:
            ax.set_title(title)

        # z = np.arange(intensity_x_total.shape[1])[::-1] * self.pixel_size
        ax.set_ylabel(f"$z$ {UNIT}")

        if max_counts is None:
            max_counts = self.intensity_srs.max()

        color_map = ax.pcolormesh(self.motor_positions, self.z_positions, self.intensity_srs,
                                  norm=LogNorm(1, max_counts), cmap="plasma", rasterized=True)
        
        if horizon:
            ax.axhline(self.z0, linestyle="--", color="#FF5349", linewidth=0.7)
            ax.axhline(self.beam_cut * self.beam_width,
                       color="w", linewidth=0.5, linestyle="--")
        
        if self.type == "om":
            ax.axvline(self.omega0, linestyle="--", color="#FF5349", linewidth=0.7)
            om_axis_plus = np.linspace(self.omega0, self.omega0 + .75, 1000)
            ax.plot(om_axis_plus, self.srs_om_fit(om_axis_plus, self.omega0, self.det_dist_fit),
                    "white", linewidth=1, alpha=0.5)
            if horizon:
                ax.plot(om_axis_plus, self.horizon_func(om_axis_plus, self.omega0, self.det_dist_fit),
                        "white", linestyle='dashed', linewidth=1, alpha=0.5)
            if critical_angle is not None:
                if isinstance(critical_angle, float):
                    critical_angle = [critical_angle]
                last = None
                for crit in critical_angle:
                    if crit == last:
                        om_axis_crit_plus = np.linspace(self.omega0 + crit, self.omega0 + crit + .05, 100)
                        ax.plot(om_axis_crit_plus,
                                self.transmission_reflection_func(om_axis_crit_plus, self.omega0, self.det_dist_fit, crit),
                                "white", linewidth=1, alpha=0.5)
                    else:
                        ax.plot(om_axis_plus, self.yoneda_func(om_axis_plus, self.omega0, self.det_dist_fit, crit),
                                "white", linewidth=1, alpha=0.5)
                        om_axis_crit_plus = np.linspace(self.omega0 + crit, self.motor_positions[-1], 1000)
                        ax.plot(om_axis_crit_plus,
                                self.transmission_func(om_axis_crit_plus, self.omega0, self.det_dist_fit, crit),
                                "white", linewidth=1, alpha=0.5)
                        om_axis_crit_minus = np.linspace(self.motor_positions.min(), self.omega0, 1000)
                        ax.plot(om_axis_crit_minus,
                                self.refraction_exiting_func(om_axis_crit_minus, self.omega0, self.det_dist_fit, crit),
                                "white", linewidth=1, alpha=0.5)
                    last = crit
            ax.set_xlabel("$\\omega$-motor position $(\\degree)$")
        elif self.type == "z":
            ax.set_xlabel(f"$z$-motor position $({UNIT})$")
        ax.set_xlim(self.motor_positions.min(), self.motor_positions.max())
        ax.set_ylabel(f"$z$ ({UNIT})")
        ax.tick_params(axis='both', which='both', color='white', labelcolor='black')
        return color_map



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

    def __init__(self, filename: Path, zinger: bool = False):
        """
        Initializes the DirectBeam instance by loading and processing the direct beam image.

        Args:
            filename (Path): The path to the direct beam image file.
        """
        image = fabio.open(filename).data
        
        if zinger:
            mask = self.detector.calc_mask_dezinger(image)
        else:
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
        try:
            self.double_result_y, _ = curve_fit(self.double_erf, self.y, self.counts_y,
                                                p0=(y0, sigma_y, sigma_y * 0.2, a_y), nan_policy="omit")
            self.center = (self.double_result_y[0], self.double_result_x[0])
            self.width = (abs(self.double_result_y[1]), abs(self.double_result_x[1]))
        except RuntimeError:
            self.double_result_y = (y0, sigma_y, a_y)
            self.center = (y0, self.double_result_x[0])
            self.width = (abs(sigma_y) * 2. * np.sqrt(2. * np.log(2.)), abs(self.double_result_x[1]))
        

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
        if self.counts_x.max() > 1e4:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 1000:.0f}'))
            ax.set_ylabel("Counts per column $(\\times 1000)$")
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
            if len(self.double_result_y) == 4:
                x_fit = self.double_erf(y_fit, *self.double_result_y)
            elif len(self.double_result_y) == 3:
                x_fit = self.gaussian(y_fit, *self.double_result_y)
            ax.set_ylim(self.center[0] - yrange, self.center[0] + yrange)
            if self.counts_y.max() > 1e4:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 1000:.0f}'))
                ax.set_xlabel("Counts per column $(\\times 1000)$")
        else:
            x = self.y
            x_fit = np.linspace(self.center[0] - yrange, self.center[0] + yrange, 1000)
            y = self.counts_y
            if len(self.double_result_y) == 4:
                y_fit = self.double_erf(x_fit, *self.double_result_y)
            elif len(self.double_result_y) == 3:
                y_fit = self.gaussian(y_fit, *self.double_result_y)
            ax.set_ylabel("Counts")
            ax.set_xlabel("row")
            ax.set_xlim(self.center[0] - yrange, self.center[0] + yrange)
            if self.counts_y.max() > 1e4:
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 1000:.0f}'))
                ax.set_ylabel("Counts per column $(\\times 1000)$")
        ax.scatter(x, y, s=10, marker='o', edgecolors='k', lw=.75, facecolor='w')
        ax.plot(x_fit , y_fit, "r")
        ax.grid()

    def plot_beam(self, ax):
        ax.set_facecolor('k')
        pos = ax.imshow(self.data, norm=LogNorm(1, self.data.max()), cmap=cmap)
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
        fig.suptitle("Direct Beam")

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
    
    def show_raw_beam(self, figsize=(3.37, 2.75)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_facecolor('k')
        pos = ax.imshow(self.data, norm=LogNorm(1, self.data.max()), cmap=cmap)
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
    # data_path = Path("tests/test-data/")
    # om_path = data_path / "ex-om-scan"
    # z_path = data_path / "ex-z-scan"
    # spec = SpatiallyResolvedScan(om_path, .150)
    # spec.fit()
    # spec.show_omega_scan()
    # spec.show_specular_plot([.24, .19, .19])
    # plt.show()
    data_path = Path(r"C:\Users\Teddy\OneDrive - UCB-O365\Rogerslab3\Teddy\XRD\GI-scans\2024-11-09\2024-11-09_om-scan_at-557-um-2")
    data_path = Path(r"C:\Users\Teddy\OneDrive - UCB-O365\Rogerslab3\Teddy\XRD\GI-scans\2024-10-25 variable det-dist - sio2\2024-10-25_om-scan_at-573-um-2")
    file = data_path / "om_scan_direct_beam.tif"
    db = DirectBeam(file)
    db.find_center()
    db.plot_quad()
    plt.show()
