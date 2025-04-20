import argparse
from pathlib import Path
from gixstools.align import SpecularScan
import matplotlib.pylab as plt


def plot():
    parser = argparse.ArgumentParser(
        prog="plot",
        description="Plot om scan as an image with horizontal integration over pixels, rows: y-pixels, columns: om, color: intensity\nThis will look for data in the latest directory of the user",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>"
    )
    parser.add_argument("dir", help="Specify a specific directory, or CWD for current working directory")
    parser.add_argument("-A", "--animate", type=int, default=None, help="Animate the images of the scan: specify frame-rate in FPS")
    # parser.add_argument("-T", "--title", help="Title of plots.")
    parser.add_argument("-Z", "--zposition", type=float, default=0., help="move the z-position of the sample relative to the beam center")
    parser.add_argument("-C", "--critical_angles", help="add critical angles (comma separated)")
    parser.add_argument("-L", "--beamloc", default="LC", help="Choose which ninth in a 3x3 grid, the beam is in: (UL, UC, UR, CL, CC, CR, LR, LC, LL)")
    parser.add_argument("-B", "--beamheight", type=float, default=3, help="change where the beam cutoff is, in standard deviations.")
    parser.add_argument("-W", "--beamwidth", type=float, default=1., help="set beam width in mm")
    parser.add_argument("-D", "--distance", type=float, default=150, help="Guess at detector distance in mm. Default is 150 mm")
    parser.add_argument("-X", "--remove", type=int, default=None, help="Remove a number of points from the specular fit (will remove the furthest ones from the fit)")
    parser.add_argument("-R", "--range", type=float, default=1.5, help="set angular range in degrees. Default is 1.5 deg")
    parser.add_argument("-S", "--save", type=int, default=None, help="save the plot at a certain DPI")
    parser.add_argument("-N", "--name", help="Title plots.")
    
    args = parser.parse_args()

    if args.dir.lower() == "cwd" or args.dir.lower() == "here":
        directory = Path.cwd()
    else:
        directory = Path(args.dir)

    directory_name_split = directory.name.split("_")
    scan_type = directory_name_split[1]
    other_position = float(directory_name_split[2].split("-")[1])
    if scan_type == "om-scan":
        other_unit = "um"
    elif scan_type == "z-scan":
        other_unit = "millidegree"
    else:
        other_unit = ""
    
    plot_name = f"{scan_type}_at-{other_position}-{other_unit}"

    if args.beamloc not in ["UL", "UC", "UR", "CL", "CC", "CR", "LR", "LC", "LL"]:
        raise ValueError("Beam locations options are: UL, UC, UR, CL, CC, CR, LR, LC, LL\nFor Upper, Center, Lower; Left, Center, Right.")

    spec = SpecularScan(
        directory,
        det_dist=args.distance,
        anglular_range=args.range,
        beam_width=args.beamwidth,
        standard_deviations=args.beamheight,
        plot_name=plot_name,
        data_remove=args.remove,
        beam_location=args.beamloc,
        plot_dpi=args.save,
        plot_title=args.name,
    )

    if spec.type == "om":
        if args.critical_angles is None:
            crit = None
        else:
            crit = [float(c) for c in args.critical_angles.split(",")]
        if args.zposition:
            spec.fit(z0=float(args.zposition))
        spec.plot_om()
        spec.plot(critical_angle=crit)

    if args.animate is not None:
        fps = args.animate
        fig, ani = spec.animate_tiffs(fps)
    
    if args.save:
        fig
    spec.save(directory)
    print("Saved data in: " + directory.as_posix())
    
    plt.show()