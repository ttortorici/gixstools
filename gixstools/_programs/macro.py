import argparse
from pathlib import Path
from gixstools.align.macro import create_om_macro, create_z_macro, create_list
from gixstools.config import load


def scan_macro():
    beamstop = load("macro")["beamstop"]

    parser = argparse.ArgumentParser(
        prog="scan-macro",
        description="Create a macro for spec to scan through motor angles to calibrate for GIXS.",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>"
    )
    parser.add_argument("type", type=str, help="Motor to scan (om or z).")
    parser.add_argument("start", type=float, help="Starting motor position")
    parser.add_argument("end", type=float, help="Ending motor position")
    parser.add_argument("step", type=float, help="Increment size of motor position")
    # parser.add_argument("-T", "-thickness", help="sample thickness")
    parser.add_argument("-N", "--name", default="", help="add a short tag to the TIFF files generated")
    parser.add_argument("-C", "--clear", action='store_true', help="remove all macro files saved")
    parser.add_argument("-B", "--beamstop", action=f'store_{(not beamstop)}'.lower(),
                        help=f'Is there a beamstop that needs to be moved? This flag will flip the default ({beamstop}).')

    dir = Path.home() / "Documents" / "GIXS" / "macros"
    args = parser.parse_args()
    if args.type.lower() not in ["om", "z"]:
        raise ValueError("Invalid type. Must be 'om' or 'z'.")
    if args.clear:
        for macro_file in dir.glob("*.txt"):
            macro_file.unlink()
    list_to_scan = create_list(args.start, args.end, args.step)
    if args.type.lower() == "om":
        final = -4
        macroname = create_om_macro(list_to_scan, dir, args.name, final, args.beamstop)
        unit = "degree"
    elif args.type.lower() == "z":
        final = -5
        macroname = create_z_macro(list_to_scan, dir, args.name, final, args.beamstop)
        unit = "mm"
    num = len(list_to_scan) + 1
    time_min = float(num) * 0.1
    minutes = int(time_min)
    seconds = round((time_min - minutes) * 60)
    print(f"Macro written with {num} images. Estimated time (min:sec): {minutes}:{seconds:02d}")
    print("Copy and paste the following into SAXS to run the macro:")
    print("do " + (dir / macroname).as_posix())
    print(f"WARNING: will leave {args.type} at {final} {unit}")