from datetime import datetime
from pathlib import Path
from gixstools.config import load


_config = load("macro")

_set_om = _config["set_omega"]
_set_vert = _config["set_vertical"]
_move_bs = _config["move_beamstop"]
_move_vert = _config["move_vertical"]
_expose = _config["expose"]
_vertical_shutter = _config["vertical_shutter"]
_horizontal_shutter = _config["horizontal_shutter"]


def create_om_macro(angles: list, directory: Path = Path.cwd(), tag: str = "", final: int = -4, beamstop=True) -> None:
    """
    Create a macro file for scanning angle for GIWAXS
    :param angles: list of angles to scan through
    :param tag: optional identifier to put in filename
    :return: None
    """
    if isinstance(directory, str):
        directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    print(angles)
    if tag:
        tag += "_"
    date = datetime.now()
    macroname = f'Specular_om_macro-{date.year:02d}{date.month:02d}{date.day:02d}-{date.hour:02d}.txt'
    print("Writing Macro...")
    with open(directory / macroname, 'w') as f:
        f.write(_vertical_shutter)
        f.write(_horizontal_shutter)
        if beamstop:
            f.write(_move_bs.format(5))  # move beam stop out of the way

        for om in angles:
            f.write(_set_om.format(om))
            formatted_angle = "{}_{}".format(*str(om).split("."))
            img_name = f"om_scan_{tag}{formatted_angle}_degrees"
            f.write(_expose.format(img_name))

        f.write(_move_vert.format(-10))  # move sample out of the way
        f.write(_expose.format("om_scan_direct_beam"))  # take direct beam exposure
        f.write(_move_vert.format(10))  # move sample back into beam
        if beamstop:
            f.write(_move_bs.format(-5))
        f.write(_set_om.format(final))
    return macroname


def create_z_macro(zs: list, directory: Path = Path.cwd(), tag: str = "", final: int = -5, beamstop=True) -> None:
    """
    Create a macro file for scanning angle for GIWAXS
    :param zs: list of z-positions to scan through
    :param tag: optional identifier to put in filename
    :return: None
    """
    if isinstance(directory, str):
        directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    print(zs)
    if tag:
        tag += "_"
    date = datetime.now()
    macroname = f'Specular_z_macro-{date.year:02d}{date.month:02d}{date.day:02d}-{date.hour:02d}.txt'
    print("Writing Macro...")
    with open(directory / macroname, 'w') as f:
        f.write(_vertical_shutter)
        f.write(_horizontal_shutter)
        if beamstop:
            f.write(_move_bs.format(5))  # move beam stop out of the way
        for z in zs:
            f.write(_set_vert.format(z))
            formatted_angle = "{}_{}".format(*str(z).split("."))
            img_name = f"z_scan_{tag}{formatted_angle}_mm"
            f.write(_expose.format(img_name))
        f.write(_move_vert.format(final))
        f.write(_expose.format("z_scan_direct_beam"))  # take direct beam exposure
        if beamstop:
            f.write(_move_bs.format(-5))
    return macroname  


def create_list(start, finish, step):
    """
    Make a list of values similar to np.arange, but with values rounded to avoid floating point precision issues
    :param start: first element of the list
    :param finish: last element of the list
    :param step: difference between sequential elements
    :return: list of values
    """
    start = float(start)
    finish = float(finish)
    step = float(step)
    # Try to determine what digit to round to
    step_decimal = str(step).split(".")  # list[str]: [left of decimal, right of decimal]
    if step_decimal[1] == 0:        # then step is an integer
        rounding = 0
        # find lowest order non-zero digit
        while True:
            if step_decimal[0][::-1].find('0') == 0:
                step_decimal[0] = step_decimal[0][:-1]
                rounding -= 1
            else:
                break
    else:                           # then step is not an integer
        rounding = len(step_decimal[1])     # number of digits right of the decimal
    return [round(x * step + start, rounding) for x in list(range(int((finish + step - start) / step)))]
