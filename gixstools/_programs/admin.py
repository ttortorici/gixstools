import argparse
import shutil
from pathlib import Path
from datetime import datetime
from gixstools import config


def new_config():
    parser = argparse.ArgumentParser(
        prog="new-config",
        description="Copy the default config file to ~/Documents/GIXS. Will rename any current config file to 'config-replaced-on-[date].toml'.",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>"
    )
    args = parser.parse_args()
    config.create_user_config()


def clear_config():
    parser = argparse.ArgumentParser(
        prog="clear-config",
        description="Clear old config files in ~/Documents/GIXS.",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>"
    )
    parser.add_argument("-A", "--all", action="store_true", 
                        help="Clear all config files, even the active one")
    args = parser.parse_args()
    
    if args.all:
        config.clear_all_config()
    else:
        config.clear_old_config()


def new_user():
    parser = argparse.ArgumentParser(
        prog="new-user",
        description="Create a new directory for a new user (must be alphanumeric).",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>"
    )
    parser.add_argument("user_name",
                        help="This will be the name (not case sensitive) of the directory in ~/Documents/GIXS.")
    args = parser.parse_args()
    
    user_name = str(args.user_name.lower())

    if " " in user_name:
        raise ValueError("There cannot be spaces in a user name. Try again")
    if not user_name.isalnum():
        raise ValueError("User name must be alphanumeric. Try again")
    
    user_dir = Path.home() / "Documents" / "GIXS" / user_name

    if user_dir.exists():
        raise ValueError("User directory already exists.")

    user_dir.mkdir(parents=True, exist_ok=True)
    print(f"Made user directory: {user_dir}")


def save_config():
    parser = argparse.ArgumentParser(
        prog="save-config",
        description="Save the active config.toml to a user directory.",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>"
    )
    parser.add_argument("user_name", help="The config will be saved into this user's directory.")
    args = parser.parse_args()

    path = Path.home() / "Documents" / "GIXS"
    user_dir = path / str(args.user_name.lower())
    file_name = "config.toml"

    if not user_dir.exists():
        raise ValueError("User does not exist. Check for typos or make a new user with `new-user`.")
    
    shutil.copy(path / file_name, user_dir / file_name)
    print(f"Saved config.toml to {user_dir}")
    

def load_config():
    parser = argparse.ArgumentParser(
        prog="load-config",
        description="Load a user config.toml.",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>"
    )
    parser.add_argument("user_name", help="The config will be saved into this user's directory.")
    args = parser.parse_args()

    path = Path.home() / "Documents" / "GIXS"
    user_dir = path / str(args.user_name.lower())
    file_name = "config.toml"

    if not user_dir.exists():
        raise ValueError("User does not exist. Check for typos or make a new user with `new-user`.")
    
    config.create_user_config()  # this will rename an existing config so it does not get lost

    (path / file_name).unlink()
    shutil.copy(user_dir / file_name, path / file_name)


def move_scan():
    now = datetime.now()
    parser = argparse.ArgumentParser(
        prog="move-scan",
        description="Move files from data folder to a user folder.",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>"
    )
    parser.add_argument("type", help="Either om or z")
    parser.add_argument("user_name", help="This will be the name of the directory the files will be moved to")
    parser.add_argument("other_position", type=float, help="Specify the position of the other motor (the one note being scanned).")
    parser.add_argument("-A", "--append", action="store_true", help="Add this tag to append the latest set of data")
    args = parser.parse_args()
    
    raw_data_path = Path(config.load("admin")["data_path"]).expanduser()
    user_dir = Path.home() / "Documents" / "GIXS" / str(args.user_name.lower())

    if not user_dir.exists():
        raise ValueError("User does not exist. Check for typos or make a new user with `new-user`.")
    
    scan_dir = user_dir / "alignment-scans" / f"{now.year:02d}-{now.month:02d}-{now.day:02d}"
    
    if not scan_dir.exists():
        scan_dir.mkdir(parents=True, exist_ok=True)

    scan_type = args.type.lower()
    if scan_type not in ["om", "z"]:
        raise ValueError("Scan type must be 'om' or 'z'.")
    if "om" in args.type.lower():
        other_unit = "um"
    else:
        other_unit = "mdeg"
    
    other_pos_in_milli_units = round(args.other_position * 1000)

    time_stamps = []
    if args.append:
        directory_name = ""
        for dir in scan_dir.glob(f"*{scan_type}-scan_at-*"):
            if dir.is_dir():
                hour, minute, second = map(int, dir.name.split("-")[:3])
                time_stamps.append((datetime(now.year, now.month, now.day, hour, minute, second), dir))
        timestamp = max(time_stamps, key=lambda timestamp: timestamp[0])[1]
    else:
        timestamp = now
    directory_name = f"{timestamp.hour:02d}-{timestamp.minute:02d}-{timestamp.second:02d}_{scan_type}-scan_at-{other_pos_in_milli_units}-{other_unit}"
    
    new_data_path = scan_dir / directory_name

    new_data_path.mkdir(parents=True)

    scan_glob = raw_data_path.glob(f"{scan_type}_scan*.tif")
    print(f"Found {len(list(scan_glob))} files to move in {raw_data_path}")
    print(f"Moving these files to {new_data_path}")

    for tif in raw_data_path.glob(f"{scan_type}_scan*.tif"):
        new_tif = new_data_path / tif.name
        print(new_tif)
        if new_tif.is_file():
            new_tif.unlink()
        shutil.move(tif, new_tif)


def clear_scan():
    parser = argparse.ArgumentParser(
        prog="clear-scan",
        description="Remove old scan files from data.",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>"
    )
    args = parser.parse_args()

    raw_data_path = Path(load_config("admin")["data_path"]).expanduser()
    for tif in raw_data_path.glob(f"om_scan*.tif"):
        tif.unlink()
    for tif in raw_data_path.glob(f"z_scan*.tif"):
        tif.unlink()
        
