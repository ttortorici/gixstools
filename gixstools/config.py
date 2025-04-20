from importlib.resources import files
import tomllib
from pathlib import Path
from datetime import datetime
import shutil


def load(subdict: str = None, path: str = None) -> dict:
    """
    Load the dict from the config file.

    :param subdict: return a sub-dictionary using a key.
    :param path: path to config file. If None is given, then the default is loaded.
    :return: config dictionary
    """
    if path is None:
        # look for user config
        path = Path.home() / "Documents" / "GIXS"
        user_config = path / "config.toml"

        if not user_config.is_file():
            user_config = files("gixstools").joinpath("default_config.toml")
    with user_config.open("rb") as f:
        config = tomllib.load(f)
    if subdict is not None:
        config = config[subdict]
    return config


def create_user_config():
    path = Path.home() / "Documents" / "GIXS"
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    user_config = path / "config.toml"

    if user_config.is_file():
        now = datetime.now()
        datestamp = f"{now.year:02d}{now.month:02d}{now.day:02d}"
        timestamp = "{:05d}".format(now.hour * 360 + now.minute * 60 + now.second)
        copy_old_to = path / f"{user_config.stem}-replaced-on-{datestamp}-{timestamp}.toml"
        
        user_config.rename(copy_old_to)
    
    shutil.copy(files("gixstools").joinpath("default_config.toml"), user_config)


def clear_old_config():
    path = Path.home() / "Documents" / "GIXS"
    print("Deleting all user config files in {}".format(path.as_posix()))
    for config_file in path.glob("config-replaced*.toml"):
        config_file.unlink()


def clear_all_config():
    path = Path.home() / "Documents" / "GIXS"
    print("Deleting all user config files in {}".format(path.as_posix()))
    for config_file in path.glob("config*.toml"):
        config_file.unlink()
