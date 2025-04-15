from importlib.resources import files
import tomllib


def load(subdict: str = None, path: str = None):
    """
    Load the dict from the config file.

    :param subdict: return a sub-dictionary using a key.
    :param path: path to config file. If None is given, then the default is loaded.
    :return: config dictionary
    """
    if path is None:
        path = files("gixstools").joinpath("default_config.toml")
    with path.open("rb") as f:
        config = tomllib.load(f)
    if subdict is not None:
        config = config[subdict]
    return config