from importlib.resources import files
import tomllib
import pprint


def test_included_files():
    print("----------\nLocate included installation files\n----------\n")
    style_path = files("gixstools").joinpath("style.mplstyle")
    with style_path.open("r") as f:
        print("style.mplstyle")
        for line in f.readlines():
            print("\t" + line.strip("\n"))
    config_path = files("gixstools").joinpath("default_config.toml")
    with config_path.open("rb") as f:
        print("\ndefault_config.toml")
        config = tomllib.load(f)
        pprint.pprint(config, indent=4, width=80, compact=False)
    print("\n----------\nPASSED: Found all files\n----------")


if __name__ == "__main__":
    test_included_files()