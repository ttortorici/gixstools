from unittest import TestCase
from importlib.resources import files
import tomllib
import pprint
import gixstools.config
from pathlib import Path


class TestInstallation(TestCase):
    def test_included_files(self):
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
        gixstools.config.create_user_config()
        user_config_path = Path.home() / "Documents" / "GIXS" / "config.toml"
        self.assertTrue(
            user_config_path.is_file(),
            "Failed to copy default config file to Documents/GIXS"
        )
        print("Successfully copied config file to user folder")
        print("\n----------\nPASSED: Found all files\n----------")


if __name__ == "__main__":
    from unittest import main
    main()