from unittest import TestCase
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
import gixstools.config


class TestPrograms(TestCase):
    def test_program(self):
        """Test individual CLI programs with the '-h' flag."""
        programs = [
            "transform",
            "scan-macro",
            "new-config",
            "clear-config",
            "new-user",
            "save-config",
            "load-config",
            "move-scan",
            "clear-scan",
        ]

        for program in programs:
            with self.subTest(program=program):
                program_name = f"{program}"
                print(f"Testing program: {program_name}")
                result = subprocess.run([program_name, "-h"], capture_output=True, text=True)
                self.assertEqual(
                    result.returncode, 0,
                    f"Program {program_name} failed with return code {result.returncode}."
                )
                print(f" - {program_name} passed")

    def test_macro_programs(self):
        def fake_run(data_path, macro_path, macroname):
            """Fake running the macro"""
            expose_command = gixstools.config.load("macro")["expose"].split(" ")[0]
            with open(macro_path / macroname, "r") as f:
                macrotext_list = f.read().split('\n')
            files_to_write = [item.split(" ")[-1] for item in macrotext_list if item.startswith(expose_command)]
            for filename in files_to_write:
                (data_path / filename).touch()
            return files_to_write

        data_path = Path(gixstools.config.load("admin")["data_path"]).expanduser()
        macro_path = Path.home() / "Documents" / "GIXS" / "macros"
        user_path = Path.home() / "Documents" / "GIXS" / "test"

        if data_path.is_dir():
            data_path_exists = True
        else:
            data_path.mkdir(parents=True)
            data_path_exists = False

        subprocess.run(["new-user", "test"], capture_output=True, text=True)

        now = datetime.now()
        subprocess.run(["scan-macro", "om", "-1", "1", "0.1"], capture_output=True, text=True)
        macroname = f'Specular_om_macro-{now.year:02d}{now.month:02d}{now.day:02d}-{now.hour:02d}.txt'
        macrotext_list_om = fake_run(data_path, macro_path, macroname)
        
        """Move the files that were created"""
        scan_dir = user_path / "alignment-scans" / f"{now.year:02d}-{now.month:02d}-{now.day:02d}"
        subprocess.run(["move-scan", "om", "test", "0"], capture_output=True, text=True)

        """REPEAT FOR Z"""
        now = datetime.now()
        subprocess.run(["scan-macro", "z", "-1", "1", "0.1"], capture_output=True, text=True)
        macroname = f'Specular_z_macro-{now.year:02d}{now.month:02d}{now.day:02d}-{now.hour:02d}.txt'
        macrotext_list_z = fake_run(data_path, macro_path, macroname)
        
        """Move the files that were created"""
        scan_dir = user_path / "alignment-scans" / f"{now.year:02d}-{now.month:02d}-{now.day:02d}"
        subprocess.run(["move-scan", "z", "test", "0"], capture_output=True, text=True)

        for dir in scan_dir.glob("*_om-scan*"):
            if dir.is_dir():
                for filename in macrotext_list_om:
                    self.assertTrue((dir / filename).is_file(),
                                    "Failed to copy files from macro run using 'move-scan'.")
        for dir in scan_dir.glob("*_z-scan*"):
            if dir.is_dir():
                for filename in macrotext_list_z:
                    self.assertTrue((dir / filename).is_file(),
                                    "Failed to copy files from macro run using 'move-scan'.")
        shutil.rmtree(user_path)


if __name__ == "__main__":
    from unittest import main
    main()