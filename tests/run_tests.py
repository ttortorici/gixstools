print("\n---------")
print("starting tests")
print("---------\n")

from transform_c_vs_py import test_transform_ones
from installation import test_included_files

if __name__ == "__main__":
    test_included_files()
    test_transform_ones()
    