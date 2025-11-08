import sys
from pathlib import Path
gixspath = Path(__file__).resolve().parents[2]#  / 'gixstools'
sys.path.insert(0, str(gixspath))
# sys.path.insert(0, str(gixspath / 'align'))
# sys.path.insert(0, str(gixspath / 'wedge'))
# sys.path.insert(0, str(gixspath / 'detector'))
print(sys.path)
import gixstools.align