from pbr.version import VersionInfo

from .latency import Latency
from .synops import SynOps

all = "__version__"
__version__ = VersionInfo("snnmetric").release_string()
