"""Credit-risk ML pipeline."""

import os

_logical_cores = os.cpu_count() or 1
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, _logical_cores - 1)))
