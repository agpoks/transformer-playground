"""Real dataset loaders shared across models/*/example.py scripts.

Each `load_*` function downloads (if needed, into ~/.cache or a local
data_cache/ dir) and returns real data -- no synthetic placeholders except
where a model's own docs explicitly says otherwise (e.g. a physics-simulated
target where no public labeled dataset exists).
"""

from __future__ import annotations
