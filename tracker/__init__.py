"""Pokemon TCG replay tracker.

Pipeline: replay logs -> flat row table -> reward components -> W&B.
See CONTEXT.md section 6. One flat table is the single source of truth;
the reward curve, the action histogram, and the W&B dashboard are all
just views of that one table.
"""

from .parse import parse_replay, parse_many
from .reward import add_reward_components, DENSE, SPARSE
from .tactics import add_tactical_flags, retreat_quality_summary

__all__ = [
    "parse_replay", "parse_many", "add_reward_components", "DENSE", "SPARSE",
    "add_tactical_flags", "retreat_quality_summary",
]
