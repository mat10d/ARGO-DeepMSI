"""MSI scorers — uniform interface across zero-shot, head-on-our-data,
and aggregator-fusion approaches."""

from .base import Scorer, ScoreColumn
from .registry import get_scorer, list_scorers, register

# Import each module so its register() call runs on package import.
from . import calibrated_pool  # noqa: F401
from . import clam_tilemil  # noqa: F401
from . import flex_bottleneck  # noqa: F401
from . import fmmap_probe  # noqa: F401
from . import nuclear_morphology  # noqa: F401
from . import protonet_cluster  # noqa: F401
from . import score_fusion  # noqa: F401
from . import selective_abstention  # noqa: F401
from . import setencoder_agg  # noqa: F401
from . import simple_grid  # noqa: F401
from . import slide_attention_mil  # noqa: F401
from . import slidefm_linearprobe  # noqa: F401
from . import tip_adapter  # noqa: F401
from . import transductive_smoothing  # noqa: F401
from . import vl_text_cosine  # noqa: F401
from . import wagner_zeroshot  # noqa: F401

__all__ = ["Scorer", "ScoreColumn", "get_scorer", "list_scorers", "register"]
