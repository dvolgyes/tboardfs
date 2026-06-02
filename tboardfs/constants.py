from pathlib import Path
from types import MappingProxyType

TENSORBOARD_EVENT_GLOB = "events.out.tfevents*"
LARGE_TENSOR_THRESHOLD = 1024
SCALAR_COLUMNS = ("epoch", "step", "wall_time", "relative_time", "value")
DEFAULT_SCALAR_FORMATS = ("json", "tsv", "npz")
CONTROL_FILES = (".cache", ".in_memory")

FIXED_TABS = (
    "scalars",
    "custom_scalars",
    "images",
    "audio",
    "videos",
    "histograms",
    "distributions",
    "text",
    "meshes",
    "pr_curves",
    "hparams",
    "tensors",
    "graphs",
    "projector",
    "profile",
    "plugins",
)

PLUGIN_TAB = MappingProxyType(
    {
        "custom_scalars": "custom_scalars",
        "histograms": "histograms",
        "hparams": "hparams",
        "hparams_keras": "hparams",
        "mesh": "meshes",
        "meshes": "meshes",
        "pr_curve": "pr_curves",
        "pr_curves": "pr_curves",
        "profile": "profile",
        "projector": "projector",
        "scalars": "scalars",
        "text": "text",
    }
)

SOURCE_PATH_TYPE = str | Path
