from types import MappingProxyType

TENSORBOARD_EVENT_GLOB = "events.out.tfevents*"
DT_UINT8 = 4
# TensorProto dtype of a string tensor.
DT_STRING = 7
LARGE_TENSOR_THRESHOLD = 1024
# A single tensor value is repeated to fill its shape up to this size.
MAX_REPEATED_TENSOR_BYTES = 1024
SCALAR_COLUMNS = ("epoch", "step", "wall_time", "relative_time", "value")
SCALAR_FORMATS = ("json", "tsv", "csv", "npz")
DEFAULT_SCALAR_FORMATS = ("json", "tsv", "npz")
MAX_STEP_DIGITS = 20  # an int64 step and its sign
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
DEFAULT_CACHE_SIZE_MB = 1024
MAX_INDEX_BYTES = 512 << 20
SOURCE_DIGEST_BYTES = 16  # XXH3-128
DEFAULT_DISCOVERY_SECONDS = 150.0
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level}</level> | <level>{message}</level>"
)

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
    "raw",
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
        "tensor": "tensors",
        "tensors": "tensors",
        "text": "text",
        "video": "videos",
        "videos": "videos",
    }
)
