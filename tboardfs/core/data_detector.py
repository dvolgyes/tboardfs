"""Data type detection logic for TensorBoard tensors."""

from typing import Any
from tensorboard.util import tensor_util
from loguru import logger

# TensorBoard tensor data type constants
DT_STRING = 7  # TensorFlow string tensor dtype


class TensorDataDetector:
    """Detect data types from TensorBoard tensors and metadata."""

    @staticmethod
    def is_image_tensor(tensor_proto: Any, tag: str) -> bool:
        """Check if a tensor seems to be an image.

        Args:
            tensor_proto: TensorBoard tensor proto
            tag: Tag name for context

        Returns:
            True if tensor appears to contain image data
        """
        try:
            arr = tensor_util.make_ndarray(tensor_proto)
            logger.debug(
                f"Checking if tensor '{tag}' is image: shape={arr.shape}, ndim={arr.ndim}, dtype={arr.dtype}"
            )

            # Check shape: (H, W, C) or (N, H, W, C) or (C, H, W) or (N, C, H, W)
            if arr.ndim < 2 or arr.ndim > 4:
                logger.debug(f"Tensor '{tag}' is not image: ndim is {arr.ndim}")
                return False

            # Check channels: last or second dimension should be 1, 3, or 4
            # For (H, W, C) or (N, H, W, C)
            if arr.shape[-1] in [1, 3, 4]:
                logger.debug(f"Tensor '{tag}' is image: shape[-1] is {arr.shape[-1]}")
                return True
            # For (C, H, W) or (N, C, H, W)
            if arr.ndim > 2 and arr.shape[-3] in [1, 3, 4]:
                logger.debug(f"Tensor '{tag}' is image: shape[-3] is {arr.shape[-3]}")
                return True
            if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:  # (C,H,W)
                logger.debug(f"Tensor '{tag}' is image: shape[0] is {arr.shape[0]}")
                return True

            logger.debug(f"Tensor '{tag}' is not image: no shape condition met")
            return False
        except Exception as e:
            logger.debug(f"Tensor '{tag}' to-image check failed: {e}")
            return False

    @staticmethod
    def is_video_data(image_data: bytes, tag: str) -> bool:
        """Check if image data is actually a GIF video.

        Args:
            image_data: Raw image bytes
            tag: Tag name for context

        Returns:
            True if data appears to be video (GIF)
        """
        try:
            # Check for GIF header (videos in TensorBoard are stored as GIF)
            if image_data.startswith(b"GIF87a") or image_data.startswith(b"GIF89a"):
                logger.debug(f"Detected GIF video for tag '{tag}'")
                return True

            # Additional heuristics: check tag name for video-like patterns
            video_keywords = ["video", "animation", "movie", "gif", "sequence"]
            tag_lower = tag.lower()
            for keyword in video_keywords:
                if keyword in tag_lower:
                    logger.debug(f"Tag '{tag}' contains video keyword '{keyword}'")
                    return True

            return False
        except Exception as e:
            logger.debug(f"Video detection failed for tag '{tag}': {e}")
            return False

    @staticmethod
    def is_pr_curve_tensor(tensor_proto: Any, tag: str) -> bool:
        """Check if a tensor contains PR curve data.

        Args:
            tensor_proto: TensorBoard tensor proto
            tag: Tag name for context

        Returns:
            True if tensor appears to contain PR curve data
        """
        try:
            arr = tensor_util.make_ndarray(tensor_proto)
            logger.debug(
                f"Checking if tensor '{tag}' is PR curve: shape={arr.shape}, dtype={arr.dtype}"
            )

            # PR curves have specific shape [6, N] where N is number of thresholds
            if arr.ndim == 2 and arr.shape[0] == 6:
                # Additional checks: tag name contains pr_curve, precision, recall
                tag_lower = tag.lower()
                pr_keywords = [
                    "pr_curve",
                    "precision",
                    "recall",
                    "pr",
                    "binary_classification",
                    "multi_class",
                    "model_comparison",
                    "threshold_analysis",
                ]

                for keyword in pr_keywords:
                    if keyword in tag_lower:
                        logger.debug(
                            f"Detected PR curve tensor for tag '{tag}' (keyword: {keyword})"
                        )
                        return True

                # Even without keyword match, [6, N] shape is strong indicator for PR curves
                logger.debug(f"Detected PR curve tensor for tag '{tag}' (shape-based)")
                return True

            return False
        except Exception as e:
            logger.debug(f"PR curve detection failed for tag '{tag}': {e}")
            return False

    @staticmethod
    def is_text_tensor(tensor_proto: Any) -> bool:
        """Check if a tensor contains text data.

        Args:
            tensor_proto: TensorBoard tensor proto

        Returns:
            True if tensor contains string data
        """
        return tensor_proto.dtype == DT_STRING

    @staticmethod
    def is_mesh_tensor(tag: str) -> bool:
        """Check if a tensor contains mesh data based on tag naming.

        Args:
            tag: Tag name

        Returns:
            True if tag indicates mesh data
        """
        return (
            tag.endswith("_VERTEX") or tag.endswith("_FACE") or tag.endswith("_COLOR")
        )

    @staticmethod
    def get_data_type_from_summary(value: Any, tag: str) -> str:
        """Determine data type from summary value and tag.

        Args:
            value: TensorBoard summary value
            tag: Tag name

        Returns:
            Data type string (scalar, image, video, histogram, etc.)
        """
        # Check metadata for plugin type first
        plugin_name = None
        if value.HasField("metadata"):
            plugin_name = value.metadata.plugin_data.plugin_name

        if plugin_name == "scalars" or value.HasField("simple_value"):
            return "scalar"
        elif plugin_name == "histograms" or value.HasField("histo"):
            return "histogram"
        elif plugin_name == "audio" or value.HasField("audio"):
            return "audio"
        elif plugin_name == "mesh" or TensorDataDetector.is_mesh_tensor(tag):
            return "mesh"
        elif plugin_name == "hparams":
            return "hyperparameter"
        elif plugin_name == "images" or value.HasField("image"):
            # Check if this is actually video data
            if value.HasField("image") and TensorDataDetector.is_video_data(
                value.image.encoded_image_string, tag
            ):
                return "video"
            return "image"
        elif value.HasField("tensor"):
            # Classify tensor-based data
            if TensorDataDetector.is_image_tensor(value.tensor, tag):
                return "image"
            elif TensorDataDetector.is_pr_curve_tensor(value.tensor, tag):
                return "pr_curve"
            elif TensorDataDetector.is_text_tensor(value.tensor):
                return "text"
            else:
                return "tensor"
        else:
            return "unknown"
