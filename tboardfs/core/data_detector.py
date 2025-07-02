"""Data type detection logic for TensorBoard tensors."""

from typing import Any
from tensorboard.util import tensor_util
from loguru import logger

from .constants import (
    TensorFlowDTypes,
    ImageFormats,
    TensorBoardConstants,
    DataTypeShapes,
)


class TensorDataDetector:
    """Detect data types from TensorBoard tensors and metadata."""

    @staticmethod
    def is_image_tensor(tensor_proto: Any, tag: str) -> bool:
        """Check if a tensor seems to be an image using shape heuristics.

        This method implements multi-format image detection by analyzing tensor
        dimensions and channel counts. It supports common image formats including:

        - Grayscale images (1 channel)
        - RGB images (3 channels)
        - RGBA images (4 channels)

        The algorithm checks for various tensor layouts:
        - 2D: (H, W) - Single channel grayscale
        - 3D: (H, W, C) or (C, H, W) - Multi-channel or single batch
        - 4D: (N, H, W, C) or (N, C, H, W) - Batched images

        Channel-first vs channel-last formats are both supported by checking
        multiple dimension positions for valid channel counts.

        Args:
            tensor_proto: TensorBoard tensor proto containing array data
            tag: Tag name for context and debugging

        Returns:
            True if tensor appears to contain image data based on shape analysis
        """
        try:
            # Convert protobuf tensor to numpy array for analysis
            arr = tensor_util.make_ndarray(tensor_proto)
            logger.debug(
                f"Analyzing tensor '{tag}' for image classification: "
                f"shape={arr.shape}, ndim={arr.ndim}, dtype={arr.dtype}"
            )

            # Validate dimensional constraints for image data
            # Images must be 2D, 3D, or 4D tensors
            if (
                arr.ndim < DataTypeShapes.IMAGE_MIN_DIMS
                or arr.ndim > DataTypeShapes.IMAGE_MAX_DIMS
            ):
                logger.debug(
                    f"Tensor '{tag}' rejected: invalid dimensions ({arr.ndim})"
                )
                return False

            # Multi-format channel detection algorithm
            # Strategy: Check multiple positions where channels could be located

            # Format 1: Channel-last (H, W, C) or (N, H, W, C)
            # Most common in TensorFlow/Keras
            if arr.shape[-1] in ImageFormats.VALID_CHANNELS:
                logger.debug(
                    f"Tensor '{tag}' detected as image: channel-last format "
                    f"with {arr.shape[-1]} channels"
                )
                return True

            # Format 2: Channel-first with batch (N, C, H, W)
            # Common in PyTorch, channels at position -3
            if arr.ndim > 2 and arr.shape[-3] in ImageFormats.VALID_CHANNELS:
                logger.debug(
                    f"Tensor '{tag}' detected as image: channel-first batched format "
                    f"with {arr.shape[-3]} channels"
                )
                return True

            # Format 3: Channel-first without batch (C, H, W)
            # Single image in channel-first format
            if arr.ndim == 3 and arr.shape[0] in ImageFormats.VALID_CHANNELS:
                logger.debug(
                    f"Tensor '{tag}' detected as image: channel-first single image "
                    f"with {arr.shape[0]} channels"
                )
                return True

            # No valid channel configuration found
            logger.debug(
                f"Tensor '{tag}' rejected: no valid channel configuration found. "
                f"Checked positions: [-1]={arr.shape[-1]}, "
                f"[-3]={arr.shape[-3] if arr.ndim > 2 else 'N/A'}, "
                f"[0]={arr.shape[0] if arr.ndim == 3 else 'N/A'}"
            )
            return False

        except Exception as e:
            logger.debug(f"Tensor '{tag}' image classification failed: {e}")
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
        """Check if a tensor contains Precision-Recall curve data using multi-criteria analysis.

        This method implements a sophisticated PR curve detection algorithm that combines
        structural analysis with semantic tag matching. PR curves in TensorBoard follow
        a specific format with exactly 6 components per threshold point:

        1. Threshold values (decision boundaries)
        2. Precision values (positive predictive value)
        3. Recall values (sensitivity/true positive rate)
        4. True Positive counts
        5. False Positive counts
        6. False Negative counts

        Detection Strategy:
        - Primary: Shape validation [6, N] where N = number of thresholds
        - Secondary: Semantic analysis of tag names for PR-related keywords
        - Fallback: Pure shape-based detection for unlabeled PR data

        This multi-criteria approach ensures high accuracy while maintaining
        robustness against variations in naming conventions.

        Args:
            tensor_proto: TensorBoard tensor proto containing potential PR curve data
            tag: Tag name for semantic analysis and debugging context

        Returns:
            True if tensor contains PR curve data based on combined analysis
        """
        try:
            # Convert tensor to numpy array for structural analysis
            arr = tensor_util.make_ndarray(tensor_proto)
            logger.debug(
                f"Analyzing tensor '{tag}' for PR curve classification: "
                f"shape={arr.shape}, dtype={arr.dtype}"
            )

            # Structural validation: PR curves must be 2D with exactly 6 components
            # Shape requirement: [6, N] where N is the number of threshold points
            if (
                arr.ndim == 2
                and arr.shape[0] == TensorBoardConstants.PR_CURVE_REQUIRED_COMPONENTS
            ):
                threshold_count = arr.shape[1]
                logger.debug(
                    f"Tensor '{tag}' matches PR curve structure: "
                    f"6 components × {threshold_count} thresholds"
                )

                # Semantic analysis: Search for PR-related keywords in tag name
                # This provides additional confidence for classification
                tag_lower = tag.lower()
                pr_keywords = [
                    "pr_curve",  # Direct PR curve indicator
                    "precision",  # Precision metric
                    "recall",  # Recall/sensitivity metric
                    "pr",  # Short form of precision-recall
                    "binary_classification",  # Binary classification context
                    "multi_class",  # Multi-class classification context
                    "model_comparison",  # Model evaluation context
                    "threshold_analysis",  # Threshold optimization context
                ]

                # Check for semantic indicators
                for keyword in pr_keywords:
                    if keyword in tag_lower:
                        logger.debug(
                            f"PR curve confirmed for '{tag}': structural match + "
                            f"semantic keyword '{keyword}'"
                        )
                        return True

                # High-confidence shape-based detection
                # Even without keyword match, [6, N] shape is a strong indicator
                # This handles cases where PR curves are stored without descriptive tags
                logger.debug(
                    f"PR curve detected for '{tag}' based on structural analysis: "
                    f"[6, {threshold_count}] shape matches PR curve format"
                )
                return True

            # Shape doesn't match PR curve requirements
            logger.debug(
                f"Tensor '{tag}' rejected for PR curve: "
                f"shape {arr.shape} doesn't match required [6, N] format"
            )
            return False

        except Exception as e:
            logger.debug(f"PR curve detection failed for tag '{tag}': {e}")
            return False

    @staticmethod
    def is_encoded_image_tensor(tensor_proto: Any, tag: str) -> bool:
        """Check if tensor contains encoded image bytes.

        Args:
            tensor_proto: TensorBoard tensor proto
            tag: Tag name for context

        Returns:
            True if tensor contains encoded image data (PNG, JPEG, GIF, BMP)
        """
        try:
            arr = tensor_util.make_ndarray(tensor_proto)
            if arr.dtype == "object" and len(arr) > 0:
                first_item = arr.flat[0]
                if isinstance(first_item, bytes):
                    return TensorDataDetector.is_valid_image_bytes(first_item)
        except Exception as e:
            logger.debug(f"Encoded image detection failed for tag '{tag}': {e}")
        return False

    @staticmethod
    def is_valid_image_bytes(data: bytes) -> bool:
        """Check if bytes represent valid image data.

        Args:
            data: Raw bytes data

        Returns:
            True if data has valid image format headers
        """
        # Check for common image format headers
        if data.startswith(b"\x89PNG"):  # PNG
            return True
        elif data.startswith(b"\xff\xd8\xff"):  # JPEG
            return True
        elif data.startswith(b"GIF8"):  # GIF
            return True
        elif data.startswith(b"BM"):  # BMP
            return True
        return False

    @staticmethod
    def decode_image_from_tensor(tensor_proto: Any) -> bytes | None:
        """Decode image data from tensor.

        Args:
            tensor_proto: TensorBoard tensor proto

        Returns:
            Image bytes if found, None otherwise
        """
        try:
            arr = tensor_util.make_ndarray(tensor_proto)
            if arr.dtype == "object" and len(arr) > 0:
                first_item = arr.flat[0]
                if isinstance(
                    first_item, bytes
                ) and TensorDataDetector.is_valid_image_bytes(first_item):
                    return first_item
        except Exception as e:
            logger.warning(f"Could not decode image from tensor: {e}")
        return None

    @staticmethod
    def is_text_tensor(tensor_proto: Any) -> bool:
        """Check if a tensor contains text data.

        Args:
            tensor_proto: TensorBoard tensor proto

        Returns:
            True if tensor contains string data
        """
        return tensor_proto.dtype == TensorFlowDTypes.DT_STRING

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
