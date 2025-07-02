"""Image processing utilities for TensorBoard data.

This module contains utilities for processing images from TensorBoard tensors,
including tensor-to-image conversion and image format detection.
"""

import io
from typing import Any

import magic
import numpy as np
from PIL import Image
from loguru import logger
from tensorboard.util import tensor_util


class ImageProcessor:
    """Handles image processing operations for TensorBoard data."""

    @staticmethod
    def decode_image_from_tensor(tensor_proto: Any) -> bytes | None:
        """Decode an image from a tensor_proto.

        Args:
            tensor_proto: TensorFlow tensor protocol buffer containing image data.

        Returns:
            PNG-encoded image bytes if successful, None if failed.
        """
        try:
            arr = tensor_util.make_ndarray(tensor_proto)
            logger.debug(f"Decoding image tensor: shape={arr.shape}, dtype={arr.dtype}")

            # Squeeze batch dimension if present
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr.squeeze(0)

            # Handle channel-first format (C, H, W) -> (H, W, C)
            if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
                arr = np.transpose(arr, (1, 2, 0))

            # Handle grayscale with no channel dim
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)

            # Normalize to 0-255
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                logger.debug("Normalizing float tensor to uint8")
                arr = (arr * 255).astype(np.uint8)

            # Ensure it is uint8
            if arr.dtype != np.uint8:
                logger.warning(f"Unsupported image tensor dtype: {arr.dtype}")
                return None

            # Handle single-channel (grayscale)
            if arr.shape[-1] == 1:
                arr = arr.squeeze(axis=-1)

            logger.debug(f"Final array shape for Pillow: {arr.shape}")
            # Convert to image
            img = Image.fromarray(arr)

            # Save to bytes buffer
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            logger.debug("Successfully decoded tensor to PNG")
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Failed to decode image from tensor: {e}")
            return None

    @staticmethod
    def get_image_extension(image_bytes: bytes, tag: str = "unknown") -> str:
        """Determine image extension from bytes using python-magic.

        Args:
            image_bytes: Raw image bytes to analyze.
            tag: Tag name for logging purposes.

        Returns:
            File extension (e.g., 'png', 'jpg', 'gif') or 'bin' if unknown.
        """
        # Use python-magic to detect the actual file type
        mime_type = magic.from_buffer(image_bytes, mime=True)
        logger.debug(
            f"MIME type for image bytes (tag='{tag}', len={len(image_bytes)}): {mime_type}"
        )

        # Map MIME types to extensions
        mime_to_ext = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/gif": "gif",
            "image/bmp": "bmp",
            "image/tiff": "tiff",
            "image/webp": "webp",
            "image/svg+xml": "svg",
        }

        ext = mime_to_ext.get(mime_type, "bin")
        if ext == "bin":
            # This is expected for non-image string tensors, so log at debug level.
            logger.debug(
                f"Could not determine image type from MIME type '{mime_type}'. Returning .bin"
            )
        return ext
