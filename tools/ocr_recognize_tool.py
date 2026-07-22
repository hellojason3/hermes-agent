#!/usr/bin/env python3
"""
OCR Recognition Tool — Lenovo OCR service (gRPC)

Provides image-to-text recognition via the local PaddleOCR backend.
Registered as ``ocr_recognize`` tool for use by Hermes agents.
"""

import logging
import sys
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# gRPC imports (lazy)
# ---------------------------------------------------------------------------
_grpc_available = None


def _ensure_grpc() -> bool:
    global _grpc_available
    if _grpc_available is not None:
        return _grpc_available
    try:
        import grpc  # noqa: F401
        sys.path.insert(0, "/home/peter/tmp/ocr_generated")
        import ocr_pb2  # noqa: F401
        import ocr_pb2_grpc  # noqa: F401
        _grpc_available = True
    except ImportError as e:
        logger.warning("OCR gRPC dependencies not available: %s", e)
        _grpc_available = False
    return _grpc_available


OCR_SERVICE_ADDR = "192.168.8.148:50051"
OCR_TIMEOUT = 60
OCR_HEALTH_TIMEOUT = 10

# ---------------------------------------------------------------------------
# Required environment variable: none (service address is hardcoded for now)
# ---------------------------------------------------------------------------


def _read_image_bytes(image_path: str) -> Optional[bytes]:
    """Read an image file and return raw bytes."""
    try:
        with open(image_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("Image not found: %s", image_path)
        return None
    except PermissionError:
        logger.error("Permission denied reading: %s", image_path)
        return None
    except Exception as e:
        logger.error("Failed to read image %s: %s", image_path, e)
        return None


def check_ocr_requirements() -> bool:
    """True if the OCR gRPC service is reachable."""
    if not _ensure_grpc():
        return False
    try:
        import grpc
        import ocr_pb2
        import ocr_pb2_grpc
        channel = grpc.insecure_channel(OCR_SERVICE_ADDR)
        stub = ocr_pb2_grpc.OcrStub(channel)
        health = stub.Health(ocr_pb2.HealthRequest(), timeout=OCR_HEALTH_TIMEOUT)
        channel.close()
        return health.ready and health.workers_ready > 0
    except Exception as e:
        logger.debug("OCR service unreachable: %s", e)
        return False


def _handle_ocr_recognize(image_path: str) -> str:
    """Call the OCR service and return recognized text.

    Args:
        image_path: Local path to the image file (JPEG/PNG).

    Returns:
        JSON string with recognition results.
    """
    import json

    if not _ensure_grpc():
        return json.dumps({
            "success": False,
            "error": "OCR gRPC dependencies not installed. Run: pip install grpcio grpcio-tools",
            "text": "",
        })

    image_bytes = _read_image_bytes(image_path)
    if image_bytes is None:
        return json.dumps({
            "success": False,
            "error": f"Cannot read image: {image_path}",
            "text": "",
        })

    try:
        import grpc
        import ocr_pb2
        import ocr_pb2_grpc

        channel = grpc.insecure_channel(OCR_SERVICE_ADDR)
        stub = ocr_pb2_grpc.OcrStub(channel)

        resp = stub.Recognize(
            ocr_pb2.RecognizeRequest(
                request_id="hermes-ocr",
                image_bytes=image_bytes,
                lang="",
                include_lines=False,
                skip_cache=False,
            ),
            timeout=OCR_TIMEOUT,
        )
        channel.close()

        if resp.status in (ocr_pb2.DONE, ocr_pb2.CACHED):
            return json.dumps({
                "success": True,
                "text": resp.text,
                "confidence": resp.confidence,
                "engine": resp.engine,
                "version": resp.version,
                "status": "DONE" if resp.status == ocr_pb2.DONE else "CACHED",
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "success": False,
                "error": f"OCR returned status: {resp.status}",
                "text": resp.text or "",
            }, ensure_ascii=False)

    except ImportError as e:
        return json.dumps({
            "success": False,
            "error": f"gRPC import error: {e}",
            "text": "",
        })
    except Exception as e:
        logger.exception("OCR recognition failed")
        return json.dumps({
            "success": False,
            "error": f"OCR error: {e}",
            "text": "",
        })


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

OCR_RECOGNIZE_SCHEMA = {
    "name": "ocr_recognize",
    "description": (
        "Extract/recognize text from an image using the local OCR service. "
        "Pass the local file path of the image (e.g. a receipt, screenshot, "
        "or scanned document). Returns recognized text with confidence score. "
        "Supported formats: JPEG, PNG. Max image size: 32 MB."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": (
                    "The absolute local file path to the image to be OCR'd. "
                    "Must be a JPEG or PNG file accessible on the local filesystem."
                ),
            },
        },
        "required": ["image_path"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error


def _handle_ocr_recognize_wrapper(image_path: str) -> str:
    """Tool handler wrapper with error handling."""
    try:
        return _handle_ocr_recognize(image_path)
    except Exception as e:
        logger.exception("Unhandled error in ocr_recognize")
        return tool_error(f"OCR tool failed: {e}")


registry.register(
    name="ocr_recognize",
    toolset="ocr",
    schema=OCR_RECOGNIZE_SCHEMA,
    handler=_handle_ocr_recognize_wrapper,
    check_fn=check_ocr_requirements,
    requires_env=[],
    is_async=False,
    emoji="🔍",
)
