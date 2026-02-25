"""
TrOCR-based OCR for license plates.
Uses microsoft/trocr-base-printed — optimized for printed text.
Falls back to EasyOCR if model fails to load.
"""

import cv2
import numpy as np

_TROCR_PROCESSOR = None
_TROCR_MODEL = None
_TROCR_AVAILABLE = None  # None = not yet checked, True/False = result


def load_trocr() -> bool:
    """
    Load TrOCR processor and model.
    Returns True on success, False on failure.
    Idempotent — safe to call multiple times.
    """
    global _TROCR_PROCESSOR, _TROCR_MODEL, _TROCR_AVAILABLE

    if _TROCR_AVAILABLE is not None:
        return _TROCR_AVAILABLE

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        print("Loading TrOCR model (microsoft/trocr-base-printed)...")
        print("  First run may take a few minutes to download (~400MB)")

        _TROCR_PROCESSOR = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-printed"
        )
        _TROCR_MODEL = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-printed"
        )
        _TROCR_MODEL.eval()

        _TROCR_AVAILABLE = True
        print("✅ TrOCR loaded successfully")
        return True

    except Exception as e:
        print(f"⚠️  TrOCR unavailable: {e}")
        _TROCR_AVAILABLE = False
        return False


def trocr_infer(img_bgr, debug=False):
    """
    Run TrOCR inference on a preprocessed plate crop.
    Returns (text, confidence) or (None, 0.0) on failure.
    """
    if not load_trocr():
        return None, 0.0

    try:
        import torch
        from PIL import Image

        # TrOCR expects RGB PIL image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        pixel_values = _TROCR_PROCESSOR(
            images=pil_img, return_tensors="pt"
        ).pixel_values

        with torch.no_grad():
            outputs = _TROCR_MODEL.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=20,
            )

        # Decode text
        text = _TROCR_PROCESSOR.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0]

        # Estimate confidence from sequence scores
        if outputs.scores:
            import math

            token_probs = [
                torch.softmax(score, dim=-1).max().item() for score in outputs.scores
            ]
            # Geometric mean of token probabilities
            confidence = math.exp(
                sum(math.log(max(p, 1e-9)) for p in token_probs) / len(token_probs)
            )
        else:
            confidence = 0.0

        if debug:
            print(f"  TrOCR raw: '{text}' (confidence: {confidence:.3f})")

        return text.strip(), confidence

    except Exception as e:
        if debug:
            print(f"  TrOCR inference error: {e}")
        return None, 0.0
