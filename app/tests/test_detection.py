import os
import sys
import pytest
import numpy as np
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection import detect_plates

def test_load_model():
    """Test model loading"""
    # Patch onnxruntime before importing load_model
    with patch('onnxruntime.InferenceSession') as mock_session_class:
        # Import here to use the patched version
        from detection import load_model

        # Configure the mock
        mock_input = Mock()
        mock_input.name = 'images'

        mock_output = Mock()
        mock_output.name = 'output0'

        mock_sess = Mock()
        mock_sess.get_inputs.return_value = [mock_input]
        mock_sess.get_outputs.return_value = [mock_output]

        mock_session_class.return_value = mock_sess

        sess, inp, out = load_model('../models/yolo/model.onnx')

        assert inp == 'images'
        assert out == 'output0'
        assert sess == mock_sess
        mock_session_class.assert_called_once_with(
            '../models/yolo/model.onnx',
            providers=["CPUExecutionProvider"]
        )

def test_detect_plates_finds_plate(mock_onnx_session):
    """Test plate detection"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    boxes = detect_plates(
        mock_onnx_session,
        'images',
        'output0',
        img,
        conf=0.5,
        debug=False
    )

    assert len(boxes) == 1
    x1, y1, x2, y2, confidence = boxes[0]
    assert confidence > 0.5
    assert 0 <= x1 < x2 <= 640
    assert 0 <= y1 < y2 <= 480

def test_detect_plates_no_detections():
    """Test with no detections"""
    mock_session = Mock()
    mock_session.run.return_value = [np.zeros((1, 5, 8400), dtype=np.float32)]

    img = np.zeros((480, 640, 3), dtype=np.uint8)

    boxes = detect_plates(
        mock_session,
        'images',
        'output0',
        img,
        conf=0.5
    )

    assert len(boxes) == 0

def test_detect_plates_filters_low_confidence(mock_onnx_session):
    """Test that low confidence detections are filtered"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    boxes = detect_plates(
        mock_onnx_session,
        'images',
        'output0',
        img,
        conf=0.9,
        debug=False
    )

    assert len(boxes) == 0

def test_detect_plates_multiple_detections():
    """Test with multiple detections"""
    mock_session = Mock()

    pred = np.zeros((1, 5, 8400), dtype=np.float32)
    pred[0, :, 0] = [100, 100, 50, 30, 0.9]
    pred[0, :, 1] = [200, 200, 60, 35, 0.7]
    pred[0, :, 2] = [300, 300, 55, 32, 0.6]
    mock_session.run.return_value = [pred]

    img = np.zeros((480, 640, 3), dtype=np.uint8)

    boxes = detect_plates(
        mock_session,
        'images',
        'output0',
        img,
        conf=0.5
    )

    assert len(boxes) == 3
    for box in boxes:
        assert box[4] > 0.5

def test_detect_plates_confidence_normalization():
    """Test confidence normalization when > 1"""
    mock_session = Mock()

    pred = np.zeros((1, 5, 8400), dtype=np.float32)
    pred[0, :, 0] = [320, 240, 100, 50, 512.0]
    mock_session.run.return_value = [pred]

    img = np.zeros((480, 640, 3), dtype=np.uint8)

    boxes = detect_plates(
        mock_session,
        'images',
        'output0',
        img,
        conf=0.5
    )

    assert len(boxes) == 1
    assert 0.7 < boxes[0][4] < 0.9

def test_detect_plates_bbox_clamping():
    """Test that bounding boxes are clamped to image boundaries"""
    mock_session = Mock()

    # Prediction that would create bbox outside image
    pred = np.zeros((1, 5, 8400), dtype=np.float32)
    # Center at 10,10 with large width/height would go negative
    pred[0, :, 0] = [10, 10, 100, 100, 0.8]
    mock_session.run.return_value = [pred]

    img = np.zeros((480, 640, 3), dtype=np.uint8)

    boxes = detect_plates(
        mock_session,
        'images',
        'output0',
        img,
        conf=0.5
    )

    assert len(boxes) == 1
    x1, y1, x2, y2, _ = boxes[0]

    # Coordinates should be clamped to [0, width] and [0, height]
    assert x1 >= 0
    assert y1 >= 0
    assert x2 <= 640
    assert y2 <= 480
