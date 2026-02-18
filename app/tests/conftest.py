import os
import sys
import pytest
import cv2
import numpy as np
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_plate_image():
    """Create a synthetic license plate image"""
    img = np.ones((110, 520, 3), dtype=np.uint8) * 255
    img[:, :40] = [139, 69, 19]  # Blue EU strip
    cv2.putText(img, 'GP462TC', (60, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return img

@pytest.fixture
def sample_ir_image():
    """Create a synthetic infrared (grayscale) image"""
    img = np.ones((110, 520, 3), dtype=np.uint8) * 200
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = 10  # Very low saturation
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

@pytest.fixture
def mock_easyocr_reader():
    """Mock EasyOCR reader"""
    reader = Mock()
    reader.readtext.return_value = [
        ([(10, 10), (100, 10), (100, 50), (10, 50)], 'GR571XC', 0.95)
    ]
    return reader

@pytest.fixture
def mock_onnx_session():
    """Mock ONNX session with proper configuration"""
    session = Mock()

    # Configure inputs
    mock_input = Mock()
    mock_input.name = 'images'  # Return string, not Mock
    session.get_inputs.return_value = [mock_input]

    # Configure outputs
    mock_output = Mock()
    mock_output.name = 'output0'  # Return string, not Mock
    session.get_outputs.return_value = [mock_output]

    # Configure predictions
    pred = np.zeros((1, 5, 8400), dtype=np.float32)
    pred[0, :, 0] = [320, 240, 100, 50, 0.8]
    session.run.return_value = [pred]

    return session
