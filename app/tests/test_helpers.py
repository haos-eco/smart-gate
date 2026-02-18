import os
import sys
import json
import tempfile
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import ensure_dir

def test_ensure_dir_creates_directory():
    """Test that ensure_dir creates directories"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, 'subdir', 'file.txt')
        ensure_dir(test_path)
        assert os.path.exists(os.path.dirname(test_path))

def test_ensure_dir_handles_existing():
    """Test that ensure_dir doesn't fail on existing directories"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, 'file.txt')
        ensure_dir(test_path)
        ensure_dir(test_path)  # Should not raise
        assert os.path.exists(tmpdir)

def test_ensure_dir_empty_path():
    """Test ensure_dir with empty path"""
    ensure_dir('')  # Should not raise
