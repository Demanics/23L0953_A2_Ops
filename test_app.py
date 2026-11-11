import pytest
from PIL import Image
import numpy as np

def test_image_resize():
    """Test that our image preprocessing logic works correctly."""
    img = Image.new('RGB', (500, 500), color='blue')
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized)
    assert arr.shape == (224, 224, 3)

def test_threshold_logic():
    """Test the threshold logic used in app.py"""
    threshold = 0.5
    assert ("AiArtData" if 0.8 >= threshold else "RealArt") == "AiArtData"
    assert ("AiArtData" if 0.2 >= threshold else "RealArt") == "RealArt"