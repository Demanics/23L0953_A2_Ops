import pytest
from PIL import Image
from inference import predict_image

def test_predict_returns_float():
    """Test that the model returns a valid probability float for a dummy image."""
    # Create a fake 224x224 red image entirely in memory (no disk access needed)
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    # Run prediction
    result = predict_image(dummy_image)
    
    # Verify the output is a float between 0 and 1
    assert isinstance(result, float), "Prediction must be a float"
    assert 0.0 <= result <= 1.0, "Probability must be between 0.0 and 1.0"

def test_predict_handles_different_sizes():
    """Test that the inference function correctly resizes non-standard images."""
    # Create a tiny 50x50 blue image
    small_image = Image.new('RGB', (50, 50), color='blue')
    
    # The function should resize this internally without crashing
    result = predict_image(small_image)
    
    assert isinstance(result, float), "Prediction must be a float"
    assert 0.0 <= result <= 1.0, "Probability must be between 0.0 and 1.0"

def test_predict_handles_grayscale():
    """Test that the function can handle 1-channel images by converting them."""
    # Create a grayscale image
    gray_image = Image.new('L', (100, 100), color=128)
    
    result = predict_image(gray_image)
    
    assert isinstance(result, float), "Prediction must be a float"
    assert 0.0 <= result <= 1.0, "Probability must be between 0.0 and 1.0"