import torch
from src.models.model import SimpleCNN

def test_model_output_shape():
    model = SimpleCNN()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 1)