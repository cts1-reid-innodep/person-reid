from torch.nn import Linear
from torchvision.models import ResNet50_Weights, resnet50


def initialize_model(model_name: str, num_classes: int):
    """
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = Linear(model.fc.in_features, num_classes)
        return model

    raise NotImplementedError(f"No implementation for model '{model_name}'")
