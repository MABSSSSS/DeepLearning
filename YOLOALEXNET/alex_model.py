import torch.nn as nn
import torchvision.models as models

def get_alexnet(num_classes=2):
    model = models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
