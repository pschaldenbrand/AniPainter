import torch
import torch.nn as nn
import torchvision.models as models


import cv2
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = 'stroke_placement/stroke_placer.pt'

class StrokePlacer(nn.Module):

    def __init__(self, pretrained=True):
        super(StrokePlacer, self).__init__()

        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.model.fc = nn.Linear(512,4)
        if pretrained:
            self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.to(device)
    def forward(self, x):
        return self.model(x)

    def save(self):
        torch.save(self.model.state_dict(), MODEL_PATH)

    def create_sample(self, reference_image, canvas, r, x0, y0, color):
        # Channels last to first
        reference_image = reference_image.permute(2, 0, 1)
        canvas = canvas.permute(2, 0, 1)

        # Turn r, x0, y0, color into channels
        r = torch.ones(1, canvas.shape[1], canvas.shape[2], device=device, dtype=torch.float) * r
        x0 = torch.ones(1, canvas.shape[1], canvas.shape[2], device=device, dtype=torch.float) * x0
        y0 = torch.ones(1, canvas.shape[1], canvas.shape[2], device=device, dtype=torch.float) * y0
        color = torch.ones(3, canvas.shape[1], canvas.shape[2], device=device, dtype=torch.float) * color[:,None,None]

        # Normalize ref_image, r, x0, y0 to 0-1
        reference_image = reference_image / 255.
        r = r / canvas.shape[1]
        x0 = x0 / canvas.shape[1]
        y0 = y0 / canvas.shape[1]

        return torch.cat((r, x0, y0, color, reference_image, canvas), 0)