# 모델 정의
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(784, 10)  # 예시로 784 입력을 받아서 10개의 클래스를 예측

    def forward(self, x):
        return self.fc(x)
