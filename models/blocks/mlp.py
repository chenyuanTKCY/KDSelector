import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# 初始化MLP
# mlp = MLP(input_dim=512)
# x_ts = mlp(inputs)  # 经过MLP的转换后，x_ts的形状应为 [256, 128]
