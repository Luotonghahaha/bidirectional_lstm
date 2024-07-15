import torch

# 假设有一个 3D Tensor
tensor = torch.randn(1, 3)
print(tensor)

# 对第二个维度进行差值
diff_tensor = torch.diff(tensor, dim=1)

print(diff_tensor)