import torch
import torchvision
import torchvision.transforms as transforms

a = torch.randn(2, 2, 3)
b = torch.randn(2, 2, 3)
c = torch.cat((a, b), 2)
d = torch.stack((a, b), 2)
print(a)
print(b)
print(c)
print(d)
print(c.shape)
print(d.shape)