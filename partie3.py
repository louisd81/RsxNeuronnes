import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader

x, y = x_validation[0:1], y_validation[0:1]
x.requires_grad = True

output = model(x)
loss = output[0, y]

loss.backward()
gradient = x.grad.data

gradient_abs = gradient.abs().squeeze()

plt.imshow(gradient_abs.numpy(), cmap='hot')
plt.title("Valeur absolue du gradient")
plt.colorbar()
plt.show()

# Le gradient met en évidence les parties de l'image qui influencent le plus la décision du modèle pour une classe donnée.
# Les pixels avec des valeurs absolues élevées dans le gradient ont un impact plus important sur la prédiction.

# Si le modèle est bien entraîné, le gradient se concentre sur les zones pertinentes de l'image. 
# En revanche, un modèle mal entraîné ou sur-apprenant produit des gradients bruités, mettant en avant des zones non pertinentes.
