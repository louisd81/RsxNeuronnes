import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader



#Pre-existing code
dataset_id = 2 # valeur à choisir dans {0, 1, 2}
dataset_loaders = [torchvision.datasets.USPS, torchvision.datasets.CIFAR10,
                   torchvision.datasets.FashionMNIST]
dataset_loader = dataset_loaders[dataset_id]


batch_size = 1000
data_x, data_y = next(iter(DataLoader(
    dataset_loader(root='./', download=True,
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor()])),
                       batch_size=batch_size, shuffle=True)))

train_dataset = TensorDataset(data_x, data_y)

idx = torch.randint(len(data_x), (1,)).item()
print('input data shape', data_x.shape)
print('class', data_y[idx].item())
plt.imshow(data_x[idx].permute(1, 2, 0).squeeze())

class CNNModel(nn.Module):
    def __init__(self, num_convolutions):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) if num_convolutions == 3 else None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14 if num_convolutions == 2 else 128 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        if self.conv3 is not None:
            x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Chargement des ensembles d'entraînement et de validation
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
dataset_loader = datasets.FashionMNIST
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validation_loader = DataLoader(
    dataset_loader(root='./', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=False)

# Configuration du modèle
model = CNNModel(2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Calculer le taux de bonne classification
def accuracy(loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Entraînement
train_accuracies = []
validation_accuracies = []
steps = 10000

for step in range(steps):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        # Pas d'entraînement
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

     # Calcul des taux de bonne classification tous les 1000 pas
        if step % 1000 == 0:
          train_acc = accuracy(train_loader, model)
          val_acc = accuracy(validation_loader, model)
          train_accuracies.append(train_acc)
          validation_accuracies.append(val_acc)
          print(f"Step {step}: Train Accuracy = {train_acc:.2f}%, Validation Accuracy = {val_acc:.2f}%")

# Affichage
plt.plot(range(1000, steps + 1, 1000), train_accuracies, label="Ensemble d'entraînement")
plt.plot(range(1000, steps + 1, 1000), validation_accuracies, label="Ensemble de validation")
plt.xlabel("Nombre de pas de gradient")
plt.ylabel("Taux de bonne classification (%)")
plt.title("Performance du modèle au fil de l'entraînement")
plt.legend()
plt.show()

# On peut voir ici que le modèle continue d'apprendre et continue de s'améliorer au fil du nombre 
# de pas de gradient. Toutefois, le modèle obtenu au bout de 3000 pas est celui qui a la meilleure 
# précision sur l'ensemble de validation et c'est donc celui que l'on peut préconiser pour le déploiement.