import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader

#Importation des sets de données
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

#Visualisation d'1 donnée aléatoire pour vérifier que le bon set est importé
idx = torch.randint(len(data_x), (1,)).item()
print('input data shape', data_x.shape)
print('class', data_y[idx].item())
plt.imshow(data_x[idx].permute(1, 2, 0).squeeze())

class CNNModel(nn.Module):
    #Initialisation du modèle utilisée
    def __init__(self, num_convolutions):
        super(CNNModel, self).__init__()
        #Définition des 3 convolutions que l'on va utilisé pour les tests
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) if num_convolutions == 3 else None
        #Définition de la couche de pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #Réduction de la dimension d'entrée pour pouvoir déterminer a quelle classe appartient l'image
        self.fc1 = nn.Linear(64 * 14 * 14 if num_convolutions == 2 else 128 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    #Fonction de passage d'une image a une prédiction de classe par convolution puis aplatissement
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        if self.conv3 is not None:
            x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Chargement des données
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

# Configurations à tester
combinations = [
    {'num_convolutions': 2, 'batch_size': 32},
    {'num_convolutions': 2, 'batch_size': 64},
    {'num_convolutions': 3, 'batch_size': 32},
    {'num_convolutions': 3, 'batch_size': 64}
]

plt.figure(figsize=(10, 6))

for comb in combinations:
    num_convolutions = comb['num_convolutions']
    batch_size = comb['batch_size']

    # DataLoader pour l'entraînement
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(train_loader)  # Itérateur infini

    # Initialisation modèle
    model = CNNModel(num_convolutions)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Liste pour stocker les pertes moyennes tous les 100 pas
    avg_losses = []

    # Entraînement sur 1000 pas
    for step in range(1000):
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

        # Évaluation tous les 100 pas
        if (step + 1) % 100 == 0:
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                eval_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
                for eval_images, eval_labels in eval_loader:
                    eval_outputs = model(eval_images)
                    total_loss += criterion(eval_outputs, eval_labels).item() * eval_images.size(0)

            avg_loss = total_loss / len(train_dataset)
            avg_losses.append(avg_loss)
            print(f"Combinaison {comb}, Step {step+1}, Loss: {avg_loss:.4f}")
            model.train()

    # Tracé de la courbe
    plt.plot(range(100, 1001, 100), avg_losses, label=f"Conv={num_convolutions}, Batch={batch_size}")

plt.xlabel("Nombre de pas de gradient")
plt.ylabel("Fonction de coût (moyenne sur le dataset)")
plt.title("Évolution de la perte par configuration")
plt.legend()
plt.grid(True)
plt.show()

#Après éxecution il semble que les hyper paramètres 2 convolutions et taille du batch de 64 soient les plus adaptées.