from DeepRobust.image.defense.pgdtraining import PGDtraining
from DeepRobust.image.attack.pgd import PGD
import torch
from torchvision import datasets, transforms
from DeepRobust.image.netmodels.CNN import Net
from DeepRobust.image.config import defense_params
import ipdb


"""
LOAD DATASETS
"""

train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('DeepRobust/image/defense/data', train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor()])),
                batch_size=100,
                shuffle=True)

test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('DeepRobust/image/defense/data', train=False,
            transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=1000,
            shuffle=True)

"""
TRAIN DEFENSE MODEL
"""

print('====== START TRAINING =====')

model = Net()

defense = PGDtraining(model, 'cuda')
defense.generate(train_loader, test_loader, **defense_params["PGDtraining_MNIST"])

print('====== FINISH TRAINING =====')

