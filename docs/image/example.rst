
Image Attack and Defense
============
We introduce the usage of attacks and defense API in image package.

.. contents::
    :local: 


Attack Example
------------

    .. code-block:: python
       
       from deeprobust.image.attack.pgd import PGD
       from deeprobust.image.config import attack_params
       from deeprobust.image.utils import download_model
       import torch
       import deeprobust.image.netmodels.resnet as resnet

       URL = "https://github.com/I-am-Bot/deeprobust_model/raw/master/CIFAR10_ResNet18_epoch_50.pt"
       download_model(URL, "$MODEL_PATH$")

       model = resnet.ResNet18().to('cuda')
       model.load_state_dict(torch.load("$MODEL_PATH$"))
       model.eval()

       transform_val = transforms.Compose([transforms.ToTensor()])
       test_loader  = torch.utils.data.DataLoader(
                    datasets.CIFAR10('deeprobust/image/data', train = False, download=True,
                    transform = transform_val),
                    batch_size = 10, shuffle=True)

       x, y = next(iter(test_loader))
       x = x.to('cuda').float()

       adversary = PGD(model, device)
       Adv_img = adversary.generate(x, y, **attack_params['PGD_CIFAR10'])

Defense Example
------------

    .. code-block:: python
       
       model = Net()
       train_loader = torch.utils.data.DataLoader(
                       datasets.MNIST('deeprobust/image/defense/data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()])),
                       batch_size=100, shuffle=True)
       test_loader = torch.utils.data.DataLoader(
                      datasets.MNIST('deeprobust/image/defense/data', train=False,
                      transform=transforms.Compose([transforms.ToTensor()])),
                      batch_size=1000,shuffle=True)
       
       defense = PGDtraining(model, 'cuda')
       defense.generate(train_loader, test_loader, **defense_params["PGDtraining_MNIST"])


