
Introduction to Image Defense with Examples
============
In this section, we introduce the image defenses provided in DeepRobust.

.. contents::
    :local: 

Adversarial Training Based Defenses for Image Classification
------------
DeepRobust provides the following adversarial training based defenses algorithms:

- :class:`deeprobust.image.defense.FGSMtraining`
- :class:`deeprobust.image.defense.PGDtraining`
- :class:`deeprobust.image.defense.Fast`
- :class:`deeprobust.image.defense.TRADES`
- :class:`deeprobust.image.defense.AWP`
- :class:`deeprobust.image.attack.Universal`

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


