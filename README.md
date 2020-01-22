# DeepRobust
For more details about attacks and defenses, you can read this paper.

[Adversarial Attacks and Defenses in Images, Graphs and Text: A Review](https://arxiv.org/pdf/1909.08072.pdf)

We would be glad if you find our work useful and cite the paper.

```
@article{xu2019adversarial,
  title={Adversarial attacks and defenses in images, graphs and text: A review},
  author={Xu, Han and Ma, Yao and Liu, Haochen and Deb, Debayan and Liu, Hui and Tang, Jiliang and Jain, Anil},
  journal={arXiv preprint arXiv:1909.08072},
  year={2019}
}
```

# Requirements
* `python3`
* `numpy`
* `pytorch v1.2.0`
* `torchvision v0.4.0`
* `matplotlib`


# Support Datasets
- MNIST
- CIFAR-10
- ImageNet

# Support Networks
- SimpleCNN
- ResNet

# Usage
- Attack
1. Import algorithm directely and implement your own program.

    Example to generate an adversary example with PGD attack:
    ```python
    from DeepRobust.image.attack.pgd import PGD
    from DeepRobust.image.config import attack_params

    adversary = PGD(model, device)
    Adv_example = adversary.generate(X, Y, **attack_params['PGD_CIFAR10')]
    ```
    More example code can be found in "DeepRobust/tutorials".


2. Use our evulation program to test attack algorithm against defense.

    Example:
    
    ```
    python -m DeepRobust.image.evaluation_attack --attack_method --attack_model --dataset --epsilon --batch_num --batch_size --num_steps --step_size --device --path
    ```
