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

# List of including algorithms
[Image classification](https://github.com/I-am-Bot/DeepRobust/tree/master/DeepRobust/image)

[Graph classification](https://github.com/I-am-Bot/DeepRobust/tree/master/DeepRobust/graph)



# Usage
1. Train model

    Example: Train a simple CNN model on MNIST dataset for 20 epoch on gpu.
    ```python
    import DeepRobust.image.netmodels.train_model as trainmodel
    trainmodel.train('CNN', 'MNIST', 'cuda', 20)
    ```
    Model would be saved in deeprobust/trained_models/.

2. Instantiated attack methods and defense methods.

    Example: Generate adversary example with PGD attack.
    ```python
    from DeepRobust.image.attack.pgd import PGD
    from DeepRobust.image.config import attack_params

    adversary = PGD(model, device)
    Adv_example = adversary.generate(X, Y, **attack_params['PGD_CIFAR10']
    ```

    Example: Train defense model.
    ```python
    from DeepRobust.image.defense.pgdtraining import PGDtraining
    from DeepRobust.image.config import defense_params

    model = Net()
    defense = PGDtraining(model, 'cuda')
    defense.generate(train_loader, test_loader, **defense_params["PGDtraining_MNIST"])
    ```
    More example code can be found in deeprobust/tutorials.

3. Use our evulation program to test attack algorithm against defense.

    Example:
    ```
    python -m DeepRobust.image.evaluation_attack --attack_method --attack_model --path --file_name --epsilon --batch_num --batch_size --num_steps --step_size --random_targeted --device
    ```
    Use python -m DeepRobust.image.evaluation_attack --help for more information.