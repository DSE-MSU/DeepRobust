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
* `matplotlib`

# Support Datasets
- MNIST
- CIFAR-10
- ImageNet

# Support Networks
- SampleCNN
- ResNet

# Attack Methods  
|   Attack Methods   | Attack Type | Apply Domain | Links |
|--------------------|-------------|--------------|------|
| LBFGS attack | White-Box | Image Classification | [Intriguing Properties of Neural Networks](https://arxiv.org/pdf/1312.6199.pdf?not-changed)|
| FGSM attack | White-Box | Image Classification | [Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572.pdf) |
| PGD attack | White-Box | Image Classification | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083.pdf) |
| DeepFool attack | White-Box | Image Classification | [DeepFool: a simple and accurate method to fool deep neural network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.pdf) |
| CW attack | White-Box | Image Classification | [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/pdf/1608.04644.pdf) |
| Nattack | Black-Box | Image Classification | [NATTACK: Learning the Distributions of Adversarial Examples for an Improved Black-Box Attack on Deep Neural Networks](https://arxiv.org/pdf/1905.00441.pdf) |

# Defense Methods
|   Defense Methods   | Defense Type | Apply Domain | Links |
|---------------------|--------------|--------------|------|
| FGSM training | Adverserial Training | Image Classification | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083.pdf) |
| PGD training | Adverserial Training | Image Classification | [Intriguing Properties of Neural Networks](https://arxiv.org/pdf/1312.6199.pdf?not-changed)|
| YOPO | Adverserial Training | Image Classification | [You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle](https://arxiv.org/pdf/1905.00877.pdf) |
| TRADES | Adverserial Training | Image Classification | [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/pdf/1901.08573.pdf) |


