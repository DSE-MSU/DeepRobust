# Setup
```
git clone https://github.com/DSE-MSU/DeepRobust.git
cd DeepRobust
python setup.py install
```

# Full README
[click here](https://github.com/DSE-MSU/DeepRobust)

# Attack Methods  
|   Attack Methods   | Attack Type | Apply Domain | Links |
|--------------------|-------------|--------------|------|
| LBFGS attack | White-Box | Image Classification | [Intriguing Properties of Neural Networks](https://arxiv.org/pdf/1312.6199.pdf?not-changed)|
| FGSM attack | White-Box | Image Classification | [Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572.pdf) |
| PGD attack | White-Box | Image Classification | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083.pdf) |
| DeepFool attack | White-Box | Image Classification | [DeepFool: a simple and accurate method to fool deep neural network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.pdf) |
| CW attack | White-Box | Image Classification | [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/pdf/1608.04644.pdf) |
| One pixel attack | White-Box | Image Classification | [One pixel attack for fooling deep neural networks](https://arxiv.org/pdf/1710.08864.pdf) |
| BPDA attack | White-Box | Image Classification | [Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples](https://arxiv.org/pdf/1802.00420.pdf) |
| Universal attack | White-Box | Image Classification | [Universal adversarial perturbations](https://arxiv.org/pdf/1610.08401.pdf) |
| Nattack | Black-Box | Image Classification | [NATTACK: Learning the Distributions of Adversarial Examples for an Improved Black-Box Attack on Deep Neural Networks](https://arxiv.org/pdf/1905.00441.pdf) |

# Defense Methods
|   Defense Methods   | Defense Type | Apply Domain | Links |
|---------------------|--------------|--------------|------|
| FGSM training | Adverserial Training | Image Classification | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083.pdf) |
| Fast(an improved version of FGSM training) | Adverserial Training | Image Classification | [Fast is better than free: Revisiting adversarial training](https://openreview.net/attachment?id=BJx040EFvH&name=original_pdf) |
| PGD training | Adverserial Training | Image Classification | [Intriguing Properties of Neural Networks](https://arxiv.org/pdf/1312.6199.pdf?not-changed)|
| YOPO(an improved version of PGD training) | Adverserial Training | Image Classification | [You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle](https://arxiv.org/pdf/1905.00877.pdf) |
| TRADES | Adverserial Training | Image Classification | [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/pdf/1901.08573.pdf) |
| Thermometer Encoding | Gradient Masking | Image Classification | [Thermometer Encoding:One Hot Way To Resist Adversarial Examples](https://openreview.net/pdf?id=S18Su--CW) |
| LID-based adversarial classifier | Detection | Image Classification | [Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality](https://arxiv.org/pdf/1801.02613.pdf) |

# Support Datasets
- MNIST
- CIFAR-10
- ImageNet

# Support Networks
- CNN
- ResNet(ResNet18, ResNet34, ResNet50)
- VGG
- DenseNet

