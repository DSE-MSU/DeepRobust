from setuptools import setup
from setuptools import find_packages

setup(name = "deeprobust",
      version = "0",
      description = "A pytorch library for adversarial robustness learning.",
      packages = find_packages(),
      install_requires = [
          'numpy',
          'torchvision>=0.4.0',
          'matplotlib',
          'ipdb',
      ]
)

