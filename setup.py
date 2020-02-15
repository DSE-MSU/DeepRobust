from setuptools import setup
from setuptools import find_packages

setup(name = "deeprobust",
      version = "0",
      description = "A pytorch library for adversarial robustness learning.",
      packages = find_packages(),
      install_requires = [
          'matplotlib>=3.1.1',
          'numpy>=1.17.1',
          'torch>=1.2.0',
          'scipy>=1.3.1',
          'torchvision>=0.4.0',
          'texttable>=1.6.2',
          'networkx>=2.4',
          'numba>=0.48.0',
          'Pillow>=7.0.0',
          'scikit_learn>=0.22.1',
          'scikit-image>=0.0',
          'tensorboardX>=2.0',
          'tqdm>=4.42.1'
      ]
)

