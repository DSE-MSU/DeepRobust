from setuptools import setup
from setuptools import find_packages


with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(name = "deeprobust",
      version = "0.1.0",
      author='MSU-DSE',
      maintainer='MSU-DSE',
      description = "A PyTorch library for adversarial robustness learning for image and graph data.",
      long_description=long_description,
      long_description_content_type = "text/markdown",
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

