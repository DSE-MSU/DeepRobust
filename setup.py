from setuptools import setup
from setuptools import find_packages


with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(name = "deeprobust",
      version = "0.2.1",
      author='MSU-DSE',
      maintainer='MSU-DSE',
      description = "A PyTorch library for adversarial robustness learning for image and graph data.",
      long_description=long_description,
      long_description_content_type = "text/markdown",
      packages = find_packages(),
      url='https://github.com/DSE-MSU/DeepRobust',
      include_package_data=True,
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
          'tqdm>=3.0',
          'gensim>=3.8, <4.0'
      ],
      classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
      ],
      license="MIT",

)

