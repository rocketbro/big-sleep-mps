import sys
import platform
from setuptools import setup, find_packages

sys.path[0:0] = ['big_sleep']
from version import __version__

# Detect if running on Apple Silicon
is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'

# Set appropriate torch and torchvision requirements based on platform
if is_apple_silicon:
    # For Apple Silicon, require torch 2.0+ which has good MPS support
    torch_requirement = 'torch>=2.0.0'
    torchvision_requirement = 'torchvision>=0.15.0'
else:
    # For other platforms, maintain existing requirements
    torch_requirement = 'torch>=1.7.1'
    torchvision_requirement = 'torchvision>=0.8.2'

setup(
  name = 'big-sleep',
  packages = find_packages(),
  include_package_data = True,
  entry_points={
    'console_scripts': [
      'dream = big_sleep.cli:main',
    ],
  },
  version = __version__,
  license='MIT',
  description = 'Big Sleep - Text to Image Generation with CLIP and BigGAN',
  author = 'Ryan Murdock, Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/big-sleep',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'text to image',
    'generative adversarial networks',
    'apple silicon',
    'mps'
  ],
  install_requires=[
    torch_requirement,
    'einops>=0.3',
    'fire',
    'ftfy',
    'pytorch-pretrained-biggan>=0.1.0',
    'regex',
    torchvision_requirement,
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
