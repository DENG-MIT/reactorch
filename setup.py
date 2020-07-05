from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='reactorch',
      version='0.1.1',
      description='ReacTorch: A Differentiable Reacting Flow Simulation Package in PyTorch',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/DENG-MIT/reactorch',
      author='Weiqi Ji, Sili Deng',
      author_email='weiqiji@mit.edu',
      license='MIT',
      packages=['reactorch'],
      zip_safe=False)
