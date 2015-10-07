from setuptools import setup
from setuptools import find_packages

setup(
      name='Sequence-to-Sequence',
      version='0.0.0.1',
      description='Experiment with Sequence to Sequence Learning with Neural Networks',
      author='Giancarlo Salton',
      author_email='giancarlo.salton@mydit.ie',
      url='https://github.com/giancds/seq_to_seq',
      license='MIT',
      install_requires=['theano', 'numpy'],
      packages=find_packages())
