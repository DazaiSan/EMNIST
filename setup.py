from setuptools import setup
from setuptools import find_packages

setup(
	name='emnist-train',
	version='1.0.0',
	packages=find_packages(),
	include_package_data=True,
	description='ML Model training for Emnist data with PyTorch frameowrk'
	)