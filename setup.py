from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = ['torch>=1.5', 'scikit-learn>=0.20', 'pandas>=1.0']

setup(
	name='emnist-train',
	version='1.0.0',
    install_requires=REQUIRED_PACKAGES,
	packages=find_packages(),
	include_package_data=True,
	description='ML Model training for Emnist data with PyTorch frameowrk'
	)