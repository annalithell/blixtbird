# setup.py

from setuptools import setup, find_packages
import os

setup(
    name='phoenix',
    version='1.3.2',
    author='Shubham Saha, Sifat Nawrin Nova',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'numpy',
        'networkx',
        'psutil',
        'pyfiglet',
        'colorama',
        'yamllint'

    ],
    entry_points={
        'console_scripts': [
            'phoenix=dfl_simulator.main:run_phoenix_shell',
        ],
    },
    include_package_data=True,
    description='Decentralized Federated Learning Simulator - Phoenix',
    long_description=open('README.md').read() if os.path.exists('README.md') else 'Phoenix Shell for DFL Simulator',
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
