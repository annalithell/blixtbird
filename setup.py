# fenics/setup.py

from setuptools import setup, find_packages
import os

setup(
    name='fenics',
    version='1.3.2',
    author=,
    author_email=,
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
        'yamllint',
        'pydantic',
        'tqdm'  
    ],
    entry_points={
        'console_scripts': [
            'fenics=fenics.cli.runner:run_fenics_shell',
        ],
    },
    include_package_data=True,
    description='Decentralized Federated Learning Simulator for Security Evaluation - Fenics',
    long_description=open('README.md').read() if os.path.exists('README.md') else 'Fenics Shell for DFL Simulator',
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
