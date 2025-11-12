# blixtbird/setup.py

from setuptools import setup, find_packages
import os

setup(
    name='blixtbird',
    version='1.0.1',
    author='Gabriel Bengtsson, Zaid Haj-Ibrhaim, Piotr Krzyczkowski and Anna Lithell',
    author_email='anna.lithell@gustas.se',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'networkx',
        'mpi4py',
        'pyfiglet',
        'colorama',
        'pydantic', 
        'pyyaml'
    ],
    entry_points={
        'console_scripts': [
            'blixtbird=blixtbird.cli.runner:run_blixtbird_shell',
        ],
    },
    include_package_data=True,
    description='Blixtbird: A Simulation Framework for Modeling Attacks in DFL networks',
    long_description=open('README.md').read() if os.path.exists('README.md') else 'Blixtbird Shell for DFL Simulator',
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
