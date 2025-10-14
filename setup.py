from setuptools import setup, find_packages

setup(
    name="pytpq",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'numexpr',
        'inspect',
        'multiprocessing',
        'functools',
        'joblib',
        'h5py',
        ],
)