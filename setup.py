from setuptools import setup, find_packages

setup(
    name='graphnn',
    version='0.1',
    packages=find_packages(include=["graphnn", "graphnn.*"]),
    install_requires=[
        "numpy",
        "torch==2.0",
        "torch_scatter",
        "torch_sparse",
        "torch_geometric",
        "tqdm",
        "datasets"
    ],
)