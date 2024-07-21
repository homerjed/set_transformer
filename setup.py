from setuptools import find_packages, setup

setup(
    name="set_transformer",
    packages=find_packages(where="src"),
    package_dir={"" : "src"},
    requires=[
        "numpy",
        "matplotlib",
        "jax[cpu]",
        "equinox",
        "optax",
        "tqdm",
        "torch",
        "torchvision"
    ]
)