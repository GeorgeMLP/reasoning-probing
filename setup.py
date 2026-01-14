from setuptools import setup, find_packages

setup(
    name="reasoning-features",
    packages=find_packages(),
    version="0.0.1",
    author="George Ma",
    install_requires=[
        "zstandard",
        "numpy",
        "scikit-learn",
        "torch",
        "torchvision",
        "einops",
        "jaxtyping",
        "datasets",
        "transformers",
        "tqdm",
        "sae_lens",
        "transformer_lens",
        "matplotlib",
        "json-repair",
    ],
    url="",
    description="",
    python_requires='>=3.9',
)
