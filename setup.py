from setuptools import setup, find_packages

setup(
    name="sv3",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        # Add your dependencies here
        # e.g., "torch>=1.9.0",
        # "jax>=0.3.0",
    ],
    author="Your Name",
    description="SV3 package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
