from setuptools import setup, find_packages

setup(
    name="nr-scft-ml",
    version="0.1.0",
    description="The package tool for neutron reflectivity and SLD profile analysis",
    author="Yuqing Qiao",
    author_email="qiao.yuqi@northeastern.edu",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "docs"]),
    python_requires=">=3.11",
    install_requires=[
        "opencv-python>=4.11.0.86",
        "torch>=2.6.0",
        "numpy>=2.2.2",
        "pandas>=2.2.3",
        "matplotlib>=3.10.0",
        "seaborn>=0.13.2",
        "scikit-learn>=1.6.1",
        "scipy>=1.15.1",
        "torchvision>=0.21.0",
        "typer>=0.15.1",
        "pyyaml>=6.0.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "nr-scft-ml=pyreflect.cli.main:app",  # Replace with your actual CLI entry point
        ],
    },
    include_package_data=True,
)
