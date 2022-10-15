import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cBCI",
    version="0.1.0",
    author="Davide Valeriani",
    author_email="davide.valeriani@gmail.com",
    description="The Collaborative Brain-Computer Interface Toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davidevaleriani/cBCI",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy", 
        "pandas", 
        "matplotlib", 
        "scipy", 
        "seaborn", 
        "pingouin"
    ]
)
