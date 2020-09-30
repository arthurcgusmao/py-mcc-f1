import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py-mcc-f1",
    version="0.1.0",
    author="Arthur Colombini Gusmão",
    description="MCC-F1 Curve",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arthurcgusmao/py-mcc-f1",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.14.0",
        "scikit-learn>=0.22"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
