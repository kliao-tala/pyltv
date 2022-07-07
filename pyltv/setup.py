import setuptools

setuptools.setup(
    name="pyltv",
    version="0.0.1",
    author="Kenneth Liao",
    author_email="kenny.liao@tala.co",
    description="pLTV Forecasting.",
    long_description="pLTV Forecasting library with plotting and backtesting functionality.",
    long_description_content_type="text/markdown",
    url="https://github.com/kliao-tala/pyltv",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)