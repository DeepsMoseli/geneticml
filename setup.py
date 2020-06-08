import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geneticml", # Replace with your own username
    version="1.2.4",
    author="Moseli Motsoehli",
    author_email="moselim@hawaii.edu",
    description="Use of Genetic algorithms for hyper-parameter optimization on common machine learning Algorithms on small arbitrary datasets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepsMoseli/geneticml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','pandas','tqdm', 'scikit-learn'],
)