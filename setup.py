import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="Mixture_Models",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Siva Rajesh Kasa",
    author_email="sivarajesh.kasa@gmail.com",
    description="A Python library for fitting mixture models using gradient based inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)
