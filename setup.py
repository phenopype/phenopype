from setuptools import setup, find_packages
import re

## read and format version from file
VERSIONFILE = "phenopype/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

## readme encoding
with open("README.md", encoding="utf-8") as readme:
    long_description = readme.read()

## setup
setup(
    name="phenopype",
    url="https://www.phenopype.org",
    author="Moritz Luerig",
    author_email="moritz.luerig@gmail.com",
    packages=find_packages(),
    package_data = {
        'assets': ['*.html'],
    },
    include_package_data=True,
    python_requires='==3.9.*',
    install_requires=[
        "colour",
        "numpy",
        "opencv-contrib-python==4.5.2.54",
        "pandas",
        "pillow",
        "pyradiomics==3.0.1",
        "ruamel.yaml==0.16.12",
        "tqdm",
        "watchdog",
    ],
    version=verstr,
    license="LGPL",
    description="A phenotyping pipeline for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    extras_require={
        "test_pks": [
            "pytest",
            "coverage",
            "coveralls",
            "mock",
            "pytest-xvfb",
            "pytest-cov",
            "pyyaml"
        ]
    },
    setup_requires=[], #"pytest-runner"
)
