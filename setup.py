from setuptools import setup
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
    url="https://github.com/mluerig/phenopype",
    author="Moritz Luerig",
    author_email="moritz.luerig@gmail.com",
    packages=["phenopype"],
    package_dir={"phenopype": "phenopype"},
    package_data={"phenopype": ["core/*.py", "templates/*.py", "templates/*.yaml"]},
    install_requires=[
        "numpy==1.18.5",
        "opencv-contrib-python==3.4.9.33",
        "pandas==1.1.2",
        "pillow==8.1.0",
        "pyradiomics==3.0.1",
        "ruamel.yaml==0.16.12",
        "tqdm",
        "watchdog==2.0.0",
    ],
    version=verstr,
    license="LGPL",
    description="a phenotyping pipeline for python",
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
    setup_requires=["pytest-runner"],
)