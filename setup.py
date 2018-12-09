from setuptools import setup

import re
VERSIONFILE="myniftyapp/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))
    
    
setup(
    name='phenopype',
    url='https://github.com/mluerig/phenopype',
    author='Moritz Luerig',
    author_email='moritz.luerig@eawag.ch',
    packages=['phenopype'],
    install_requires=['setuptools-git-version', "pandas", "opencv-contrib-python>=3.4.4", "exifread", "Pillow", "pytesseract", "trackpy"], 
    version=verstr,
    license='LGPL',
    description='a phenotyping pipeline for python',
    long_description=open('README.md').read(),
)
