from setuptools import setup

import re
VERSIONFILE="phenopype/_version.py"
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
	install_requires=[
		"exifread==2.1.2",
		"numpy",
		"pandas",
		"pytesseract==0.2.6",
		"pytest==4.2.0",
		"opencv-contrib-python==3.4.5.20",
		"trackpy==0.4.1"
],
	version=verstr,
	license='LGPL',
	description='a phenotyping pipeline for python',
	long_description=open('README.md').read(), 
	tests_require=["pytest"],
	setup_requires=["pytest-runner"]
)
