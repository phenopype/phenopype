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

test_deps = ["pytest", 
             "coverage", 
             "coveralls", 
             "pyyaml",
             "mock",
             "pytest-xvfb",
             "pytest-cov"]

extras = {'test': test_deps}

setup(
	name='phenopype',
	url='https://github.com/mluerig/phenopype',
	author='Moritz Luerig',
	author_email='moritz.luerig@gmail.com',
	packages=['phenopype'],
    package_dir={'phenopype': 'phenopype'},
    package_data={'phenopype': ['core/*.py', 'settings/*.py']},
	install_requires=[
		"numpy==1.18.5",
		"opencv-contrib-python==3.4.9.33",
        "pandas",
		"Pillow",
        "pyradiomics",
        "tqdm",
        "ruamel.yaml",
        "watchdog"
],
	version=verstr,
	license='LGPL',
	description='a phenotyping pipeline for python',
	long_description=open('README.md').read(), 
	long_description_content_type='text/markdown',
    tests_require=test_deps,
    extras_require=extras,
	setup_requires=["pytest-runner"]
)
