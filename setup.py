from setuptools import setup

setup(
    name='phenopype',
    url='https://github.com/mluerig/phenopype',
    author='Moritz Luerig',
    author_email='moritz.luerig@eawag.ch',
    packages=['phenopype'],
    install_requires=['setuptools-git-version', "pandas", "opencv-contrib-python>=3.4.4", "exifread", "Pillow", "pytesseract", "trackpy"], 
    version='{tag}.dev{commitcount}+{gitsha}',
    license='LGPL',
    description='a phenotyping pipeline for python',
    long_description=open('README.md').read(),
)
