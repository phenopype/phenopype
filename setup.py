from setuptools import setup

setup(
    name='phenopype',
    url='https://github.com/mluerig/phenopype',
    author='Moritz Lürig',
    author_email='moritz.luerig@eawag.ch',
    packages=['phenopype'],
    install_requires=['numpy'],
    version='0.4.5',
    license='LGPL',
    description='a phenotyping pipeline for python',
    long_description=open('README.md').read(),
)