from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='phenopype',
    url='https://github.com/mluerig/phenopype',
    author='Moritz Lürig',
    author_email='moritz.luerig@eawag.ch',
    # Needed to actually package something
    packages=['phenopype'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.3.0',
    # The license can be anything you like
    license='LGPL',
    description='phenotyping pipeline for python',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)