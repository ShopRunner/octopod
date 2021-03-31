from setuptools import find_packages, setup

with open('octopod/_version.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()

with open('LICENSE.txt') as l:
    license = l.read()

setup(
    name='octopod',
    version=__version__,
    description='General purpose multi-task classification library',
    long_description=readme+'\n\n\nLicense\n-------\n'+license,
    long_description_content_type='text/markdown',
    author='Nicole Carlson, Michael Sugimura',
    url='https://github.com/shoprunner/octopod',
    license='BSD-3-Clause',
    data_files=[('', ['LICENSE.txt'])],
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'fastprogress',
        'joblib',
        'm2r2',
        'matplotlib',
        'numpy',
        'Pillow<7.0.0',
        'sentencepiece!=0.1.92',
        'scikit-learn',
        'sphinx-rtd-theme',
        'torch',
        'torchvision==0.2.1',
        'transformers>=2.3.0,<3.0.0',  # we have had issues training models with V3
        'wildebeest',
    ],
)
