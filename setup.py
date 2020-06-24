from setuptools import find_packages, setup

with open('tonks/_version.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()

with open('LICENSE.txt') as l:
    license = l.read()

setup(
    name='tonks',
    version=__version__,
    description='General purpose multi-task classification library',
    long_description=readme+'\n\n\nLicense\n-------\n'+license,
    long_description_content_type='text/markdown',
    author='Nicole Carlson, Michael Sugimura',
    url='https://github.com/shoprunner/tonks',
    license='BSD-3-Clause',
    data_files=[('', ['LICENSE.txt'])],
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'creevey',
        'fastprogress',
        'joblib',
        'numpy',
        'Pillow<7.0.0',
        'transformers>=2.3.0',
        'sentencepiece!=0.1.92',
        'scikit-learn',
        'torch==1.2.0',
        'torchvision==0.2.1',
    ],
    extras_requires={
        'dev': [
            'flake8',
            'flake8-docstrings',
            'flake8-import-order',
            'm2r',
            'pydocstyle<4.0.0',
            'pytest',
            'pytest-cov',
            'sphinx-rtd-theme==0.4.3'
        ]
    },
)
