from setuptools import setup, find_packages

setup(
    name='devtools_scicomp_project',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
