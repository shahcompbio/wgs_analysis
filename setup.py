from setuptools import find_packages, setup

setup(
    name='wgs_analysis',
    description='Whole genome sequencing analysis library',
    packages=find_packages(),
    package_data={'':['data/*']},
)
