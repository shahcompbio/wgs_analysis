from setuptools import find_packages, setup
import versioneer


setup(
    name='wgs_analysis',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Whole genome sequencing analysis library',
    packages=find_packages(),
    package_data={'':['data/*']},
)
