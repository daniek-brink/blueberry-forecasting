
# System imports
from setuptools import setup, find_packages

with open('requirements_pip.txt') as f:
    requirements = f.read().splitlines()

with open('README.md', 'r') as readme_file:
    readme = readme_file.read()

setup(name='blueberry_forecasting_api',
      version='0.0.0.0',
      description='A blueberry yield forecasting API',
      long_description=readme,
      long_description_content_type='text/markdown',
      license='PROPRIETY',
      packages=find_packages(),
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False,
      requires=[])
