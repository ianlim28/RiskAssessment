from setuptools import find_packages, setup

setup(name='RiskAssessment',
      version='0.1',
      description='Functions for assessing stock risks',
      url='https://github.com/ianlim28/RiskAssessment',
      author='ianlim',
      author_email='ianlim28@hotmail.com',
      license='MIT',
      packages=find_packages(),
      package_dir={"": "src"},
      zip_safe=False)