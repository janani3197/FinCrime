from setuptools import setup, find_packages
from typing import List



def get_requirements(file_path:str)->List[str]:
    """This function returns a list of requirements from the requirements.txt file"""
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    if '-e .' in requirements:
        requirements.remove('-e .')
    return requirements

setup(
    name="fincrime_project",
    version="0.1",
    author="Janani Selvam",
    description="A package for financial crime detection",
    author_email="jananiselvam15@gmail.com",
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt'))