from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

setup(
    name='wrapper_method',
    version='0.0.1',
    description='for wrapper method. (by using boruta_py)',
    author='yu-9824',
    author_email='yu.9824@gmail.com',
    install_requires=install_requirements,
    url='https://github.com/yu-9824/wrapper_method',
    license=license,
    packages=find_packages(exclude=['example'])
)
