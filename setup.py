from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

setup(
    name='yi',
    version='0.0.3',
    description='',
    author='yu-9824',
    author_email='{0}@{1}'.format('yu.9824.job', 'gmail.com'),
    install_requires=install_requirements,
    url='https://github.com/yu-9824/yi',
    license=license,
    packages=find_packages(exclude=['example'])
)
