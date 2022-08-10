from setuptools import setup, find_packages

setup(
    name='pybehavior',
    version='0.0.4',
    description='PyBehavior',
    author='jup014',
    author_email='jup014@ucsd.edu',
    url='https://cwphs.ucsd.edu',
    license='MIT',
    py_modules=['pybehavior'],
    python_requires='>=3',
    install_requires=['numpy', 'pandas'],
    packages=['pybehavior']
)