from setuptools import setup, find_packages

setup(
    name='tinybot',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    package_data={
        'tinybot.beeps': ['*.wav'],
    },
    install_requires=[
        'dynamixel-sdk',
        'sounddevice',
        'soundfile',
        'numpy',
        'pyserial',
    ],
)
