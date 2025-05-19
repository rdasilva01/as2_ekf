from setuptools import setup, find_packages

package_name = 'as2_ekf'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name,
         [package_name + '/ekf_wrapper.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CVAR-UPM',
    maintainer_email='cvar.upm3@gmail.com',
    description='AS2 EKF',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros_ekf = as2_ekf.ros_ekf:main',
        ],
    },
)
