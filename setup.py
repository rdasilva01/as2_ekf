from setuptools import setup, find_packages

package_name = 'ekf'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name,
         ['ekf_definition' + '/ekf_wrapper.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CVAR-UPM',
    maintainer_email='cvar.upm3@gmail.com',
    description='EKF',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros_ekf = ekf_definition.ros_ekf:main',
        ],
    },
)
