from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'py_code'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tkuai',
    maintainer_email='ugo.roux31@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lidar_node = py_code.lidar_node:main',
            'yolo_node = py_code.yolo_node:main',
            'main_controller = py_code.main_controller:main',
            'camera_calibration_node = py_code.camera_calibration_node:main',
            'motor_driver = py_code.motor_driver:main',
        ],
    },
)
