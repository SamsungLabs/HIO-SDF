#!/usr/bin/env python
import os
from setuptools import find_packages, setup
from glob import glob

package_name = 'hio_sdf'

setup(
    name=package_name,
    version='0.0.0',
    packages=['hio_sdf', 'pointcloud_sdf', 'dataset_adapter'],
    package_dir = {
        'hio_sdf': 'src/hio_sdf',
        'pointcloud_sdf': 'src/pointcloud_sdf',
        'dataset_adapter': 'src/dataset_adapter',
    },
    maintainer='vasileios.v',
    maintainer_email='vasileios.v@samsung.com',
)