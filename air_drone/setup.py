from setuptools import find_packages, setup

package_name = 'air_drone'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ei_admin',
    maintainer_email='ei_admin@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drone_env = air_drone.scripts.drone_env:main',
            'target_points = air_drone.scripts.target_points:main',
        ],
    },
)
