from setuptools import find_packages, setup

package_name = 'rov_mode_manager'

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
    maintainer='mustafa',
    maintainer_email='mustafa@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mode_manager_node = rov_mode_manager.mode_manager_node:main',
            'rov_mode_manager_gui = rov_mode_manager.rov_mode_manager_gui:main',
        ],
    },
)
