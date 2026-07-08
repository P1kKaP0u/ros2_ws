from setuptools import find_packages, setup

package_name = 'manual_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'geometry_msgs'],
    zip_safe=True,
    maintainer='mustafa',
    maintainer_email='mustafa@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'manual_control_node = manual_control.manual_control_node:main',
            'cmd_vel_listener = manual_control.cmd_vel_listener:main'
        ],
    },
)
