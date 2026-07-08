from setuptools import find_packages, setup

package_name = 'pipe_detection'

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
            'pipe_detection_node = pipe_detection.pipe_detection_node:main',
            'end_of_pipe_node = pipe_detection.end_of_pipe_node:main'
        ],
    },
)
