from setuptools import find_packages, setup

package_name = "ouroboros_ros"

setup(
    name=package_name,
    version="2.0.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="aaron",
    maintainer_email="aaronray@mit.edu",
    description="ROS interface for Ouroboros VLC Server",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vlc_server_node = ouroboros_ros.nodes.vlc_server_node:main",
            "vlc_multirobot_server_node = ouroboros_ros.nodes.vlc_multirobot_server_node:main",
        ],
    },
)
