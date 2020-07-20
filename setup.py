import sys

if sys.version_info.major == 3:
    from setuptools import setup

    setup(
        name='anomaly_detector',
        version='0.0.1',
        author='Ludwig Waffenschmidt',
        packages=['feature_extractor', 'anomaly_model', 'common', 'scripts'],
        package_dir={'': 'anomaly_detector'}
    )
else:
    from distutils.core import setup
    from catkin_pkg.python_setup import generate_distutils_setup

    setup_args = generate_distutils_setup(
        name='anomaly_detector',
        version='0.0.1',
        packages=['feature_extractor', 'anomaly_model', 'common', 'scripts'],
        package_dir={'': 'anomaly_detector'}
    )

    setup(**setup_args)