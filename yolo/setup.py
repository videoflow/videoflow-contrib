from setuptools import setup

# Runtime deps (tensorflow, tf-keras, Pillow) live in requirements.txt /
# requirements-gpu.txt, which the Dockerfiles install before `pip install --no-deps .`.
# Only the videoflow dependency is declared here for a direct source install.
name = 'videoflow_contrib_yolo'
install_requires = ['videoflow']

setup(
    name=name,
    version='0.1',
    description='YOLOv3 object detection component for the Videoflow library',
    author='Jadiel de Armas',
    author_email='jadielam@gmail.com',
    url='https://github.com/videoflow/videoflow-contrib',
    license='MIT',
    # Explicit list (not find_packages): videoflow_contrib is a PEP 420 namespace
    # package with no top-level __init__.py, which find_packages would skip.
    packages=['videoflow_contrib.yolo', 'videoflow_contrib.yolo.yolo3'],
    zip_safe=False,
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
