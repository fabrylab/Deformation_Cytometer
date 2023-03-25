from setuptools import setup
import sys
python_version = (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
if python_version[0] < 3 or python_version[1] > 8:
      raise Exception("Deformation Cytometer requires python version <= 3.8.6, "
            "your version is %d.%d.%d" % python_version)


setup(name='deformationcytometer',
      version="0.1",
      packages=['deformationcytometer'],
      description='Cell deformation under shear flow analysis package in python.',
      author='Richard Gerum',
      author_email='richard.gerum@fau.de',
      license='MIT',
      install_requires=[
            'numpy',
            'scipy',
            'tqdm',
            'tensorflow == 2.11.1',
            'tensorflow-addons == 0.12.1',
            'scikit-image>=0.17.2',
            'imageio',
            'tifffile',
            "opencv-python",
            'qtawesome>=1.0.0',
            "pandas",
            "h5py",
      ],
)

