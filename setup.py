from setuptools import setup


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
            'tensorflow == 2.4.0',
            'tensorflow-addons == 0.12.1',
            'scikit-image>=0.17.2',
            'imageio',
            'tifffile',
            "fill_voids == 2.0.1",
            "opencv-python"

      ],
)
