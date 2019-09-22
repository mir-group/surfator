from setuptools import setup, find_packages

setup(name = 'surfator',
      version = '0.0.1',
      description = 'Site analysis for surfaces with `sitator`.',
    #  download_url = "https://github.com/Linux-cpp-lisp/sitator",
      author = 'Alby Musaelian',
      license = "MIT",
      python_requires = '>=3.2',
      packages = find_packages(),
      install_requires = [
        "numpy",
        "scipy",
        "ase",
        "sklearn",
        "sitator"
      ],
      zip_safe = True)
