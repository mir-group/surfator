from setuptools import setup, find_packages

setup(name = 'surfator',
      version = '1.0.0',
      description = '"Atomic democracy" for site analysis of surfaces and bulks with known lattice structure(s).',
      download_url = "https://github.com/mir-group/surfator",
      author = 'Alby Musaelian',
      license = "MIT",
      python_requires = '>=3.2',
      packages = find_packages(),
      install_requires = [
        "numpy",
        "scipy",
        "ase",
        "sklearn",
        "sitator>=2" # sitator 2.0.0 is the py3 version
      ],
      zip_safe = True)
