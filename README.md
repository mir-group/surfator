# surfator

"Atomic democracy" for site analysis of surfaces and bulks with known lattice
structure(s).

Built on [`sitator`](https://github.com/Linux-cpp-lisp/sitator), which provides a general framework for site analysis.

## Installation

`surfator` depends on [`sitator`](https://github.com/Linux-cpp-lisp/sitator).
(`sitator`'s optional dependencies are **not** required.)

Once `sitator` is installed, `surfator` is installed like any other Python package:

```bash
$ git clone https://github.com/mir-group/surfator.git
$ cd surfator/
$ pip install .
```

## Algorithm and Usage

Please see our preprint for a full description of the atomic democracy
algorithm (Methods -> Clamping):

> **Evolution of Metastable Structures in Bimetallic Catalysts from Microscopy and Machine-Learning Molecular Dynamics**
> Jin Soo Lim, Jonathan Vandermause, Matthijs A. van Spronsen, Albert Musaelian, Christopher R. O’Connor, Tobias Egle, Yu Xie, Lixin Sun, Nicola Molinari, Jacob Florian, Kaining Duanmu, Robert J. Madix, Philippe Sautet, Cynthia M. Friend, Boris Kozinsky
> https://doi.org/10.26434/chemrxiv.11811660.v1

Documentation of parameters can be found in docstrings in the source.

## License

This software is made available under the MIT License. See [LICENSE](./LICENSE) for details.
