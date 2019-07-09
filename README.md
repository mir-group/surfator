# surfator

"Atomic democracy" for site analysis of surfaces and bulks with known lattice
structure(s).

Built on [`sitator`](https://github.com/Linux-cpp-lisp/sitator), which provides a general framework for site analysis.

## Installation

`surfator` depends on [`sitator`](https://github.com/Linux-cpp-lisp/sitator).

Once `sitator` is installed, `surfator` is installed like any other Python package:

```
# git clone | unzip | ...
cd surfator/
pip install .
```

## Usage

TODO.

## The Algorithm

Three core ideas:

 - A reference site is simply a point in space where, *depending on the state of the system*, an atom can "live."
 - An "agreement group" is a group of atoms in the system, *at some frame*, that must agree on a single (or a group of compatible) "structure group(s)." Every atom is always a member of exactly one agreement group, but which agreement group it is a part of, and even the total number of agreement groups, can change from frame to frame.
 - "Structure groups" are groups of reference sites that "make sense" together; in an FCC surface where a layer can sit on either the layer below's FCC or HCP sites, those two different sets of sites would be two structure groups. Structure groups can be compatible or incompatible with one another; in the same example, where there are three possible horizontal offsets for each layer, a layer cannot have the same offset as the layer below. Therefore, one would mark those two structure groups as incompatible. Two alternative layouts for a single layer are also obviously incompatible.

The reference sites, the structure groups to which they belong, and the matrix of pairwise compatibilities between those structure groups are provided as input.

The agreement groups are determined each frame by a user-supplied function that, given the current positions of the atoms in the system, divides them into agreement groups. Any function could be used; `surfator` provides a number of useful agreement group functions in `surfator.grouping`. The agreement groups are ordered: earlier ones take precedence over later in filling sites and when resolving structure group compatibilities. In a typical surface, one would typically have each layer as an agreement group and order them from the bottom-most bulk layer to the most volatile surface layer.

At each frame, the atoms within each agreement group "choose" among all available sites compatible with the structure groups previous agreement groups have chosen. (Each atom chooses the site nearest to it.) Based on the membership of those sites in various structure groups, the atoms then "vote" on which structure group to choose. Having chosen a structure group, the atoms are reassigned to the nearest site *within or compatible with that structure group*. 
