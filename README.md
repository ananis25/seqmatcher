# seqmatcher

[![PyPI](https://img.shields.io/pypi/v/seqmatcher.svg)](https://pypi.org/project/seqmatcher/)
[![Changelog](https://img.shields.io/github/v/release/ananis25/seqmatcher?include_prereleases&label=changelog)](https://github.com/ananis25/seqmatcher/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ananis25/seqmatcher/blob/main/LICENSE)

`seqmatcher` provides a DSL to match and edit sequences of events. Similar to how regular expressions help match patterns in text (which is a stream of characters), a collection of sequences (stream of events) can be analyzed similarly. This is a total ripoff of the work done [here](https://observablehq.com/@mikpanko/sequence-pattern-matching?collection=@mikpanko/sequences) by Mikhail Panko. 

The original notebooks introduce the semantics for the regex-like syntax and implement it as javascript code over lists of objects. Without a JIT like V8, it would be pretty slow to execute the same code in python. So instead we,
* persist the dataset in the parquet format to read it quick.
* read it using the `awkward` [library](https://github.com/scikit-hep/awkward) which supports jagged arrays and optional datatypes. 
* compile the pattern matching routines at runtime using the `numba` [library](https://numba.pydata.org/) and run it against the awkward array data. 

Performance wins:
* Numba implements bindings to LLVM, so the compiled code runs pretty quick. 
* Awkward arrays are immutable and store all attributes, including nested ones, in contiguous buffers. So, matching and extracting subsequences copies very little data, and just record slices of the original arrays to use as output. 

Things that are tricky:
* Numba requires static variable types for compilation, so that constrains us to a consistent schema across all sequences and events. 
* A columnar data layout also makes modifying the matched sequences tricky (TODO: still gotta implement it in jitted code). 


## Installation

Install this library using `pip`:

    $ pip install seqmatcher

## Usage

Usage instructions go here.

## Development

To contribute to this library, checkout the code in a new virtual enviroment.

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest
