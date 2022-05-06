# seqmatcher

[![PyPI](https://img.shields.io/pypi/v/seqmatcher.svg)](https://pypi.org/project/seqmatcher/)
[![Changelog](https://img.shields.io/github/v/release/ananis25/seqmatcher?include_prereleases&label=changelog)](https://github.com/ananis25/seqmatcher/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ananis25/seqmatcher/blob/main/LICENSE)

`seqmatcher` is a python library that provides a DSL to match and edit sequences of events. Similar to how regular expressions help match patterns in text (which is a stream of characters), a collection of sequences (stream of events) can be analyzed similarly. 

This is a total ripoff of the work done [here](https://observablehq.com/@mikpanko/sequence-pattern-matching?collection=@mikpanko/sequences).

## Installation

Install this library using `pip`:

    $ pip install seqmatcher

## Usage

Usage instructions go here.

## Development

To contribute to this library, checkout the code and install `Flit`. 

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest
