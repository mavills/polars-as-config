# Transformation format

Transformations are defined once, and can run on any file that follows the structure of the defined transformation's input.
In this section, we assume the transformation is in the final format, the one that is ready to transform files.

It is out of scope for this repository to link input file prefixes (or any other type of linking) to their respective transformation;
we just expect an input file name (or multiple input file names), and the transformation definition to run it with

The transformation itself is defined as a set of polars operations (in JSON), that operate on a specific frame (LazyFrame in code).
