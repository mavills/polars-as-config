# Solution Space

This section describes what we want to build, what features we dream of,
why we decided on some initial design decisions.

## Ideation

The solution to this problem has to be:
- as fast as possible
- well-documented
- extensible (you can do a lot with it)
- flexible (we don't want to get stuck and have to move to another tool)

The solution that is decided upon:
> A layer on top of polars, that allows us to configure and describe polars transformation components, combine pre-defined transformation components into a final transformation pipeline, and run final transformation pipelines with file inputs on the polars library.

This comes with the following list benefits and features:
- It is fast (written in rust, super optimized, tabular operations,...)
- It is documented from the start:
  - for everything about transformations themselves, we can refer to the polars documentation
  - for everything about how to configure tranformation components and combine pipelines, we document ourselves, but this is only a small layer
- polars has a vast set of operations
- polars is extensible itself
- we can extend upon this tool with minimal effort, since polars carries a lot of the heavy lifting.
- polars supports writing SQL-like queries: this allows us to short-cut transformations and write SQL in this tool too. This is relevant when a user of your tools you built upon this project requires more complex operations or ways of configuring.
- staying close to the polars format allows tech people already versed in polars to go fast
