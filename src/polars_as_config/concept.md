# Concept

## Defining operations

We can create an operation in polars by defining it as JSON such that it exactly matches the function signature of an existing polars function.

## Defining operation choices

Since the end-user will not have the option to choose anything, but only from a limited set of options we provide, we must define a way for creating such operations.
We do this by adding a definition of inputs and outputs onto each operation.

This allows any front-end application to correctly display what possible inputs and outputs are composed of.

Inputs and outputs define types. These types can be used to validate whether or not two connection points can be connected.
Possible types are:

- str
- datetime
- enum
  - further defined by specifying enum options
  - in front-ends, possibly defined by using a dropdown

## Building a chain

Definitions have:

1. a block type
1. an operation
1. inputs
1. outputs
1. can have some custom functionality for a limited amount of blocks
