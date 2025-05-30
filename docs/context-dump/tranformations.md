# Transformations

This section describes some steps, concepts and definitions about the method of transformation in project.

## Steps involved in defining transformations

1. Define building blocks/template blocks: The different allowed operations you can choose from to build your transformation with. These can be defined using JSON syntax similar to how you would code in polars, and add connection points to it so it can be used as a building block.
2. Create your transformation: Combine building blocks (templates from step 1) of any type into a desired transformation. The idea is that this would be done through a flexible interface, where a user is enabled to connect any outputs of building blocks to any inputs of building blocks, steadily creating the desired output.
3. The result of the user's creation is a directed acyclic graph (if it is cyclic, it can never finish). The next step is to evaluate this graph into a single array of instructions that can be executed. Think of this as the compilation of a programming language, where the language consists of the template blocks we defined, the program is the graph, and the compilation is the instruction set that follows from the graph, but then in JSON specifying polars.
4. The result of evaluating the graph is the transformation configuration. It can now be applied to input files, which is the same as running the polars instructions on the file. The only things needed are: (1) the input files and how they map to the inputs required for the transformation; (2) the output file names and how they map onto the outputs of the transformation; and (3) the transformation configuration created before.

So to summarize and naming some concepts:

1. Define `template blocks`
2. Assimilate a `transformation graph`
3. Evaluate the `transformation graph` into a `transformation configuration`
4. Map files to the `transforamtion configuration inputs` and `transformation configuration outputs`
5. Apply the `transformation configuration` to `transform` inputs into outputs.
