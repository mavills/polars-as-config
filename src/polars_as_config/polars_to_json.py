import ast


class PolarsToJson:
    def __init__(
        self,
        custom_functions: set = None,
        allow_function_discovery: bool = False,
    ):
        self.dataframes = set()
        self.custom_functions = custom_functions or set()
        self.allow_function_discovery = allow_function_discovery

    def parse_attribute(self, attribute: ast.Attribute) -> str:
        """
        Parses an attribute expression and returns
        the expression and the attribute.
        """
        expr = attribute.attr
        while isinstance(attribute.value, ast.Attribute):
            expr = attribute.value.attr + "." + expr
            attribute = attribute.value
        return expr, attribute

    def parse_arg(self, arg: ast.expr) -> dict:
        if isinstance(arg, ast.Constant):
            return arg.value
        elif isinstance(arg, ast.Name):
            # If the argument is a name, it must be a dataframe name
            # or a custom function name.
            if arg.id in self.dataframes:
                return arg.id
            elif arg.id in self.custom_functions:
                return {"custom_function": arg.id}
            elif self.allow_function_discovery:
                self.custom_functions.add(arg.id)
                return {"custom_function": arg.id}
            else:
                raise ValueError(f"Invalid dataframe or custom function name: {arg.id}")
            # While we could do more extensive checks,
            # we will omit those for now.
            # It would involve verifying that the name references
            # is indeed an existing dataframe.
            # Could be done by passing all know dataframes and just
            # doing a lookup.
        elif isinstance(arg, ast.Call):
            # A call can either be on the polars object again, or on another expression.
            # We will parse the call recursively.
            result = {}
            # 1. Parse the expression name, then args, then the "on" attribute.
            # The function must be an attribute of the polars object or a polars expression.
            assert isinstance(arg.func, ast.Attribute)
            expr, attribute = self.parse_attribute(arg.func)
            # 2. Parse the args and kwargs.
            args = []
            for a in arg.args:
                parsed_arg = self.parse_arg(a)
                args.append(parsed_arg)
            result["args"] = args
            kwargs = {}
            for kwarg in arg.keywords:
                parsed_arg = self.parse_arg(kwarg.value)
                kwargs[kwarg.arg] = parsed_arg
            result["kwargs"] = kwargs
            # 3. Parse the "on" attribute.
            if isinstance(attribute.value, ast.Name):
                # If the value is a name, it can only be called on the polars object.
                assert attribute.value.id in ["pl", "polars"]
            else:
                result["on"] = self.parse_arg(attribute.value)
            result["expr"] = expr
            return result

        else:
            raise NotImplementedError(f"Unsupported argument type: {type(arg)}")

    def parse_operation(self, node: ast.Assign) -> dict:
        # Returns something of the form:
        # {
        #     "operation": "read_csv",
        #     "args": ["data.csv"],
        #     "kwargs": {},
        #     "dataframe": "df",
        # }

        # each assignment must have only one target
        if len(node.targets) != 1:
            raise ValueError("Each assignment must have only one target")
        target = node.targets[0]
        # the target must be a variable name
        assert isinstance(target, ast.Name)
        dataframe_name = target.id

        # The value must be a Call.
        assert isinstance(node.value, ast.Call)
        # The function must be called on an attribute;
        # either polars, or a dataframe name.
        # In our current implementation, the dataframe name must match the target.
        assert isinstance(node.value.func, ast.Attribute)
        # The value of the attribute is polars or a dataframe
        value = node.value.func.value
        assert isinstance(value, ast.Name)
        # The id is polars or a dataframe name
        name = value.id
        # We can now verify that the dataframe name is valid or if it is polars,
        # but we don't need to do anything with it.
        if name == "polars" or name == "pl":
            pass
        elif name == dataframe_name:
            # Add the dataframe to the list of allowed dataframes.
            self.dataframes.add(dataframe_name)
        else:
            raise ValueError(f"Invalid dataframe name: {name}")

        # The thing we care about now is the function call and its arguments.
        function_name = node.value.func.attr
        # There is args and kwargs.
        # Both need to be parsed recursively.
        args = []
        for arg in node.value.args:
            parsed_arg = self.parse_arg(arg)
            args.append(parsed_arg)
        kwargs = {}
        for kwarg in node.value.keywords:
            parsed_arg = self.parse_arg(kwarg.value)
            kwargs[kwarg.arg] = parsed_arg

        return {
            "operation": function_name,
            "args": args,
            "kwargs": kwargs,
            "dataframe": dataframe_name,
        }

    def polars_to_json(self, code: str) -> dict:
        tree = ast.parse(code)
        # Get the assignments
        operations = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                # at this point we have a single step at our hands;
                # We can parse it as an individual operation, operating on a dataframe
                operations.append(self.parse_operation(node))
        return operations
