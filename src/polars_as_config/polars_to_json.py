import ast
import inspect
import textwrap
from typing import Any, Callable


class PolarsToJson:
    def __init__(
        self,
        custom_functions: set[str] | None = None,
        allow_function_discovery: bool = False,
    ):
        self.dataframes: set[str] = set()
        self.custom_functions = custom_functions or set()
        self.allow_function_discovery = allow_function_discovery

    def parse_attribute(self, attribute: ast.Attribute) -> tuple[str, ast.Attribute]:
        """
        Parses an attribute expression and returns
        the expression and the attribute.
        """
        expr = attribute.attr
        while isinstance(attribute.value, ast.Attribute):
            expr = attribute.value.attr + "." + expr
            attribute = attribute.value
        return expr, attribute

    def parse_arg(self, arg: ast.expr | None) -> Any:
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
            result: dict[str, Any] = {}
            # 1. Parse the expression name, then args, then the "on" attribute.
            # The function must be an attribute of the polars object or
            # a polars expression.
            if not isinstance(arg.func, ast.Attribute):
                raise ValueError(f"Call must be on Attribute, got {arg.func}")
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
                if attribute.value.id not in ["pl", "polars"]:
                    raise ValueError(
                        f"Call must be on polars object, got {attribute.value.id}"
                    )
            else:
                result["on"] = self.parse_arg(attribute.value)
            result["expr"] = expr
            return result
        elif isinstance(arg, ast.List):
            return [self.parse_arg(a) for a in arg.elts]
        elif isinstance(arg, ast.Dict):
            return {
                self.parse_arg(k): self.parse_arg(v)
                for k, v in zip(arg.keys, arg.values, strict=True)
            }
        elif isinstance(arg, ast.Attribute):
            return self.parse_attribute(arg)[0]
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
        if not isinstance(target, ast.Name):
            raise ValueError("Assignment targets must be variable names")
        dataframe_name = target.id

        # The value must be a Call.
        if not isinstance(node.value, ast.Call):
            raise ValueError("Assignment values must be function calls")
        # The function must be called on an attribute;
        # either polars, or a dataframe name.
        # In our current implementation, the dataframe name must match the target.
        if not isinstance(node.value.func, ast.Attribute):
            raise ValueError("Assignment values must be function calls as attributes")
        # The value of the attribute is polars or a dataframe
        value = node.value.func.value
        if not isinstance(value, ast.Name):
            raise ValueError("Call must be on a dataframe or polars module")
        # The id is polars or a dataframe name
        name = value.id
        # We can now verify that the dataframe name is valid or if it is polars,
        # but we don't need to do anything with it.
        dataframe_in = None
        if name == "polars" or name == "pl":
            pass
        elif name in self.dataframes:
            dataframe_in = name
        else:
            raise ValueError(f"Invalid dataframe name: {name}")
        self.dataframes.add(dataframe_name)

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

        step = {
            "operation": function_name,
            "args": args,
            "kwargs": kwargs,
            "dataframe_out": dataframe_name,
        }
        if dataframe_in:
            step["dataframe_in"] = dataframe_in
        return step

    def polars_to_json(self, code: str) -> list[dict]:
        tree = ast.parse(code)
        # Get the assignments
        operations = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                # at this point we have a single step at our hands;
                # We can parse it as an individual operation, operating on a dataframe
                operations.append(self.parse_operation(node))
        return operations

    def polars_function_to_json(self, function: Callable) -> list[dict]:
        # Get the source code of the function.
        code = inspect.getsource(function)
        # Clip off the function name, de-indent the rest, and return the code.
        code = textwrap.dedent("\n".join(code.split("\n")[1:]))
        operations = self.polars_to_json(code)
        return operations
