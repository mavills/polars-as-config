import inspect
from inspect import Parameter
from typing import Callable, ForwardRef, Optional, Union, get_args, get_origin

import polars as pl
from polars import DataFrame, LazyFrame
from polars._typing import PolarsDataType  # noqa: F401, needed for type checking
from polars.functions.col import Col


class Config:
    custom_functions: dict[str, Callable] = {}

    def __init__(self):
        self.current_dataframes: dict[str | None, pl.DataFrame] = {}

    def add_custom_functions(self, functions: dict) -> "Config":
        self.custom_functions.update(functions)
        return self

    def _get_parameter_types(self, method: Callable) -> dict[str, Parameter]:
        try:
            return inspect.signature(method).parameters
        except TypeError:
            pass
        if isinstance(method, Col):
            return {
                "name": Parameter(
                    "name", Parameter.POSITIONAL_OR_KEYWORD, annotation="str"
                )
            }

    def _is_type(
        self, key: str, type_hints: dict[str, Parameter], type_to_check: type
    ) -> bool:
        if key not in type_hints:
            return False
        if isinstance(key, int):
            args = list(type_hints.values())
            positionals = [
                p
                for p in args
                if p.kind
                in [Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD]
            ]
            param = positionals[key]
        else:
            param = type_hints[key]

        try:
            annotation = eval(param.annotation)
            origin = get_origin(annotation)
            # Check literally for DataFrame; this is why we need the import;
            # So that the module is loaded and can detect the type.
            if origin is Union:
                for arg in get_args(annotation):
                    if type_to_check == arg:
                        return True
                    if isinstance(arg, ForwardRef):
                        if arg.__forward_arg__ == type_to_check.__name__:
                            return True
                return False
            else:
                return type_to_check == annotation or isinstance(
                    annotation, type_to_check
                )
        except NameError:
            print(f"NameError: {param.annotation}")
            return False

    def is_dataframe(self, key: str, type_hints: dict[str, Parameter]) -> bool:
        """
        Check if the key is a dataframe type hint.
        """
        return self._is_type(key, type_hints, DataFrame) or self._is_type(
            key, type_hints, LazyFrame
        )

    def is_polars_type(self, key: str, type_hints: dict[str, Parameter]) -> bool:
        return self._is_type(key, type_hints, pl.DataType)

    def handle_expr(
        self,
        expr: str,
        expr_content: dict,
        variables: dict,
    ) -> pl.Expr:
        subject = pl
        if "on" in expr_content:
            on_expr = self.handle_expr(
                expr=expr_content["on"]["expr"],
                expr_content=expr_content["on"],
                variables=variables,
            )
            subject = on_expr
        # Handle polars expression prefixes like str.len etc.
        if "." in expr:
            prefix, expr = expr.split(".", 1)
            subject = getattr(subject, prefix)

        method = getattr(subject, expr)
        parameter_types = self._get_parameter_types(method)

        if "args" in expr_content:
            expr_content["args"] = [
                self.parse_kwargs(
                    {i: expr_content["args"][i]}, variables, type_hints=parameter_types
                )[i]
                for i in range(len(expr_content["args"]))
            ]
        if "kwargs" in expr_content:
            expr_content["kwargs"] = self.parse_kwargs(
                expr_content["kwargs"], variables, type_hints=parameter_types
            )
        return method(*expr_content.get("args", []), **expr_content.get("kwargs", {}))

    def parse_kwargs(self, kwargs: dict, variables: dict, type_hints: dict = None):
        """
        Parse the kwargs of a step or expression.
        """
        for key, value in kwargs.items():
            if isinstance(value, str):
                # Try to parse the value as a dataframe
                if type_hints is not None and self.is_dataframe(key, type_hints):
                    if value not in self.current_dataframes:
                        raise ValueError(
                            f"Dataframe {value} not found in current dataframes."
                            f"It is possible that the dataframe was not created"
                            f"in the previous steps."
                        )
                    kwargs[key] = self.current_dataframes[value]
                elif (
                    type_hints is not None
                    and getattr(pl, value, None)
                    and self.is_polars_type(key, type_hints)
                ):
                    kwargs[key] = getattr(pl, value)
                elif value.startswith("$$"):
                    # Handle escaped dollar sign - replace $$ with $
                    kwargs[key] = value[1:]  # Remove the first $ to unescape
                elif value.startswith("$"):
                    # Handle variable substitution
                    kwargs[key] = variables[value[1:]]
            elif isinstance(value, dict):
                if "expr" in value:
                    kwargs[key] = self.handle_expr(
                        expr=value["expr"], expr_content=value, variables=variables
                    )
                elif "custom_function" in value:
                    kwargs[key] = self.custom_functions[value["custom_function"]]
            elif isinstance(value, list):
                kwargs[key] = [
                    (
                        self.handle_expr(
                            expr=i["expr"], expr_content=i, variables=variables
                        )
                        if isinstance(i, dict)
                        else i
                    )
                    for i in value
                ]
        return kwargs

    def handle_step(
        self, current_data: Optional[pl.DataFrame], step: dict, variables: dict
    ):
        operation = step["operation"]
        args = step.get("args", [])
        kwargs = step.get("kwargs", {})
        if current_data is None:
            method = getattr(pl, operation)
        else:
            method = getattr(current_data, operation)
        parameter_types = self._get_parameter_types(method)

        # Hack our way into using the same parsing logic for args and kwargs
        parsed_args = [
            self.parse_kwargs({i: args[i]}, variables, type_hints=parameter_types)[i]
            for i in range(len(args))
        ]
        parsed_kwargs = self.parse_kwargs(kwargs, variables, type_hints=parameter_types)
        return method(*parsed_args, **parsed_kwargs)

    def run_config(self, config: dict):
        variables = config.get("variables", {})
        steps = config["steps"]
        for step in steps:
            dataframe_name = step.get("dataframe", None)
            self.current_dataframes[dataframe_name] = self.handle_step(
                self.current_dataframes.get(dataframe_name), step, variables
            )
        return self.current_dataframes


def run_config(config: dict) -> pl.DataFrame:
    return Config().run_config(config)[None]
