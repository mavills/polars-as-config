import inspect
from inspect import Parameter
from typing import Any, Callable, ForwardRef, Optional, Union, get_args, get_origin

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

    def _get_type_from_hints(
        self, key: str | int, type_hints: dict[str, Parameter]
    ) -> type | None:
        if key not in type_hints:
            return None
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
        except NameError:
            return None
        return annotation

    def _is_type(self, annotation: type | None, type_to_check: type) -> bool:
        """
        Check if the annotation is a type.
        Returns false if the annotation is None.
        """
        if annotation is None:
            return False
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
            return type_to_check == annotation or isinstance(annotation, type_to_check)

    def is_dataframe(self, annotation: type | None) -> bool:
        """
        Check if the annotation is a dataframe type.

        Example dataframe types:
        - pl.DataFrame
        - pl.LazyFrame
        """
        return self._is_type(annotation, DataFrame) or self._is_type(
            annotation, LazyFrame
        )

    def is_polars_type(self, annotation: type | None) -> bool:
        """
        Check if the annotation is a polars type.

        Example polars types:
        - pl.Int64
        - pl.Float64
        - pl.Boolean
        - pl.Utf8
        - pl.Date
        - pl.Datetime
        - pl.Time
        """
        return self._is_type(annotation, pl.DataType)

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

    def parse_nesting(self, value: Any, variables: dict) -> str:
        if isinstance(value, list):
            return [self.parse_nesting(i, variables) for i in value]
        elif isinstance(value, dict):
            return {k: self.parse_nesting(v, variables) for k, v in value.items()}
        else:
            return value

    def parse_string(self, value: str, variables: dict, type_hint):
        """
        Parse a string value when encountered.
        Can be a dataframe, polars type, variable, or escaped dollar sign.
        """
        if self.is_dataframe(type_hint):
            if value not in self.current_dataframes:
                raise ValueError(
                    f"Dataframe {value} not found in current dataframes."
                    f"It is possible that the dataframe was not created"
                    f"in the previous steps."
                )
            return self.current_dataframes[value]
        elif getattr(pl, value, None) and self.is_polars_type(type_hint):
            return getattr(pl, value)
        elif value.startswith("$$"):
            # Handle escaped dollar sign - replace $$ with $
            return value[1:]  # Remove the first $ to unescape
        elif value.startswith("$"):
            # Handle variable substitution
            return variables[value[1:]]
        else:
            return value

    def parse_dict(self, value: dict, variables: dict, type_hint):
        """
        Parse a dict value when encountered.
        Can be a dataframe, polars type, variable, or escaped dollar sign.
        """
        if "expr" in value:
            return self.handle_expr(
                expr=value["expr"], expr_content=value, variables=variables
            )
        elif "custom_function" in value:
            return self.custom_functions[value["custom_function"]]
        return {k: self.parse_value(v, variables, None) for k, v in value.items()}

    def parse_value(self, value: str | dict | list, variables: dict, type_hint):
        if isinstance(value, str):
            return self.parse_string(value, variables, type_hint)
        elif isinstance(value, dict):
            return self.parse_dict(value, variables, type_hint)
        elif isinstance(value, list):
            return self.parse_list(value, variables, type_hint)

    def get_list_subtype(self, type_hint):
        if type_hint is None:
            return None
        origin = get_origin(type_hint)
        if origin is list:
            return get_args(type_hint)[0]
        return type_hint

    def parse_list(self, value: list, variables: dict, type_hint):
        subtype = self.get_list_subtype(type_hint)
        return [self.parse_value(i, variables, subtype) for i in value]

    def parse_kwargs(self, kwargs: dict, variables: dict, type_hints: dict = None):
        """
        Parse the kwargs of a step or expression.
        """
        for key, value in kwargs.items():
            kwargs[key] = self.parse_value(value, variables, type_hints.get(key))
            type_to_check = self._get_type_from_hints(key, type_hints)
            if isinstance(value, str):
                # Try to parse the value as a dataframe
                kwargs[key] = self.parse_string(value, variables, type_to_check)
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
