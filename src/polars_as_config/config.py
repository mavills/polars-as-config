from typing import Callable
import polars as pl


class Config:
    custom_functions: dict[str, Callable] = {}

    def add_custom_functions(self, functions: dict) -> "Config":
        self.custom_functions.update(functions)
        return self

    def handle_expr(
        self,
        expr: str,
        expr_content: dict,
        variables: dict,
    ) -> pl.Expr:
        if "args" in expr_content:
            expr_content["args"] = [
                self.parse_kwargs({i: expr_content["args"][i]}, variables)[i]
                for i in range(len(expr_content["args"]))
            ]
        if "kwargs" in expr_content:
            expr_content["kwargs"] = self.parse_kwargs(
                expr_content["kwargs"], variables
            )

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
        return getattr(subject, expr)(
            *expr_content.get("args", []), **expr_content.get("kwargs", {})
        )

    def parse_kwargs(self, kwargs: dict, variables: dict):
        """
        Parse the kwargs of a step or expression.
        """
        for key, value in kwargs.items():
            if isinstance(value, str):
                if value.startswith("$$"):
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
        return kwargs

    def handle_step(self, current_data, step: dict, variables: dict):
        operation = step["operation"]
        args = step.get("args", [])
        kwargs = step.get("kwargs", {})
        if current_data is None:
            method = getattr(pl, operation)
        else:
            method = getattr(current_data, operation)

        # Hack our way into using the same parsing logic for args and kwargs
        parsed_args = [
            self.parse_kwargs({i: args[i]}, variables)[i] for i in range(len(args))
        ]
        parsed_kwargs = self.parse_kwargs(kwargs, variables)
        return method(*parsed_args, **parsed_kwargs)

    def run_config(self, config: dict):
        variables = config.get("variables", {})
        steps = config["steps"]
        current_data = None
        for step in steps:
            current_data = self.handle_step(current_data, step, variables)
        return current_data


def run_config(config: dict) -> pl.DataFrame:
    return Config().run_config(config)
