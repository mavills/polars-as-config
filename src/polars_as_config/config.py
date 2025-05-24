import polars as pl


def handle_expr(
    expr: str,
    expr_content: dict,
    variables: dict,
) -> pl.Expr:
    if "kwargs" in expr_content:
        expr_content["kwargs"] = parse_kwargs(expr_content["kwargs"], variables)

    subject = pl
    if "on" in expr_content:
        on_expr = handle_expr(
            expr=expr_content["on"]["expr"],
            expr_content=expr_content["on"],
            variables=variables,
        )
        subject = on_expr
    expr_content["kwargs"].pop("kwargs", None)
    expr_content["kwargs"].pop("on", None)
    expr_content["kwargs"].pop("expr", None)
    if expr.startswith("str."):
        subject = subject.str
        expr = expr[4:]
    return getattr(subject, expr)(**expr_content["kwargs"])


def parse_kwargs(kwargs: dict, variables: dict):
    """
    Parse the kwargs of a step or expression.
    """
    for key, value in kwargs.items():
        if key == "kwargs":
            kwargs[key] = parse_kwargs(value, variables)
        elif isinstance(value, str):
            if value.startswith("$") and not value.startswith("$$"):
                kwargs[key] = variables[value[1:]]
        elif isinstance(value, dict):
            if "expr" in value:
                kwargs[key] = handle_expr(
                    expr=value["expr"], expr_content=value, variables=variables
                )
            else:
                raise ValueError(f"Invalid kwarg object type: {type(value)}, {value}")
    return kwargs


def handle_step(current_data, step: dict, variables: dict):
    operation = step["operation"]
    kwargs = step["kwargs"]
    if current_data is None:
        method = getattr(pl, operation)
    else:
        method = getattr(current_data, operation)

    parsed_kwargs = parse_kwargs(kwargs, variables)
    return method(**parsed_kwargs)


def run_config(config: dict):
    variables = config.get("variables", {})
    steps = config["steps"]
    current_data = None
    for step in steps:
        current_data = handle_step(current_data, step, variables)
    return current_data
