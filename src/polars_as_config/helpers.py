from typing import Literal


def arg_to_polars(expr):
    if isinstance(expr, dict) and "expr" in expr:
        operation = expr["expr"]
        if "on" in expr:
            prefix = arg_to_polars(expr["on"])
            operation = prefix + "." + operation
        else:
            operation = "polars." + operation
        args = ", ".join([arg_to_polars(arg) for arg in expr["args"]])
        kwargs = ", ".join(
            [
                f"{key}={arg_to_polars(value)}"
                for key, value in expr.get("kwargs", {}).items()
            ]
        )
        return f"{operation}({', '.join([i for i in [args, kwargs] if i])})"
    return repr(expr)


def json_to_polars(steps, format: Literal["oneliner", "dataframe"]):
    code = []
    for step in steps:
        operation = step["operation"]
        args = ", ".join(arg_to_polars(arg) for arg in step.get("args", []))
        kwargs = ", ".join(
            f"{key}={arg_to_polars(value)}"
            for key, value in step.get("kwargs", {}).items()
        )
        code.append(f"{operation}({', '.join([i for i in [args, kwargs] if i])})")
    if format == "oneliner":
        return "polars." + ".".join(code)
    elif format == "dataframe":
        multiline = f"df = polars.{code[0]}"
        for line in code[1:]:
            multiline += f"\ndf = df.{line}"
        return multiline
