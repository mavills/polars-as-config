class JsonToPolars:
    def arg_to_polars(self, expr):
        if isinstance(expr, dict) and "expr" in expr:
            operation = expr["expr"]
            if "on" in expr:
                prefix = self.arg_to_polars(expr["on"])
                operation = prefix + "." + operation
            else:
                operation = "polars." + operation
            args = ", ".join([self.arg_to_polars(arg) for arg in expr["args"]])
            kwargs = ", ".join(
                [
                    f"{key}={self.arg_to_polars(value)}"
                    for key, value in expr.get("kwargs", {}).items()
                ]
            )
            return f"{operation}({', '.join([i for i in [args, kwargs] if i])})"
        return repr(expr)

    def json_to_polars(self, steps):
        code = []
        index = 0
        for step in steps:
            operation = step["operation"]
            args = ", ".join(self.arg_to_polars(arg) for arg in step.get("args", []))
            kwargs = ", ".join(
                f"{key}={self.arg_to_polars(value)}"
                for key, value in step.get("kwargs", {}).items()
            )
            code_line = f"{operation}({', '.join([i for i in [args, kwargs] if i])})"
            if "dataframe_in" in step or "dataframe_out" in step:
                if "dataframe_in" in step:
                    code_line = f"{step['dataframe_in']}.{code_line}"
                else:
                    code_line = f"polars.{code_line}"
                if "dataframe_out" in step:
                    code.append(f"{step['dataframe_out']} = {code_line}")
            else:
                df_name = step.get("dataframe", "df")
                code.append(
                    f"{df_name} = {df_name if index > 0 else 'polars'}.{code_line}"
                )
            index += 1

        code = "\n".join(code)
        return code
