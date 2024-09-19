"""This module implements custom errors to ensure
a better understand of errors in the agentlib"""


class InitializationError(Exception):
    """
    Exception raised for errors due to wrong initialization of modules.
    """


class ConfigurationError(Exception):
    """
    Exception raised for errors due to wrong configuration of modules.
    """


class OptionalDependencyError(Exception):
    """
    Exception to indicate that an optional dependency is missing
    which can always be fixed by installing the missing
    dependency.
    """

    def __init__(
        self,
        used_object: str,
        dependency_install: str,
        dependency_name: str = None,
    ):
        message = (
            f"{used_object} is an optional dependency which you did not "
            f"install yet. Install the missing dependency "
            f"using `pip install {dependency_install}`"
        )
        if dependency_name is not None:
            message += (
                ", or by re-installing the agentlib using "
                "`pip install agentlib[full]` or just the "
                f"key (e.g. for {dependency_name}: "
                f"`pip install agentlib[{dependency_name}]`"
            )
        super().__init__(message)
