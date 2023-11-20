"""
The module contains the relevant classes
to execute and use the DataBroker.
Besides the DataBroker itself, the BrokerCallback is defined.

Internally, uses the tuple _map_tuple in the order of

(alias, source)

to match callbacks and AgentVariables.

"""

import abc
import inspect
import logging
import threading
import queue
from typing import (
    List,
    Callable,
    Dict,
    Tuple,
    Optional,
    Protocol,
    runtime_checkable,
    Any,
)

from pydantic import BaseModel, field_validator, model_validator, ConfigDict

from agentlib.core.datamodels import AgentVariable, Source
from agentlib.core.environment import Environment

logger = logging.getLogger()


@runtime_checkable
class CallbackFunction(Protocol):
    """Protocol defining the signature of a valid Callback Function"""

    def __call__(self, variable: AgentVariable, **kwargs: Any) -> None:
        ...

    __name__: str


class NoCopyBrokerCallback(BaseModel):
    """
    Basic broker callback.
    This object does not copy the AgentVariable
    before calling the callback, which can be unsafe.

    This class checks if the given callback function
    adheres to the signature it needs to be correctly called.
    The first argument will be an AgentVariable. If a type-hint
    is specified, it must be `AgentVariable` or `"AgentVariable"`.
    Any further arguments must match the kwargs
    specified in the class and will also be the ones you
    pass to this class.

    Example:
    >>> def my_callback(variable: "AgentVariable", some_static_info: str):
    >>>     print(variable, some_other_info)
    >>> NoCopyBrokerCallback(
    >>>     callback=my_callback,
    >>>     kwargs={"some_static_info": "Hello World"}
    >>> )

    """

    # pylint: disable=too-few-public-methods
    callback: CallbackFunction
    alias: Optional[str] = None
    source: Optional[Source] = None
    kwargs: dict = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def check_valid_callback_function(cls, data):
        """Ensures the callback function signature is valid."""
        func_params = dict(inspect.signature(data["callback"]).parameters)
        par = func_params.pop(next(iter(func_params)))
        if par.annotation is not par.empty and par.annotation not in (
            "AgentVariable",
            AgentVariable,
        ):
            raise RuntimeError(
                "Defined callback Function does not take an "
                "AgentVariable as first parameter"
            )

        if not list(data["kwargs"]) == list(func_params):
            kwargs_not_in_function_args = set(list(data["kwargs"])).difference(
                list(func_params)
            )
            function_args_not_in_kwargs = set(list(func_params)).difference(
                list(data["kwargs"])
            )
            if function_args_not_in_kwargs:
                missing_kwargs = "Missing arguments in kwargs: " + ", ".join(
                    function_args_not_in_kwargs
                )
            else:
                missing_kwargs = ""
            if kwargs_not_in_function_args:
                missing_func_args = "Missing kwargs in function call: " + ", ".join(
                    kwargs_not_in_function_args
                )
            else:
                missing_func_args = ""
            raise RuntimeError(
                "The registered Callback secondary arguments do not match the given kwargs:\n"
                f"{missing_func_args}\n"
                f"{missing_kwargs}"
            )
        return data

    def __eq__(self, other: "NoCopyBrokerCallback"):
        """
        Check equality to another callback using equality of all fields
        and the name of the callback function
        """
        return (self.alias, self.source, self.kwargs, self.callback.__name__) == (
            other.alias,
            other.source,
            other.kwargs,
            other.callback.__name__,
        )


class BrokerCallback(NoCopyBrokerCallback):
    """
    This broker callback always creates a deep-copy of the
    AgentVariable it is going to send.
    It is considered the safer option, as the receiving module
    only get's the values and is not able to alter
    the AgentVariable for other modules.
    """

    @field_validator("callback")
    @classmethod
    def auto_copy(cls, callback_func: CallbackFunction):
        """Automatically supply the callback function with a copy"""

        def callback_copy(variable: AgentVariable, **kwargs):
            callback_func(variable.copy(deep=True), **kwargs)

        callback_copy.__name__ = callback_func.__name__
        return callback_copy


class DataBroker(abc.ABC):
    """
    Handles communication and Callback triggers within an agent.
    Write variables to the broker with ``send_variable()``.
    Variables send to the broker will trigger callbacks
    based on the alias and the source of the variable.
    Commonly, this is used to provide other
    modules with the variable.

    Register and de-register Callbacks to the DataBroker
    with ``register_callback`` and ``deregister_callback``.
    """

    def __init__(self):
        """
        Initialize lock, callbacks and entries
        """
        self._mapped_callbacks: Dict[Tuple[str, Source], List[BrokerCallback]] = {}
        self._unmapped_callbacks: List[BrokerCallback] = []
        self._variable_queue = queue.SimpleQueue()

    def send_variable(self, variable: AgentVariable, copy: bool = True):
        """
        Send variable to data_broker. Evokes callbacks associated with this variable.

        Args:
            variable AgentVariable:
                The variable to set.
            copy boolean:
                Whether to copy the variable before sending.
                Default is True.
        """
        if copy:
            self._send_variable_to_modules(variable=variable.copy(deep=True))
        else:
            self._send_variable_to_modules(variable=variable)

    def _send_variable_to_modules(self, variable: AgentVariable):
        """
        Enqueue AgentVariable in local queue for executing relevant callbacks.

        Args:
            variable AgentVariable: The variable to append to the local queue.
        """
        self._variable_queue.put(variable)

    def _execute_callbacks(self):
        """
        Run relevant callbacks for AgentVariable's from local queue.
        """
        variable = self._variable_queue.get(block=True)
        _map_tuple = (variable.alias, variable.source)
        # First the unmapped cbs
        callbacks = self._filter_unmapped_callbacks(map_tuple=_map_tuple)
        # Then the mapped once.
        # Use try-except to avoid possible deregister during check and execution
        try:
            callbacks.extend(self._mapped_callbacks[_map_tuple])
        except KeyError:
            pass

        # Then run the callbacks
        for cb in callbacks:
            cb.callback(variable, **cb.kwargs)

    def _filter_unmapped_callbacks(self, map_tuple: tuple) -> List[BrokerCallback]:
        """
        Filter the unmapped callbacks according to the given
        tuple of variable information.

        Args:
            map_tuple tuple:
                The tuple of alias and source in that order

        Returns:
            List[BrokerCallback]: The filtered list

        """
        # Filter all callbacks matching the given variable
        callbacks = self._unmapped_callbacks
        # First filter source
        source = map_tuple[1]
        callbacks = [
            cb for cb in callbacks if (cb.source is None) or (cb.source.matches(source))
        ]
        # Now alias
        callbacks = [
            cb for cb in callbacks if (cb.alias is None) or (cb.alias == map_tuple[0])
        ]

        return callbacks

    def register_callback(
        self,
        callback: Callable,
        alias: str = None,
        source: Source = None,
        _unsafe_no_copy: bool = False,
        **kwargs,
    ):
        """
        Register a callback to the data_broker.

        Args:
            callback callable: The function of the callback
            alias str: The alias of variables to trigger callback
            source Source: The Source of variables to trigger callback
            kwargs dict: Kwargs to be passed to the callback function
            _unsafe_no_copy: If True, the callback will not be passed a copy, but the
                original AgentVariable. When using this option, the user promises to not
                modify the AgentVariable, as doing so could lead to
                wrong and difficult to debug behaviour in other modules (default False)
        """
        if _unsafe_no_copy:
            callback = NoCopyBrokerCallback(
                alias=alias, source=source, callback=callback, kwargs=kwargs
            )
        else:
            callback = BrokerCallback(
                alias=alias, source=source, callback=callback, kwargs=kwargs
            )
        _map_tuple = (alias, source)
        if self.any_is_none(alias=alias, source=source):
            self._unmapped_callbacks.append(callback)
        elif _map_tuple in self._mapped_callbacks:
            self._mapped_callbacks[_map_tuple].append(callback)
        else:
            self._mapped_callbacks[_map_tuple] = [callback]

    def deregister_callback(
        self, callback: Callable, alias: str = None, source: Source = None, **kwargs
    ):
        """
        Deregister the given callback based on given
        alias and source.

        Args:
            callback callable: The function of the callback
            alias str: The alias of variables to trigger callback
            source Source: The Source of variables to trigger callback
            kwargs dict: Kwargs of the callback function
        """
        try:
            callback = BrokerCallback(
                alias=alias, source=source, callback=callback, kwargs=kwargs
            )
            _map_tuple = (alias, source)
            if self.any_is_none(alias=alias, source=source):
                self._unmapped_callbacks.remove(callback)
            elif _map_tuple in self._mapped_callbacks:
                self._mapped_callbacks[_map_tuple].remove(callback)
            else:
                return  # No delete necessary
            logger.debug("Callback de-registered: %s", callback)
        except ValueError:
            pass

    @staticmethod
    def any_is_none(alias: str, source: Source) -> bool:
        """
        Return True if any of alias or source are None.

        Args:
            alias str:
                The alias of the callback
            source Source:
                The Source of the callback
        """
        return (
            (alias is None)
            or (source is None)
            or (source.agent_id is None)
            or (source.module_id is None)
        )


class LocalDataBroker(DataBroker):
    """Local variation of the DataBroker written for fast-as-possible
    simulation within a single non-realtime Environment."""

    def __init__(self, env: Environment):
        """
        Initialize env
        """
        self.env = env
        super().__init__()
        self._callbacks_available = self.env.event()

    def _send_variable_to_modules(self, variable: AgentVariable):
        """
        Enqueue AgentVariable in local queue for executing relevant callbacks.

        Args:
            variable AgentVariable: The variable to append to the local queue.
        """
        super()._send_variable_to_modules(variable)
        self._callbacks_available.callbacks.append(self._execute_callback_simpy)
        self._callbacks_available.succeed()
        self._callbacks_available = self.env.event()

    def _execute_callback_simpy(self, ignored):
        """
        Run relevant callbacks for AgentVariable's from local queue.
        To be appended to the callback of the callbacks available event.
        """
        self._execute_callbacks()


class RTDataBroker(DataBroker):
    """DataBroker written for Realtime operation regardless of Environment."""

    def __init__(self, env: Environment):
        """
        Initialize env.
        Adds the function to start callback execution to the environment as a process.
        Since the databroker is initialized before the modules, this will always be
        the first triggered event, so no other process starts before the broker is
        ready
        """
        super().__init__()
        self.thread = threading.Thread(
            target=self._callback_thread, daemon=True, name="DataBroker"
        )

        env.process(self._start_executing_callbacks(env))

    def _start_executing_callbacks(self, env: Environment):
        """
        Starts the callback thread.
        Thread is started after it is registered by the agent. Should be fine, since
        the monitor process is started after the process in this function
        """
        self.thread.start()
        yield env.event()

    def _callback_thread(self):
        """Thread to check and process the callback queue in Realtime
        applications."""
        while True:
            self._execute_callbacks()
