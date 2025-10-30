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
import queue
import threading
from typing import (
    List,
    Callable,
    Dict,
    Tuple,
    Optional,
    Union,
)

from pydantic import BaseModel, field_validator, model_validator, ConfigDict

from agentlib.core.datamodels import AgentVariable, Source
from agentlib.core.environment import Environment
from agentlib.core.logging_ import CustomLogger
from agentlib.core.module import BaseModule


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
    callback: Callable
    alias: Optional[str] = None
    source: Optional[Source] = None
    kwargs: dict = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)
    module_id: Optional[str] = None

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
        # note from which module this callback came. If it is not a bound method, we
        # assign it to none
        try:
            if isinstance(data["callback"].__self__, BaseModule):
                module_id = data["callback"].__self__.id
            else:
                module_id = None
        except AttributeError:
            module_id = None
        data["module_id"] = module_id
        return data

    def __eq__(self, other: "NoCopyBrokerCallback"):
        """
        Check equality to another callback using equality of all fields
        and the name of the callback function
        """
        return (
            self.alias,
            self.source,
            self.kwargs,
            self.callback.__name__,
            self.module_id,
        ) == (
            other.alias,
            other.source,
            other.kwargs,
            other.callback.__name__,
            other.module_id,
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
    def auto_copy(cls, callback_func: Callable):
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

    def __init__(self, logger: CustomLogger, max_queue_size: int = 1000):
        """
        Initialize lock, callbacks and entries
        """
        self.logger = logger
        self._max_queue_size = max_queue_size
        self._mapped_callbacks: Dict[Tuple[str, Source], List[BrokerCallback]] = {}
        self._unmapped_callbacks: List[BrokerCallback] = []
        self._variable_queue = queue.Queue(maxsize=max_queue_size)

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
        log_queue_status(
            logger=self.logger,
            queue_name="Callback-Distribution",
            queue_object=self._variable_queue,
            max_queue_size=self._max_queue_size,
        )
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
        self._run_callbacks(callbacks, variable)

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
    ) -> Union[BrokerCallback, NoCopyBrokerCallback]:
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
            callback_ = NoCopyBrokerCallback(
                alias=alias, source=source, callback=callback, kwargs=kwargs
            )
        else:
            callback_ = BrokerCallback(
                alias=alias, source=source, callback=callback, kwargs=kwargs
            )
        _map_tuple = (alias, source)
        if self.any_is_none(alias=alias, source=source):
            self._unmapped_callbacks.append(callback_)
        elif _map_tuple in self._mapped_callbacks:
            self._mapped_callbacks[_map_tuple].append(callback_)
        else:
            self._mapped_callbacks[_map_tuple] = [callback_]
        return callback_

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
            self.logger.debug("Callback de-registered: %s", callback)
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

    @staticmethod
    def _run_callbacks(callbacks: List[BrokerCallback], variable: AgentVariable):
        """Runs the callbacks on a single AgentVariable."""
        raise NotImplementedError


class LocalDataBroker(DataBroker):
    """Local variation of the DataBroker written for fast-as-possible
    simulation within a single non-realtime Environment."""

    def __init__(
        self, env: Environment, logger: CustomLogger, max_queue_size: int = 1000
    ):
        """
        Initialize env
        """
        self.env = env
        super().__init__(logger=logger, max_queue_size=max_queue_size)
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

    def _run_callbacks(self, callbacks: List[BrokerCallback], variable: AgentVariable):
        """Runs callbacks of an agent on a single AgentVariable in sequence.
        Used in fast-as-possible execution mode."""
        for cb in callbacks:
            cb.callback(variable, **cb.kwargs)


class RTDataBroker(DataBroker):
    """DataBroker written for Realtime operation regardless of Environment."""

    def __init__(
        self, env: Environment, logger: CustomLogger, max_queue_size: int = 1000
    ):
        """
        Initialize env.
        Adds the function to start callback execution to the environment as a process.
        Since the databroker is initialized before the modules, this will always be
        the first triggered event, so no other process starts before the broker is
        ready
        """
        super().__init__(logger=logger, max_queue_size=max_queue_size)
        self._stop_queue = queue.SimpleQueue()
        self.thread = threading.Thread(
            target=self._callback_thread, daemon=True, name="DataBroker"
        )
        self._module_queues: dict[Union[str, None], queue.Queue] = {}

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
            if not self._stop_queue.empty():
                err, module_id = self._stop_queue.get()
                raise RuntimeError(
                    f"A callback failed in the module {module_id}."
                ) from err
            self._execute_callbacks()

    def register_callback(
        self,
        callback: Callable,
        alias: str = None,
        source: Source = None,
        _unsafe_no_copy: bool = False,
        **kwargs,
    ) -> Union[NoCopyBrokerCallback, BrokerCallback]:
        # check to which object the callable is bound, to determine the module
        callback = super().register_callback(
            callback=callback,
            alias=alias,
            source=source,
            _unsafe_no_copy=_unsafe_no_copy,
            **kwargs,
        )
        if callback.module_id not in self._module_queues:
            self._start_module_thread(callback.module_id)
        return callback

    def _start_module_thread(self, module_id: str):
        """Starts a consumer thread for callbacks registered from a module."""
        module_queue = queue.Queue(maxsize=self._max_queue_size)
        threading.Thread(
            target=self._execute_callbacks_of_module,
            daemon=True,
            name=f"DataBroker/{module_id}",
            kwargs={"queue": module_queue, "module_id": module_id},
        ).start()
        self._module_queues[module_id] = module_queue

    def _execute_callbacks_of_module(self, queue: queue.SimpleQueue, module_id: str):
        """Executes the callbacks associated with a specific module."""
        try:
            while True:
                cb, variable = queue.get(block=True)
                cb.callback(variable=variable, **cb.kwargs)
        except Exception as e:
            self._stop_queue.put((e, module_id))
            raise e

    def _run_callbacks(self, callbacks: List[BrokerCallback], variable: AgentVariable):
        """Distributes callbacks to the threads running for each module."""
        for cb in callbacks:
            self._module_queues[cb.module_id].put_nowait((cb, variable))
            log_queue_status(
                logger=self.logger,
                queue_name=cb.module_id,
                queue_object=self._module_queues[cb.module_id],
                max_queue_size=self._max_queue_size,
            )


def log_queue_status(
    logger: logging.Logger,
    queue_object: queue.Queue,
    max_queue_size: int,
    queue_name: str,
):
    """
    Log the current load of the given queue in percent.

    Args:
         logger (logging.Logger): A logger instance
         queue_object (queue.Queue): The queue object
         max_queue_size (int): Maximal queue size
         queue_name (str): Name associated with the queue
    """
    if max_queue_size < 1:
        return
    number_of_items = queue_object.qsize()
    percent_full = round(number_of_items / max_queue_size * 100, 2)
    if percent_full < 10:
        return
    elif percent_full < 80:
        logger_func = logger.debug
    else:
        logger_func = logger.warning
    logger_func(
        "Queue '%s' fullness is %s percent (%s items).",
        queue_name,
        percent_full,
        number_of_items,
    )
