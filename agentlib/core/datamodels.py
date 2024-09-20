"""
The datamodels module contains all classes
defining basic models to handle data.
"""

import abc
import functools
import json
import logging
import math
import numbers
from copy import copy, deepcopy
from enum import Enum
from io import StringIO
from itertools import chain
from typing import Union, Any, List, Optional, TypeVar, Set, Container, get_args

import attrs
import numpy as np
import pandas as pd
from attrs import define, field
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from pydantic_core.core_schema import CoreSchema, SerializationInfo

logger = logging.getLogger(__name__)


ATTRS_MODELS = [
    "AgentVariables",
    "AgentVariable",
    "BaseVariable",
    "ModelInput",
    "ModelState",
    "ModelOutput",
    "ModelParameter",
    "ModelInputs",
    "ModelStates",
    "ModelOutputs",
    "ModelParameters",
    "ModelVariable",
    "Source",
]


__all__ = ATTRS_MODELS + [
    "Causality",
    "Variability",
]


class Causality(str, Enum):
    """
    Enumeration that defines the causality of a variable.

    The default causality is “local”.

    Allowed params of his enumeration:
    """

    # _init_ = "value __doc__"
    parameter = "parameter"
    calculatedParameter = "calculatedParameter"

    input = "input"
    output = "output"
    local = "local"
    independent = "independent"


class Variability(str, Enum):
    """
    Enumeration that defines the time dependency of the variable, in other
    words,it defines the time instants when a variable can change its value.
    [The purpose of this attribute is to define when a result value needs to
    be inquired and to be stored. For example, discrete variables change
    their params only at event instants (ModelExchange) or at a
    communication point (CoSimulation) and it is therefore only necessary
    to inquire and store them at event times].

    The default is “continuous”

    Allowed params of this enumeration:
    """

    constant = "constant"
    fixed = "fixed"
    tunable = "tunable"
    discrete = "discrete"
    continuous = "continuous"


###############################################################################
# Custom Field types
###############################################################################


class AttrsToPydanticAdaptor(abc.ABC):
    """
    Class to use the attrs-based class in pydantic models.
    """

    @abc.abstractmethod
    def dict(self):
        """Returns the dict object of the class."""
        raise NotImplementedError

    @abc.abstractmethod
    def json(self) -> str:
        """Returns json serialization of the class"""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def create(cls, data: Union[dict, "AttrsToPydanticAdaptor"]):
        raise NotImplementedError

    def _serialize(self, _info: SerializationInfo):
        """Function required for pydantic"""
        if _info.mode == "python":
            return self.dict()
        return self.json()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Tells pydantic how to instantiate and validate this class."""
        return core_schema.no_info_after_validator_function(
            cls.create,  # validator
            core_schema.any_schema(ref=cls.__name__),  # what to call before validator
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize,
                info_arg=True,
                return_schema=core_schema.any_schema(),
            ),  # serialization
        )

    @classmethod
    def get_json_schema(cls):
        """
        Return the JSON-Schema of the class.
        Will try to create the schema based on the attrs-fields
        and existing types. Specially nested types may not work,
        e.g. Union[Dict[str, AgentVariable], List[Source]] o.s.
        However, the current Variables and Source do not require
        such complex type annotations.
        """

        def _get_type_schema(_type: object) -> (dict, list):
            """
            Function to return the schema for the given _type.
            The type is the one given in the attr-field.
            Special types, which are not the _type_map, will not work properly.
            Avoiding the .get method to explicitly raise the error if
            future versions of custom AttrsToPydanticAdaptor use such types.

            Returns the schema as the first argument and a list of references
            as the second.
            """
            # Map to infer the json-schema name for the type
            # from the python object
            _type_map = {
                str: "string",
                bool: "boolean",
                int: "integer",
                float: "number",
                list: "array",
            }
            if issubclass(_type, AttrsToPydanticAdaptor):
                return {"$ref": f"#/definitions/{_type.__name__}"}, _type.__name__
            else:
                return {"type": _type_map[_type]}, []

        def _get_typing_schema(_type) -> (dict, list):
            """
            Recursive to extract the type schema for all possible types
            which currently occur in the attrs fields. This includes
            standard types like str and typing-types like Union[str, float].

            Returns the schema as the first argument and a list of references
            as the second.
            """
            if _type == Any:
                # Any can't be used by GUIs or OpenAPI anyway. At the
                # same time, it is the only type in typing with no __origin__.
                # TODO-ses: We could also return string as the type, as
                #   I had to tweak streamlit-pydantic to render Any as
                #   string. Depends on streamlit-pydantic fork moves forward
                return {}, []
            if isinstance(_type, type):
                return _get_type_schema(_type)

            # If it's not a type object, it currently is always a typing object,
            # which indicates the actual type using __origin__.
            # We could also use `get_origin` from typing.
            if isinstance(_type.__origin__, type):
                return _get_type_schema(_type.__origin__)
            _types = get_args(_type)
            if type(None) in _types:  # Is optional
                return _get_typing_schema(_types[0])  # 2nd entry will be None
            if _type.__origin__ == Union:
                refs = []
                _any_of_types = []
                for _type in _types:
                    _any_of_type, _ref = _get_type_schema(_type)
                    refs.append(_ref)
                    _any_of_types.append(_any_of_type)
                return {"anyOf": _any_of_types}, refs
            raise TypeError(
                f"Given type '{_type}' is not supported for JSONSchema export"
            )

        field_schemas = {}
        required = []
        all_refs = []
        for attr in attrs.fields(cls):
            field_schemas[attr.name] = dict(attr.metadata)
            _type_schema, _refs = _get_typing_schema(attr.type)
            all_refs.extend(_refs)
            field_schemas[attr.name].update(_type_schema)
            # Get default
            if attr.default:
                field_schemas[attr.name].update({"default": attr.default})
            else:
                required.append(attr.name)

        schema = {
            "title": cls.__name__,
            "description": cls.__doc__,
            "type": "object",
            "properties": field_schemas,
            "required": required,
            "definitions": [
                f"$defs/{ref}" for ref in set([r for r in list(all_refs) if r])
            ],
            "additionalProperties": False,
        }

        return schema


@define(slots=True, frozen=True)
class Source(AttrsToPydanticAdaptor):
    """
    Object to define the source of a variable or possible
    other object. As objects are passed both module-internally
    by agents or across multiple agents, both the
    agent_id and module_id build up the source object.

    However, with methods like 'matches', one can indicate
    setting any id to None that the id is irrelevant.
    """

    agent_id: Optional[str] = None
    module_id: Optional[str] = None

    def __str__(self):
        return f"{self.agent_id}_{self.module_id}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, dict):
            return self.agent_id == other.get(
                "agent_id"
            ) and self.module_id == other.get("module_id")
        if isinstance(other, Source):
            return self.agent_id == other.agent_id and self.module_id == other.module_id
        return False

    def dict(self):
        """Overwrite pydantic method to be faster."""
        # pylint: disable=unused-argument
        return {"agent_id": self.agent_id, "module_id": self.module_id}

    def json(self) -> str:
        """Returns json serialization of the Source"""
        return json.dumps(self.dict())

    @classmethod
    def create(cls, data: Union[dict, "Source", str]):
        """Constructor for this class, used by pydantic."""
        if isinstance(data, str):
            return cls(agent_id=data)
        if isinstance(data, Source):
            return data
        if isinstance(data, dict):
            return cls(**data)

    def matches(self, other) -> bool:
        """
        Define if the current source matches another source:

        First, convert other object to a dict. The dict must contain
        the variables agent_id and module_id. If one of the values is None,
        it is excluded from the comparison.
        Args:
            other Union[Source, Dict]: Another source to compare

        Returns:
            Boolean if they match
        """
        if self.agent_id is None:
            ag_match = True
        else:
            ag_match = self.agent_id == other.agent_id
        if self.module_id is None:
            mo_match = True
        else:
            mo_match = self.module_id == other.module_id

        return mo_match and ag_match


################################################################################
# Variable Definitions
################################################################################
_TYPE_MAP: dict = {
    "Real": float,
    "Boolean": bool,
    "Integer": int,
    "Enumeration": int,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
    "pd.Series": pd.Series,
    "list": list,
}


def none_to_inf(x):
    """
    Convert x to inf if it is None.
    """
    if x is None:
        return math.inf
    return x


def none_to_minus_inf(x):
    """
    Convert x to -inf if it is None.
    """
    if x is None:
        return -math.inf
    return x


@functools.lru_cache
def slot_helper(cls: type) -> Set[str]:
    """Return the set of all slots of a class and its parents.

    This function is needed, as cls.__slots__ only returns the new slots that class
    defined, but not the slots which it inherited. We have to manually traverse the
    inheritance chain to get all slots.
    Since this function is called whenever a variable has its dict called, we cache the
     results for performance.
    """
    return set(chain.from_iterable(parent.__slots__ for parent in cls.mro()[:-1]))


@define(slots=True, weakref_slot=False, kw_only=True)
class BaseVariable(AttrsToPydanticAdaptor):
    """
    BaseVariable for all Variables inside the agentlib.
    This includes Model Variables and AgentVariables.
    A Variable can have an arbitrary value with several forms of validation for type,
    boundaries and allowed values.
    """

    name: str = field(
        metadata={"title": "Name", "description": "The name of the variable"}
    )
    type: Optional[str] = field(
        default=None,
        metadata={
            "title": "Type",
            "description": "Name the type of the variable using a string",
        },
    )
    timestamp: Optional[float] = field(
        default=None,
        metadata={
            "title": "Timestamp",
            "description": "Timestamp of the current value",
        },
    )
    unit: str = field(
        default="Not defined",
        metadata={"title": "Unit", "description": "Unit of the variable"},
    )
    description: str = field(
        default="Not defined",
        metadata={"title": "Description", "description": "Description of the variable"},
    )
    ub: Union[float, int] = field(
        default=math.inf,
        converter=none_to_inf,
        metadata={
            "title": "Upper boundary",
            "description": "Upper bound of the variables value, "
            "used only for numeric values.",
        },
    )
    lb: Union[float, int] = field(
        default=-math.inf,
        converter=none_to_minus_inf,
        metadata={
            "title": "Lower boundary",
            "description": "Lower bound of the variables value, "
            "used only for numeric values.",
        },
    )
    clip: bool = field(
        default=False,
        metadata={
            "title": "Clip values to boundaries",
            "description": "If set to true, values outside "
            "of bounds will be clipped of",
        },
    )
    # we use list here, because json deserializes to list by default. While membership
    # check is slower with list, on average we think this is more performant, since
    # the feature is not used very often
    allowed_values: List[Any] = field(
        default=[],
        metadata={
            "title": "Allowed values",
            "description": "If provided, the value may only "
            "be any value inside the given set "
            "of allowed values. "
            "Example would be to only allow only "
            "the string options 'Create', 'Update' and "
            "'Delete'. Then you should pass "
            "allowed_values=['Create', 'Update', 'Delete']",
        },
    )
    value: Any = field(
        default=None,
        metadata={"title": "Value", "description": "The value of the variable"},
    )

    @classmethod
    def validate_data(cls, data: dict) -> "BaseVariable":
        """Constructor that performs all validation."""
        instance = cls(**data)
        instance.run_validation()
        return instance

    @classmethod
    def create(cls, data: Union[dict, "BaseVariable"]) -> "BaseVariable":
        """Constructor for pydantic."""
        if isinstance(data, cls):
            return data
        return cls.validate_data(data)

    def run_validation(self):
        """Performs all validations."""
        self._check_bound()
        self._check_value_type()
        self._check_type_of_allowed_values()
        self.set_value(self.value, validate=True)

    def _check_bound(self):
        """
        First checks if the boundaries lb and ub are either numeric
        or if they can be converted to a float. If they can, they
        are converted.
        Second, lower bound must be lower or equal than upper bound
        """
        for name, bound in {"lb": self.lb, "ub": self.ub}.items():
            if isinstance(bound, numbers.Real):
                continue
            try:
                self.__setattr__(name, float(bound))
            except ValueError:
                raise ValueError(f"Given bound {name} is not a valid number.")
        if self.lb > self.ub:
            raise ValueError("Given upper bound (ub) is lower than lower bound (lb)")

    def _check_value_type(self):
        """Validator for the type field. Makes sure the type is in the type map."""
        if self.type is None:
            return
        if not isinstance(self.type, str):
            raise TypeError(
                f"Given types value is of type {type(self.type)} "
                f"but should be a string."
            )
        if self.type not in _TYPE_MAP:
            raise ValueError(
                f"Given type '{self.type}' is not supported. "
                f"Currently supported options are "
                f"{', '.join(_TYPE_MAP.keys())}"
            )

    def _check_type_of_allowed_values(self):
        """
        Check if all given allowed values
        are in line with the given type.
        """

        type_string = self.type
        if type_string is None or not self.allowed_values:
            return
        if _TYPE_MAP[type_string] == pd.Series:
            logger.error(
                "The filed allowed_values is not "
                "supported for pd.Series objects. "
                "Equality is not proof-able in a clear way."
                "Going to ignore the setting."
            )
            self.allowed_values = []
            return

        allowed_values_converted = []
        for allowed_value in self.allowed_values:
            value = _convert_value_to_type(value=allowed_value, type_string=type_string)
            if _TYPE_MAP[type_string] in (float, int):
                if (value < self.lb or value > self.ub) and self.clip:
                    raise ValueError(
                        f"Given allowed_value '{value}' is outside of given bounds. "
                        "Set of allowed values is hence infeasible."
                    )
            allowed_values_converted.append(value)
        self.allowed_values = allowed_values_converted

    def set_value(self, value, validate: bool = False):
        """
        Sets the value of the variable. If validate is True (default False), also do
        the following:

        - Convert to the given type
        - Check if inside the list of allowed_values
        - If bounds can be applied, check if inside the given
          bounds and maybe even clip accordingly
        """
        # Unpack given values and convert the value
        if not validate:
            self.value = value
            return

        name_string = self.name
        value = _convert_value_to_type(value=value, type_string=self.type)
        if isinstance(value, (float, int)):
            # Else handle boundaries
            if value > self.ub:
                logger.error(
                    "Given value '%s' is higher than upper bound '%s' for '%s'",
                    value,
                    self.ub,
                    name_string,
                )
                if self.clip:
                    logger.error("Clipping value accordingly to '%s'", self.ub)
                    value = self.ub
            if value < self.lb:
                logger.error(
                    "Given value '%s' is lower than lower bound '%s' for '%s'",
                    value,
                    self.lb,
                    name_string,
                )
                if self.clip:
                    logger.error("Clipping value accordingly to '%s'", self.lb)
                    if isinstance(value, pd.Series):
                        value[value < self.lb] = self.lb
                    else:
                        value = self.lb
        # Check if value is inside allowed values
        if not self.allowed_values or isinstance(value, pd.Series):
            self.value = value
            return

        if value not in self.allowed_values:
            raise ValueError(
                f"Given value for {name_string} is equal to {value} but has "
                f"to be in the set of allowed values: "
                f"{self.allowed_values}"
            )
        self.value = value

    def dict(self, exclude: Container[str] = None) -> dict:
        """Generates a dict from the Variable."""
        slots = slot_helper(self.__class__)
        if not exclude:
            dump = {slot: self.__getattribute__(slot) for slot in slots}
        else:
            dump = {
                slot: self.__getattribute__(slot)
                for slot in slots
                if slot not in exclude
            }
        if isinstance(self.value, pd.Series):
            dump["value"] = self.value.to_dict()
        return dump

    def json(self) -> str:
        """Serializes the Variable in json format and returns a string"""
        dump = self.dict()
        return json.dumps(dump, default=lambda o: o.dict())


BaseVariableT = TypeVar("BaseVariableT", bound=BaseVariable)


def _convert_value_to_type(value: Any, type_string: Optional[str]):
    """Convert the given value to the type of the value"""
    if type_string is None or value is None:
        return value

    type_of_value = _TYPE_MAP[type_string]
    if isinstance(value, type_of_value):
        return value  # If already the type just return
    try:
        # Use the try block to pretty print any error occurring.
        if type_of_value == pd.Series:
            return convert_to_pd_series(value)
        return type_of_value(value)
    except Exception as err:
        raise ValueError(
            f"Given value '{value}' could not be converted "
            f"to the specified type '{type_string}'. Error-message: "
            f"\n{err}"
        )


def convert_to_pd_series(value):
    if isinstance(value, str):
        srs = pd.read_json(StringIO(value), typ="series")
    elif isinstance(value, dict):
        srs = pd.Series(value, dtype=np.float64)
    else:
        raise ValueError(
            f"Specified a variable as a pd.Series, but the given value {value} "
            f"could not be converted. Please pass a json string or a dict."
        )
    if isinstance(srs.index[0], str):
        srs.index = srs.index.astype(float)
    return srs


@define(slots=True, weakref_slot=False, kw_only=True)
class AgentVariable(BaseVariable):
    """
    The basic AgentVariable.
    The AgentVariable is the central messaging piece in the AgentLib. It can hold
    arbitrary (but json-serializable!) values as Agent States, Configuration objects or
     messages.

    In addition to fields defined in BaseVariable,
    any AgentVariable holds the
    - alias: The publicly known name of the variable
    - source: Indicating which agent and or module the variable belong to
    - shared: Whether the variable is going to be shared to other Agents
    - rdf_class: Class in the resource description framework

    Check the description of each field for further information.
    """

    alias: str = field(
        default=None,
        metadata={
            "title": "Alias",
            "description": "Alias, i.e. public name, of the AgentVariable",
        },
    )
    source: Union[Source, str] = field(
        default=Source(),
        metadata={"title": "Place where the variable has been generated"},
        converter=Source.create,
    )
    shared: Optional[bool] = field(
        default=None,
        metadata={
            "title": "shared",
            "description": "Indicates if the variable is going to be shared "
            "with other agents. If False, no external "
            "communication of this variable should take "
            "place in any module.",
        },
    )
    rdf_class: Optional[str] = field(
        default=None,
        metadata={
            "title": "Class in the resource description framework (rdf). "
            "Describes what of (real) object is represented by the"
            "AgentVariable."
        },
    )

    def __attrs_post_init__(self):
        # when creating an AgentVariable not with cls.validate_data(), we still set
        # the alias, but don't validate the value.
        self._check_source()
        self._check_alias()

    def run_validation(self):
        super().run_validation()
        self._check_source()
        self._check_alias()

    def _check_alias(self):
        """Sets the default value for the alias."""
        if self.alias is None:
            self.alias = self.name

    @classmethod
    def from_json(cls, s: Union[str, bytes], validate=False):
        """Instantiates a new AgentVariable from json."""
        data = json.loads(s)
        data.setdefault("name", data["alias"])
        if not validate:
            variable = cls(**data)
            # we do no validation, but we have to at least do type conversion
            variable.value = _convert_value_to_type(variable.value, variable.type)
            return variable
        return cls.validate_data(data)

    def _check_source(self):
        """Convert possible str into source"""
        if isinstance(self.source, str):
            self.source = Source(agent_id=self.source)

    def copy(self, update: Optional[dict] = None, deep: bool = False):
        """Creates a copy of the Variable."""
        _copy = copy(self)
        if deep:
            _copy.value = deepcopy(self.value)
        if update:
            for field, field_value in update.items():
                _copy.__setattr__(field, field_value)
        return _copy

    def dict(self, exclude: Container[str] = None) -> dict:
        result = super().dict(exclude)
        if "source" in result:
            # check needed in case source is in exclude
            result["source"] = result["source"].dict()
        return result


@define(slots=True, weakref_slot=False, kw_only=True)
class BaseModelVariable(BaseVariable):
    """
    Add the causalities used for model specific variables.
    """

    causality: Causality = field(
        default=None,
        metadata={"title": "causality", "description": "The causality of the variable"},
    )
    variability: Variability = field(
        default=None,
        metadata={
            "title": "variability",
            "description": "The variability of the variable",
        },
    )
    type: Optional[str] = field(
        default="float",
        metadata={
            "title": "Type",
            "description": "Name the type of the variable using a string. For model "
            "variables, this is float by default.",
        },
    )

    def __attrs_post_init__(self):
        # for model variables, we always want to validate them, since they are
        # typically not created on the fly
        self.run_validation()

    def check_causality(self):
        """Check if causality equals the default value.
        Else, convert it to the default."""
        default = attrs.fields(type(self)).causality.default
        if default is not None:
            if self.causality != default:
                self.causality = default

    def run_validation(self):
        super().run_validation()
        self.check_causality()
        self.check_fmu_compliance()

    def check_fmu_compliance(self):
        """Check if combination of causality and variability
        is supported according to fmu standard."""

        if self.variability is None:
            if self.causality in [Causality.parameter, Causality.calculatedParameter]:
                self.variability = Variability.tunable
            else:
                self.variability = Variability.continuous
        # Specify allowed combinations and reasons for them being not allowed:
        # Source: FMU Standard
        _reason_a = (
            "The combinations “constant / parameter”, “constant / "
            "calculatedParameter” and “constant / input” do not "
            "make sense, since parameters and inputs "
            "are set from the environment, whereas a constant "
            "has always a value."
        )
        _reason_b = (
            "The combinations “discrete / parameter”, “discrete / "
            "calculatedParameter”, “continuous / parameter”and "
            "continuous / calculatedParameter do not make sense, "
            "since causality = “parameter” and “calculatedParameter” "
            "define variables that do not depend on time, whereas “discrete” "
            "and “continuous” define variables where the values can "
            "change during simulation."
        )
        _reason_c = (
            "For an “independent” variable only variability = “continuous” "
            "makes sense."
        )
        _reason_d = (
            "A fixed or tunable “input” has exactly the same properties "
            "as a fixed or tunable parameter. For simplicity, only"
            " fixed and tunable parameters shall be defined."
        )
        _reason_e = (
            "A fixed or tunable “output” has exactly the same properties "
            "as a fixed or tunable calculatedParameter. For simplicity, "
            "only fixed and tunable calculatedParameters shall be defined"
        )
        _unsupported_combinations = {
            (Causality.parameter, Variability.constant): _reason_a,
            (Causality.parameter, Variability.discrete): _reason_b,
            (Causality.parameter, Variability.continuous): _reason_b,
            (Causality.calculatedParameter, Variability.constant): _reason_a,
            (Causality.calculatedParameter, Variability.discrete): _reason_b,
            (Causality.calculatedParameter, Variability.continuous): _reason_b,
            (Causality.input, Variability.constant): _reason_a,
            (Causality.input, Variability.fixed): _reason_d,
            (Causality.input, Variability.tunable): _reason_d,
            (Causality.output, Variability.fixed): _reason_e,
            (Causality.output, Variability.tunable): _reason_e,
            (Causality.independent, Variability.constant): _reason_c,
            (Causality.independent, Variability.fixed): _reason_c,
            (Causality.independent, Variability.tunable): _reason_c,
            (Causality.independent, Variability.discrete): _reason_c,
        }

        _combination = (self.causality, self.variability)
        # if combination is not supported, raise an error
        assert _combination not in _unsupported_combinations, _unsupported_combinations[
            _combination
        ]


@define(slots=True, weakref_slot=False, kw_only=True)
class ModelVariable(BaseModelVariable):
    """
    The basic ModelVariable.
    Aside from only allowing number for values,
    this class enables calculation with the object itself.
    """

    sim_time: float = field(default=0.0, metadata={"title": "Current simulation time"})

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __pow__(self, power, modulo=None):
        return self.value**power

    def __rpow__(self, other):
        return other**self.value


@define(slots=True, weakref_slot=False, kw_only=True)
class ModelInput(ModelVariable):
    """
    The ModelInput variable.
    Inherits

    - BaseInput: The causality and variability associated with an input
    - ModelVariable: The fields unique to a ModelVariable.
    - BaseModelVariable: All fields associated with any model variable.
    """

    def __attrs_post_init__(self):
        self.causality = Causality.input
        self.variability = Variability.continuous


@define(slots=True, weakref_slot=False, kw_only=True)
class ModelOutput(ModelVariable):
    """
    The ModelOutput variable.
    Inherits

    - BaseOutput: The causality and variability associated with an output
    - ModelVariable: The fields unique to a ModelVariable.
    - BaseModelVariable: All fields associated with any model variable.
    """

    def __attrs_post_init__(self):
        self.causality: Causality = Causality.output
        self.variability: Variability = Variability.continuous


@define(slots=True, weakref_slot=False, kw_only=True)
class ModelState(ModelVariable):
    """
    The ModelState variable.
    Inherits

    - BaseLocal: The causality and variability associated with a local
    - ModelVariable: The fields unique to a ModelVariable.
    - BaseModelVariable: All fields associated with any model variable.
    """

    def __attrs_post_init__(self):
        self.causality: Causality = Causality.local
        self.variability: Variability = Variability.continuous


@define(slots=True, weakref_slot=False, kw_only=True)
class ModelParameter(ModelVariable):
    """
    The ModelParameter variable.
    Inherits

    - BaseParameter: The causality and variability associated with a parameter
    - ModelVariable: The fields unique to a ModelVariable.
    """

    def __attrs_post_init__(self):
        self.causality: Causality = Causality.parameter
        self.variability: Variability = Variability.tunable


# Types section
# Agents
AgentVariables = List[AgentVariable]
# Models
ModelInputs = List[ModelInput]
ModelStates = List[ModelState]
ModelOutputs = List[ModelOutput]
ModelParameters = List[ModelParameter]
