import abc
import time
from functools import cached_property
from typing import Union, List, Optional

from pydantic import AnyUrl, Field, ValidationError, field_validator

from agentlib.modules.communicator.communicator import (
    Communicator,
    SubscriptionCommunicatorConfig,
)
from agentlib.core import Agent
from agentlib.core.datamodels import AgentVariable
from agentlib.core.errors import InitializationError
from agentlib.utils.validators import convert_to_list
from agentlib.core.errors import OptionalDependencyError

try:
    from paho.mqtt.client import (
        Client as PahoMQTTClient,
        MQTTv5,
        MQTT_CLEAN_START_FIRST_ONLY,
        MQTT_LOG_ERR,
        MQTT_LOG_WARNING,
    )
    from paho.mqtt.enums import CallbackAPIVersion
except ImportError as err:
    raise OptionalDependencyError(
        dependency_name="mqtt",
        dependency_install="paho-mqtt",
        used_object="Module type 'mqtt'",
    ) from err


class BaseMQTTClientConfig(SubscriptionCommunicatorConfig):
    keepalive: int = Field(
        default=60,
        description="Maximum period in seconds between "
        "communications with the broker. "
        "If no other messages are being "
        "exchanged, this controls the "
        "rate at which the client will "
        "send ping messages to the "
        "broker.",
    )
    clean_start: bool = Field(
        default=True,
        description="True, False or "
        "MQTT_CLEAN_START_FIRST_ONLY."
        "Sets the MQTT v5.0 clean_start "
        "flag always, never or on the "
        "first successful connect "
        "only, respectively.  "
        "MQTT session data (such as "
        "outstanding messages and "
        "subscriptions) is cleared "
        "on successful connect when "
        "the clean_start flag is set.",
    )
    subtopics: Union[List[str], str] = Field(
        default=[], description="Topics to that the agent subscribes"
    )
    prefix: str = Field(default="/agentlib", description="Prefix for MQTT-Topic")
    qos: int = Field(default=0, description="Quality of Service", ge=0, le=2)
    connection_timeout: float = Field(
        default=10,
        description="Number of seconds to wait for the initial connection "
        "until throwing an Error.",
    )
    username: Optional[str] = Field(default=None, title="Username to login")
    password: Optional[str] = Field(default=None, title="Password to login")
    use_tls: Optional[bool] = Field(
        default=None, description="Option to use TLS certificates"
    )
    tls_ca_certs: Optional[str] = Field(
        default=None,
        description="Path to the Certificate Authority certificate files. "
        "If None, windows certificate will be used.",
    )
    client_id: Optional[str] = Field(default=None, title="Client ID")

    # Add validator
    check_subtopics = field_validator("subtopics")(convert_to_list)


class MQTTClientConfig(BaseMQTTClientConfig):
    url: AnyUrl = Field(
        title="Host",
        description="Host is the hostname or IP address " "of the remote broker.",
    )

    @field_validator("url")
    @classmethod
    def check_url(cls, url):
        if url.scheme in ["mqtts", "mqtt"]:
            return url
        if url.scheme is None:
            url.scheme = "mqtt"
            return url
        raise ValidationError


class BaseMqttClient(Communicator):
    # We use the paho-mqtt module and are
    # thus required to use their function signatures and function names
    # pylint: disable=unused-argument,too-many-arguments,invalid-name
    config: BaseMQTTClientConfig
    mqttc_type = PahoMQTTClient

    def _log_all(self, client, userdata, level, buf):
        """
        client:     the client instance for this callback
        userdata:   the private user data as set in Client() or userdata_set()
        level:      gives the severity of the message and will be one of
                    MQTT_LOG_INFO, MQTT_LOG_NOTICE, MQTT_LOG_WARNING,
                    MQTT_LOG_ERR, and MQTT_LOG_DEBUG.
        buf:        the message itself
        Args:
            *args:

        Returns:

        """
        if level == MQTT_LOG_ERR or level == MQTT_LOG_WARNING:
            self.logger.error("ERROR OR WARNING: %s", buf)

    def __init__(self, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        self._subcribed_topics = 0
        self._mqttc = self.mqttc_type(
            client_id=self.config.client_id or str(self.source),
            protocol=MQTTv5,
            callback_api_version=CallbackAPIVersion.VERSION2,
        )
        if self.config.username is not None:
            self.logger.debug("Setting password and username")
            self._mqttc.username_pw_set(
                username=self.config.username, password=self.config.password
            )
            #  Add TLS-Settings (default behavior)
            if self.config.use_tls is None:
                self._mqttc.tls_set(ca_certs=self.config.tls_ca_certs)
        # Add TLS-Settings
        if self.config.use_tls:
            self._mqttc.tls_set(ca_certs=self.config.tls_ca_certs)

        self._mqttc.on_connect = self._connect_callback
        self._mqttc.on_disconnect = self._disconnect_callback
        self._mqttc.on_message = self._message_callback
        self._mqttc.on_subscribe = self._subscribe_callback
        self._mqttc.on_log = self._log_all
        self._mqttc.loop_start()

        self.connect()

        self.logger.info(
            "Agent %s waits for mqtt connections to be ready ...", self.agent.id
        )
        started_wait = time.time()
        while True:
            if (
                self._mqttc.is_connected()
                and self._subcribed_topics == self.topics_size
            ):
                break
            if time.time() - started_wait > self.config.connection_timeout:
                raise InitializationError("Could not connect to MQTT broker.")

        self.logger.info("Module is fully connected")

    @abc.abstractmethod
    def connect(self):
        raise NotImplementedError

    def terminate(self):
        """Disconnect from client and join loop"""
        self.disconnect()
        super().terminate()

    # The callback for when the client receives a CONNACK response from the server.
    def _connect_callback(self, client, userdata, flags, reasonCode, properties):
        if reasonCode != 0:
            err_msg = f"Connection failed with error code: '{reasonCode}'"
            self.logger.error(err_msg)
            raise ConnectionError(err_msg)
        self.logger.debug("Connected with result code: '%s'", reasonCode)

    def disconnect(self, reasoncode=None, properties=None):
        """Trigger the disconnect"""
        self._mqttc.disconnect(reasoncode=reasoncode, properties=properties)

    def _disconnect_callback(self, client, userdata, reasonCode, properties):
        """Stop the loop as a result of the disconnect"""
        self.logger.warning(
            "Disconnected with result code: %s | userdata: %s | properties: %s",
            reasonCode,
            userdata,
            properties,
        )
        self.logger.info("Active: %s", self._mqttc.is_connected())

    def _message_callback(self, client, userdata, msg):
        """
        The default callback for when a PUBLISH message is
        received from the server.
        """
        agent_inp = AgentVariable.from_json(msg.payload)
        self.logger.debug(
            "Received variable %s = %s from source %s",
            agent_inp.alias,
            agent_inp.value,
            agent_inp.source,
        )
        self.agent.data_broker.send_variable(agent_inp)

    def _subscribe_callback(self, client, userdata, mid, reasonCodes, properties):
        """Log if the subscription was successful"""
        for reason_code in reasonCodes:
            if reason_code == self.config.qos:
                self._subcribed_topics += 1
                self.logger.info(
                    "Subscribed to topic %s/%s",
                    self._subcribed_topics,
                    self.topics_size,
                )
            else:
                msg = f"{self.agent.id}'s subscription failed: {reason_code}"
                self.logger.error(msg)
                raise ConnectionError(msg)

    @property
    def topics_size(self):
        return len(self.config.subtopics) + len(self.config.subscriptions)


class MqttClient(BaseMqttClient):
    config: MQTTClientConfig

    @cached_property
    def pubtopic(self):
        return self.generate_topic(agent_id=self.agent.id, subscription=False)

    @property
    def topics_size(self):
        return len(self._get_all_topics())

    def generate_topic(self, agent_id: str, subscription: bool = True):
        """
        Generate the topic with the given agent_id and
        configs prefix
        """
        if subscription:
            topic = "/".join([self.config.prefix, agent_id, "#"])
        else:
            topic = "/".join([self.config.prefix, agent_id])
        topic.replace("//", "/")
        return topic

    def connect(self):
        port = self.config.url.port
        if port is None:
            port = 1883
        else:
            port = int(port)
        self._mqttc.connect(
            host=self.config.url.host,
            port=port,
            keepalive=self.config.keepalive,
            bind_address="",
            bind_port=0,
            clean_start=MQTT_CLEAN_START_FIRST_ONLY,
            properties=None,
        )

    def _get_all_topics(self):
        """
        Helper function to return all topics the client
        should listen to.
        """
        topics = set()
        for subscription in self.config.subscriptions:
            topics.add(self.generate_topic(agent_id=subscription))
        topics.update(set(self.config.subtopics))
        return topics

    def _connect_callback(self, client, userdata, flags, reasonCode, properties):
        super()._connect_callback(
            client=client,
            userdata=userdata,
            flags=flags,
            reasonCode=reasonCode,
            properties=properties,
        )
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        self._subcribed_topics = 0  # Reset counter as well

        for topic in self._get_all_topics():
            client.subscribe(topic=topic, qos=self.config.qos)
            self.logger.info("Subscribes to: '%s'", topic)

    def _send(self, payload: dict):
        """Publish the given output"""
        topic = "/".join([self.pubtopic, payload["alias"]])
        self._mqttc.publish(
            topic=topic,
            payload=self.to_json(payload),
            qos=self.config.qos,
            retain=False,
            properties=None,
        )
