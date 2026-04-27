import json
import base64
import sys
from pydantic import Field

from agentlib.core.datamodels import AgentVariable
from agentlib.modules.communicator.communicator import (
    CommunicationDict,
    LocalCommunicator,
    LocalCommunicatorConfig,
)
from agentlib.utils.ds_utils.ds_onenet_client import OnenetClient
from agentlib.utils.ds_utils.settings import GetConnectorSettings
from agentlib.utils.ds_utils.ds_broker import DataSpaceBroker


class DataSpaceClientConfig(LocalCommunicatorConfig):
    kind: str = Field(
        title="Indicates which data space connector is to be used - the one for the MPC side or the one for the simulation side",
        default=""
    )
    service_id: str = Field(
        title="Service ID used by the Agent's Data Space connector. Must be set up in TwinEU UI first",
        default=""
    )


class DataSpaceClient(LocalCommunicator):
    """
    This communicator implements the communication between agents via a
    broadcast broker central process.
    Note: The broker is implemented as singleton. This means that all agents must
    be in the same process!
    """

    broker: DataSpaceBroker
    config: DataSpaceClientConfig
    client: OnenetClient

    def setup_broker(self):
        self.logger.debug(f"SETTING UP BROKER WITH CONFIG: {self.config}")
        self.client = OnenetClient(connector_settings=GetConnectorSettings(kind=self.config.kind))
        
        """Use the LocalBroadcastBroker"""
        return DataSpaceBroker()

    def upload_json(self, title: str, data:bytes, service_id: str, category_code: str):
        """Upload JSON data to the onenet-connector

        Parameters
        ----------
        file : Path
            Filepath to which to save the file to. Needs to include the filename (subject to change)
        service_id : str
            ID of the service the file should be uploaded to
        category_code: str
            String identifying the type of document uploaded (defined by the connector standard)
        """
        #self.client.authenticate()
        file_size = sys.getsizeof(data)
        return self.client.post_data(
            title=title,
            file_name=f"{title}.json",
            code=category_code,
            data_offering_id=service_id,
            file_content=data,
            file_size=file_size,
        )

    def download_json(self, id: str):
        """Download a file from the onenet-connector and return data as json

        Parameters
        ----------
        id : str
            ID of the file to be downloaded
        """
        
        #self.client.authenticate()
        data = self.client.get_data(id)
        return self.to_json(data)

    def _send(self, payload: CommunicationDict):
        # upload message to DS Connector
        res = self.upload_json(self.config.kind, self.to_json(payload), self.config.service_id, "00")
        self.logger.debug(f"uploaded{self.to_json(payload)} with result: {res}")
    
        # then make all other agents receive it
        self.broker.broadcast(payload["source"], res['id'])
        
    def _receive(self, msg_id):
        # download message from DS Connector and return super.receive
        msg = self.download_json(msg_id)

        msg_obj = base64.b64decode(msg).decode("utf-8")
        self.logger.debug(f"downloaded message with id {msg_id}: {msg_obj}")
        
        return super()._receive(msg_obj)
    

