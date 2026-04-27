import base64
import sys
from datetime import datetime
from pathlib import Path

import dotenv
import httpx
import typer
from agentlib.utils.ds_utils.common import BaseModel
from pydantic import ConfigDict, HttpUrl, SecretStr
from pydantic_settings import BaseSettings
from agentlib.utils.ds_utils.settings import GetConnectorSettings


BASE_PATH = Path(__file__).parent


class UnauthorizedError(RuntimeError):
    pass


class NoTokenError(RuntimeError):
    pass


class OnenetClient(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    auth_token: SecretStr | None = None
    connector_settings: BaseSettings

    def extended_url(self, ext: str):
        """
        Generate an extended URL by appending a specified path to the connector URL.

        Parameters
        ----------
        ext : str
            The extension/path to append to the connector URL.

        Returns
        -------
        str
            The complete extended URL.
        """
        ext = ext.removeprefix("/")
        if self.connector_settings.url.path is None:
            return f"{self.connector_settings.url}/{ext}"
        return f"{str(self.connector_settings.url).removesuffix('/')}/{ext}"

    def authenticate(self):
        """
        Authenticate a user and obtain an access token.

        Parameters
        ----------
        user : str
            The username for authentication.

        password : str
            The password for authentication.

        Raises
        ------
        UnauthorizedError
            If the authentication fails.
        """
        # TODO convert connector url to httpx url
        res = httpx.post(url=self.extended_url("user/auth"), json={"username": self.connector_settings.username, "password": self.connector_settings.password.get_secret_value()})
        if res.status_code == 200:
            self.auth_token = res.json()["accessToken"]
        else:
            raise UnauthorizedError()

    def _check_token(self):
        """
         Check if the authentication token is set. Raises an error if not.

        Raises
        ------
        NoTokenError
            If no token has been set for authentication.
        """
        if self.auth_token is None:
            self.authenticate()

    def list_available_data(self):
        """
        Retrieve a list of available data after checking the authentication token.

        Returns
        -------
        dict
            A JSON response containing available data.
        """
        self._check_token()
        # TODO convert connector url to httpx url
        res = httpx.get(
            self.extended_url("consume-data/list"),
            headers={f"Authorization": f"Bearer {self.auth_token.get_secret_value()}"},
        )
        return res.json()

    def get_data(self, id: str) -> str:
        """
        Retrieve specific data by its ID after validating the authentication token.

        Parameters
        ----------
        id : str
            The identifier of the data to retrieve.

        Returns
        -------
        str
            The file data retrieved from the server.

        Raises
        ------
        AssertionError
            If the retrieval status is not successful.
        """
        self._check_token()
        res = httpx.get(
            self.extended_url(f"consume-data/{id}"),
            headers={f"Authorization": f"Bearer {self.auth_token.get_secret_value()}"},
        )
        res_json = res.json()
        assert res_json["retrieved"]
        return res_json["filedata"]

    def list_service(self):
        """
        List all offered services after checking the authentication token.

        Returns
        -------
        dict
            A JSON response containing offered services.

        Note
        -----
            This currently selects only the first service from the response.
        """
        self._check_token()
        res = httpx.get(
            self.extended_url(f"offered-services/list"),
            headers={f"Authorization": f"Bearer {self.auth_token.get_secret_value()}"},
        )
        return res.json()

    def list_offered_data(self):
        """
        List all offered data after checking the authentication token.

        Returns
        -------
        dict
            A JSON response containing offered data.
        """
        self._check_token()
        res = httpx.get(
            self.extended_url("provide-data/list"),
            headers={f"Authorization": f"Bearer {self.auth_token.get_secret_value()}"},
        )
        return res.json()

    def post_data(
        self,
        title: str,
        file_name: str,
        code: str,
        data_offering_id: str,
        file_content: str | bytes,
        file_size: int,
    ):
        """
        Upload new data along with metadata and file content after checking the authentication token.

        Parameters
        ----------
        title : str
            Title of the data being uploaded.

        file_name : str
            Name of the file being uploaded.

        code : str
            Code associated with this upload.

        data_offering_id : str
            Identifier for the data offering (may be optional).

        file_content : Union[str, bytes]
            Content of the file being uploaded as string or bytes.

        file_size : int
            Size of the file being uploaded in bytes.

        Returns
        -------
        dict
            A JSON response confirming upload status and details.
        """

        self._check_token()
        body = {
            "title": title,  # get from data_list
            "description": "just some echoed data for testing",
            "filename": file_name,
            "fileSize": file_size,
            "data_offering_id": data_offering_id,  # Probably don't need it
            "code": code,  # get from data_list
        }

        if isinstance(file_content, str):
            file_content = file_content.encode()

        body["file"] = f"data:text/plain;base64,{base64.b64encode(file_content).decode()}"

        if file_size is not None:
            body["fileSize"] = str(file_size)

        res = httpx.post(
            self.extended_url(f"provide-data"),
            json=body,
            headers={f"Authorization": f"Bearer {self.auth_token.get_secret_value()}"},
        )

        return res.json()

    def delete_file(self, id: str):
        """
        Delete a specified file by its ID after validating the authentication token.

        Parameters
        ----------
        id : str
                The identifier of the file to delete.

        Returns
        -------
        bool
                True if deletion was successful; otherwise False.
        """

        self._check_token()
        res = httpx.delete(
            self.extended_url(f"provide-data/{id}"),
            headers={f"Authorization": f"Bearer {self.auth_token.get_secret_value()}"},
        )

        if res.status_code == 200:
            return True
        print(res.json())
        return False






app = typer.Typer()

@app.command("provide")
@app.command("upload", hidden=True, deprecated=True)
def upload(file: Path, service_id: str, category_code: str):
    """Upload a FILE to the onenet-connector

    Parameters
    ----------
    file : Path
        Filepath to which to save the file to. Needs to include the filename (subject to change)
    service_id : str
        ID of the service the file should be uploaded to
    category_code: str
        String identifying the type of document uploaded (defined by the connector standard)
    """
    with open(file, "rb") as infile:
        data = infile.read()
    global onenet_client
    onenet_client.authenticate()
    file_size = file.stat().st_size
    res = onenet_client.post_data(
        title=file.name.rsplit(".", 1)[0],
        file_name=file.name,
        code=category_code,
        data_offering_id=service_id,
        file_content=data,
        file_size=file_size,
    )

@app.command("consume")
@app.command("download", hidden=True, deprecated=True)
def download(id: str, file: Path):
    """Download a file from the onenet-connector

    Parameters
    ----------
    id : str
        ID of the file to be downloaded
    file : Path
        Filepath to which to save the file to. Needs to include the filename (subject to change)
    """
    global onenet_client
    onenet_client.authenticate()
    data: str = onenet_client.get_data(id)
    with open(file, "wb") as outfile:
        outfile.write(base64.b64decode(data))
    print(f"File saved under {file}")

@app.command()
def delete(id: str):
    """Delete a file from the onenet-connector

    Parameters
    ----------
    id : str
        ID of the file to be deleted
    """
    global onenet_client
    onenet_client.authenticate()
    was_deleted = onenet_client.delete_file(id)
    print(f"Entry was delete: {was_deleted}")

if __name__ == "__main__":
    onenet_client = OnenetClient(connector_settings=GetConnectorSettings("sim"))

    ENV_PATH = BASE_PATH / ".env_sim"
    dotenv.load_dotenv(ENV_PATH)
    app()

