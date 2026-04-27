from pydantic import BaseModel as PydBaseModel, HttpUrl


class BaseModel(PydBaseModel):
    """Custom BaseModel allowing global configuration"""

    pass


def sub_path(url: HttpUrl, ext: str):
    ext = ext.removeprefix("/")
    if url.path is None:
        return f"{url}/{ext}"
    return f"{str(url).removesuffix('/')}/{ext}"
