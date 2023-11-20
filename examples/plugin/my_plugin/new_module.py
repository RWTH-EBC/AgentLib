import pydantic

import agentlib


class NewModuleConf(agentlib.BaseModuleConfig):
    my_property: str = pydantic.Field(default="wow a plugin")


class NewModule(agentlib.BaseModule):
    def process(self):
        while True:
            yield self.env.timeout(10)
            print("Hello I am NewModule.")

    def register_callbacks(self):
        pass

    config: NewModuleConf
