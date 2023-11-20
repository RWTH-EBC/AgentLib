import pydantic

import agentlib


class NewModuleConf2(agentlib.BaseModuleConfig):
    my_property2: str = pydantic.Field(default="wow a plugin with two modules")


class NewModule2(agentlib.BaseModule):
    def process(self):
        while True:
            yield self.env.timeout(5)
            print("Hello I am NewModule2.")

    def register_callbacks(self):
        pass

    config: NewModuleConf2
