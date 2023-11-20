"""
Module containing a basic Broker which
may be inherited by specialized classes.
"""

import threading


class Singleton(type):
    """Global singleton to ensure only one broker exists"""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Broker(metaclass=Singleton):
    """Base Broker class"""

    def __init__(self):
        self.lock = threading.Lock()
        self._clients = set()

    def register_client(self, client):
        """Append the given client to the list
        of clients."""
        with self.lock:
            self._clients.add(client)

    def delete_client(self, client):
        """Delete the given client from the list of clients"""
        with self.lock:
            try:
                self._clients.remove(client)
            except KeyError:
                pass

    def delete_all_clients(self):
        """Delete all clients from the list of clients"""
        with self.lock:
            for client in list(self._clients):
                try:
                    self._clients.remove(client)
                except KeyError:
                    pass

    def __repr__(self):
        """Overwrite build-in function."""
        return f"{self.__class__.__name__} with registered agents: {self._clients}"
