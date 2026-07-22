#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
from typing import Any, Optional, Callable, Set, Dict, List, Tuple, Any
from typing_extensions import Protocol
from collections import deque

from agentlace.zmq_wrapper.req_rep import ReqRepServer, ReqRepClient
from agentlace.zmq_wrapper.broadcast import BroadcastServer, BroadcastClient
from agentlace.zmq_wrapper.pipeline import Consumer, Producer
from agentlace.internal.utils import compute_hash
from agentlace.data.data_store import DataStoreBase

import time
import threading
import logging
import json
from dataclasses import dataclass, asdict, field


##############################################################################


@dataclass
class TrainerConfig():
    """
    Configuration for the edge server and client
    NOTE: Client and server should have the same config
          client will raise Error if configs are different, thus can use
          `version` to check for compatibility
    """
    port_number: int = 5555
    broadcast_port: int = 5556
    request_types: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None
    version: str = "0.0.2"
    experimental_pipeline_port: Optional[str] = None


class DataCallback(Protocol):
    """
    Define the data type of the data callback function
        :param store_name: Name of the data store
        :param payload: Payload to send to the trainer, e.g. training data
    """

    def __call__(self, store_name, payload: dict):
        ...


class RequestCallback(Protocol):
    """
    Define the data type of the request callback function
        :param type: Name of the custom request
        :param payload: Payload to send to the trainer, e.g. training data
        :return: Response to send to the client, can return updated network.
    """

    def __call__(self, type: str, payload: dict) -> dict:
        ...


##############################################################################


class TrainerServer:
    def __init__(self,
                 config: TrainerConfig,
                 data_callback: Optional[DataCallback] = None,
                 request_callback: Optional[RequestCallback] = None,
                 log_level=logging.INFO):
        """
        Args
            :param config: Config object
            :param data_callback: Callback fn when a new data arrives
            :param request_callback: Callback fn to impl custom requests
        """
        self.queue = deque()  # FIFO queue
        self.request_types = set(config.request_types)  # faster lookup
        self.data_stores = {}  # map of {ds_name: DataStoreBase}
        self.last_update_id_map = {}  # map of {ds_name: last_update_id}
        self.config = config
        self.data_callback = data_callback

        ############################################################################
        def __callback_impl(data: dict) -> dict:
            """Callback impl when a request is received from the client"""
            _type, _payload = data.get("type"), data.get("payload")

            # insert data to the data store
            if _type == "datastore":
                store_name = data.get("store_name")
                last_update_id = _payload.get("last_id", -1)
                if store_name not in self.data_stores:
                    return {"success": False, "message": "Invalid datastore name"}
                batch_data = _payload.get("data", [])
                print(f"[diag-server] received {len(batch_data)} transitions for '{store_name}', buffer_size={len(self.data_stores[store_name])}")
                self.data_stores[store_name].batch_insert(batch_data)
                print(f"[diag-server] after insert buffer_size={len(self.data_stores[store_name])}")
                if data_callback:
                    data_callback(store_name, _payload)
                self.last_update_id_map[store_name] = last_update_id
                return {"success": True}

            # get last update id of the data store
            elif _type == "get_last_update_id":
                store_name = _payload.get("store_name")
                if store_name not in self.data_stores or store_name not in self.last_update_id_map:
                    return {"success": False, "message": "Invalid datastore name"}
                return {"success": True, "payload": self.last_update_id_map.get(store_name, -1)}

            # get the config of the server, client can check compatibility
            elif _type == "hash":
                config_json = json.dumps(asdict(config), separators=(',', ':'))
                return {"success": True, "payload": config_json}

            # custom request
            elif _type in self.request_types:
                return request_callback(_type, _payload) if request_callback else {}

            return {"success": False, "message": "Invalid type or payload"}
        ############################################################################

        self.req_rep_server = ReqRepServer(
            config.port_number, __callback_impl, log_level=log_level)
        self.broadcast_server = BroadcastServer(
            config.broadcast_port, log_level=log_level)

        # NOTE: experimental feature to use pipeline for data update
        if config.experimental_pipeline_port:
            def _pipe_callback_impl(data: Any):
                # NOTE insert data to the data store similar as above
                _payload = data.get("payload")
                store_name = data.get("store_name")
                last_update_id = _payload.get("last_id", -1)
                if store_name not in self.data_stores:
                    return
                batch_data = _payload.get("data", [])
                self.data_stores[store_name].batch_insert(batch_data)
                if self.data_callback:
                    self.data_callback(store_name, _payload)
                self.last_update_id_map[store_name] = last_update_id
                return
            self.consumer = Consumer(
                _pipe_callback_impl, config.experimental_pipeline_port)
            self.consumer.async_start()

        logging.basicConfig(level=log_level)
        logging.debug(
            f"Trainer server is listening on port {config.port_number}")

    def register_data_store(self, name, data_store: DataStoreBase):
        """
        Register a datastore to the server
            :param name: Name of the data store
            :param data_store: Datastore object
        """
        self.data_stores[name] = data_store
        self.last_update_id_map[name] = -1

    def data_store(self, name) -> Optional[DataStoreBase]:
        """
        Get the datastore reference from the server
            :param name: Name of the data store
            :return: Datastore object
        """
        return self.data_stores.get(name)

    def store_names(self) -> Set[str]:
        """
        Returns the set of table names available on the server
            :return: Set of table names available on the server
        """
        return set(self.data_stores.keys())

    def publish_network(self, payload: dict):
        """
        Publishes the network to the broadcast server.
        """
        self.broadcast_server.broadcast(payload)

    def start(self, threaded: bool = False):
        """
        Starts the server, defaulting to blocking mode
            :param threaded: Whether to start the server in a separate thread
        """
        if threaded:
            self.thread = threading.Thread(
                target=self.req_rep_server.run, daemon=True)
            self.thread.start()
        else:
            self.req_rep_server.run()

    def stop(self):
        """Stop the server"""
        self.req_rep_server.stop()


##############################################################################


class TrainerClient:
    def __init__(self,
                 name: str,
                 server_ip: str,
                 config: TrainerConfig,
                 data_store: DataStoreBase = None,
                 data_stores: Dict[str, DataStoreBase] = {},
                 log_level=logging.INFO,
                 wait_for_server: bool = False,
                 timeout_ms: float = 800,
                 ):
        """
        Args:
            :param name: Name of the client
            :param server_ip: IP address of the server
            :param config: Config object
            :param data_store: Datastore to store the data (deprecated soon)
            :param data_stores: Map of data stores (str id -> DataStoreBase)
            :param log_level: Logging level
            :param wait_for_server: Whether to wait for the server to start
            :param timeout_ms: Request Timeout in milliseconds
        Note:
            name, and data_store will deprecate in future versions, to support
            multiple data stores in single client, use `data_stores_map` instead.
        """
        self.client_name = name
        self.req_rep_client = ReqRepClient(
            server_ip, config.port_number, log_level=log_level, timeout_ms=timeout_ms)
        self.broadcast_client = None
        self.server_ip = server_ip
        self.config = config
        self.request_types = set(config.request_types)  # faster lookup
        self.data_store = data_store
        self.update_thread = None
        self.last_request_time = 0
        self.log_level = log_level

        # Supporting multiple data stores
        self.data_stores_map = data_stores  # dict of str -> DataStoreBase

        logging.basicConfig(level=log_level)
        res = self.req_rep_client.send_msg({"type": "hash"})

        # If wait for server
        if wait_for_server:
            while res is None:
                logging.warning("Failed to connect to server, retrying...")
                time.sleep(2)
                res = self.req_rep_client.send_msg({"type": "hash"})

        if res is None:
            raise Exception("Failed to connect to server")

        # try check configuration compatibility
        config_json = json.dumps(asdict(config), separators=(',', ':'))
        if compute_hash(config_json) != compute_hash(res["payload"]):
            raise Exception(
                f"Incompatible config with hash with server. "
                "Please check the config of the server and client")

        # NOTE: experimental feature to use pipeline for data update
        if config.experimental_pipeline_port:
            self.producer = Producer(
                ip=server_ip,
                port=config.experimental_pipeline_port
            )

        # First update the server's datastore
        res = self.update()
        if not res:
            logging.error("Failed to get res when update server datastore")
        logging.debug(
            f"Initiated trainer client at {server_ip}:{config.port_number}")

    def update(self) -> bool:
        """
        This will explicity trigger an update to the data store
        Args:
            :return True if the update is successful, False otherwise
        """
        # if single data store is used
        if self.data_store is not None:
            from_id = self.get_server_last_update_id(self.client_name)
            if from_id is None:
                return False
            return self.update_datastore(self.client_name, from_id)

        # if multiple data stores is used
        elif self.data_stores_map:
            for name, data_store in self.data_stores_map.items():
                print(f"[diag-agentlace] update() get_server_last_update_id({name})")
                from_id = self.get_server_last_update_id(name)
                print(f"[diag-agentlace] update() from_id={from_id} for {name}")
                if from_id is None:
                    return False
                n = len(data_store)
                print(f"[diag-agentlace] update() calling update_datastore({name}, {from_id}), store_size={n}")
                self.update_datastore(name, from_id)
                print(f"[diag-agentlace] update() update_datastore({name}) done")
            return True
        return False

    def update_datastore(
        self,
        name: str,
        from_id: int,
        confirm_update=False,
    ) -> bool:
        """
        This provide the api for user to explicitly update the data store
        with the server, with more options
        Args:
            :param name: Name of the data store
            :param from_id: Last update id of the client
            :param confirm_update: Whether to confirm the update with the server
            :return: True if the update is successful, False otherwise
        """
        # for backward compatibility
        if self.data_store is not None:
            _data_store = self.data_store
        elif self.data_stores_map:
            _data_store = self.data_stores_map.get(name)
            if _data_store is None:
                logging.error(f"Datastore {name} not found")
                return False
        else:
            logging.error("Datastore not defined")
            return False

        client_latest_id = _data_store.latest_data_id()
        batch_data = _data_store.get_latest_data(from_id)
        if len(batch_data) == 0:
            return True

        data_dict = {"data": batch_data, "last_id": client_latest_id}
        res = self._update_ds(name, data_dict)

        # with confirm_update, check if the server has updated the data store
        if confirm_update:
            server_last_id = self.get_server_last_update_id(name)
            if server_last_id is None:
                return False
            return True if server_last_id == client_latest_id else False

        if res is None or not res["success"]:
            logging.warning("Failed to get res when update server datastore")
            return False
        return True

    def get_server_last_update_id(self, name: str) -> Optional[int]:
        """
        Get the last update id of the server's data store
        Args:
            :param name: Name of the data store
            :return: Last update id of the server's data store
        """
        msg = {"type": "get_last_update_id", "payload": {"store_name": name}}
        res = self.req_rep_client.send_msg(msg)
        if res is None or not res["success"]:
            logging.warning("Failed to get last update id")
            return None
        return res["payload"]

    def request(self, type: str, payload: dict) -> Optional[dict]:
        """
        Call the trainer server with custom requests
        Some common use cases: checkpointing, get-stats, etc.
            :param type: Name of the custom request, defined in `config.request_types`
            :param payload: Payload to send to the trainer
            :return: Response from the trainer, return None if timeout
        """
        if type not in self.request_types:
            return None
        msg = {"type": type, "payload": payload}
        if self.config.rate_limit and \
                time.time() - self.last_request_time < 1 / self.config.rate_limit:
            logging.warning("Rate limit exceeded")
            return None

        self.last_request_time = time.time()
        return self.req_rep_client.send_msg(msg)

    def recv_network_callback(self, callback: Callable[[dict], None]):
        """Registers the callback function for receiving an updated network"""
        self.broadcast_client = BroadcastClient(
            self.server_ip, self.config.broadcast_port, log_level=self.log_level)
        self.broadcast_client.async_start(callback)

    def start_async_update(self, interval: int = 10):
        """
        Start a thread that server's datastore every `interval` seconds.
        :param interval: Interval in seconds.
        """
        def _periodic_update():
            while not self.stop_update_flag.is_set():
                self.update()
                time.sleep(interval)

        self.stop_update_flag = threading.Event()
        if self.update_thread is None or not self.update_thread.is_alive():
            self.stop_update_flag.clear()
            self.update_thread = threading.Thread(
                target=_periodic_update, daemon=True)
            self.update_thread.start()

    def stop(self):
        """Stop the client"""
        if self.broadcast_client:
            self.broadcast_client.stop()
        if self.update_thread and self.update_thread.is_alive():
            self.stop_update_flag.set()
            self.update_thread.join()

        logging.debug("Stopped trainer client")

    def _update_ds(self, name: str, data: dict) -> Optional[dict]:
        """
        internal update method that will send the data to the server
            :param name Name of the data store
            :param data: Payload to send to the trainer
            :return: Response from the trainer, return None if timeout
        """
        msg = {"type": "datastore", "store_name": name, "payload": data}

        # NOTE: experimental feature to use pipeline for data update
        if self.config.experimental_pipeline_port:
            self.producer.send_msg(msg)
            return {"success": True}

        if self.config.rate_limit and \
                time.time() - self.last_request_time < 1 / self.config.rate_limit:
            logging.warning("Rate limit exceeded")
            return None
        self.last_request_time = time.time()
        self.req_rep_client.send_msg(msg)
        return {"success": True}

##############################################################################


class TrainerSMInterface:
    """
    Utilized shared-memory to recreate the transport layer interface
    of TrainerServer and TrainerClient. Helpful for single-process
    multithreaded operation, while maintaining the same interface as
    the TrainerServer and TrainerClient.
        NOTE: this suppose to be an experimental feature
    """

    def __init__(self):
        self._recv_network_fn = None
        self._req_callback_fn = None

    def recv_network_callback(self, callback_fn: Callable):
        """Refer to TrainerClient.recv_network_callback()"""
        self._recv_network_fn = callback_fn

    def publish_network(self, params: dict):
        """Refer to TrainerClient.recv_network_callback()"""
        if self._recv_network_fn:
            self._recv_network_fn(params)

    def start(self, *args, **kwargs):
        pass  # Do nothing

    def stop(self):
        pass  # Do nothing

    def update(self):
        """Refer to TrainerClient.update()"""
        pass  # Do nothing since assume shared datastore

    def register_request_callback(self, callback_fn: Callable):
        """A impl within TrainerServer.init()"""
        self._req_callback_fn = callback_fn

    def request(self, type: str, payload: dict) -> Optional[dict]:
        """Refer to TrainerClient.request()"""
        if self._req_callback_fn:
            return self._req_callback_fn(type, payload)
        return None

    def start_async_update(self, interval: int = 10):
        """
        Refer to TrainerClient.start_async_update()
        no impl needed since assume shared datastore
        """
        pass

