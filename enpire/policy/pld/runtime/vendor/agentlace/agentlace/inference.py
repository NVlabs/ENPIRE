#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
from typing import Optional, Callable, Set, Dict, List

from agentlace.zmq_wrapper.req_rep import ReqRepServer, ReqRepClient
from agentlace.internal.utils import compute_hash

import threading

##############################################################################


class InferenceServer:
    def __init__(self, port_num: int):
        """
        Create a server with the given port number and interface names
        """
        def __parser_cb(payload: dict) -> dict:
            # if is list_interfaces type
            if payload.get('type') == 'list_interfaces':
                return {"interfaces": list(self.interfaces.keys())}
            elif payload.get('type') == 'call_interface':
                interface_name = payload.get('interface')
                if interface_name and interface_name in self.interfaces:
                    return self.interfaces[interface_name](payload['payload'])
            return {"success": False, "message": "Invalid interface or payload"}

        self.server = ReqRepServer(port_num, __parser_cb)
        self.interfaces = {}

    def start(self, threaded: bool = False):
        """
        Starts the server, defaulting to blocking mode
            :param threaded: Whether to start the server in a separate thread
        """
        if threaded:
            self.thread = threading.Thread(target=self.server.run)
            self.thread.start()
        else:
            self.server.run()

    def register_interface(self, name: str, callback: Callable):
        """
        Registers the callback function for the interface
            :param name: Name of the interface
            :param callback: Callback function for the interface
        """
        self.interfaces[name] = callback

    def stop(self):
        """Stop the server"""
        self.server.stop()

##############################################################################


class InferenceClient:
    def __init__(self, server_ip: str, port_num: int):
        self.client = ReqRepClient(server_ip, port_num)

    def interfaces(self) -> Set[str]:
        """
        Returns the set of interfaces available on the server
            :return: Set of interfaces available on the server
        """
        response = self.client.send_msg({"type": "list_interfaces"})
        if response:
            return set(response.get('interfaces', []))
        return set()

    def call(self, name: str, payload: dict) -> Optional[dict]:
        """
        Calls the interface on the server with the given payload
            :param name: Name of the interface
            :param payload: Payload to send to the interface
            :return: Response from the interface
        """
        return self.client.send_msg(
            {"type": "call_interface", "interface": name, "payload": payload})

