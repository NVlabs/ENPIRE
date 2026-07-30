# SPDX-FileCopyrightText: Copyright (c) 2023 You Liang Tan
# SPDX-License-Identifier: MIT

"""
Pipe method is a simple way to send data from one process to another.
Different to ReqRep, pipe doesnt require a response from the receiver.
The PUSH socket sends messages to the PULL socket

https://zeromq.org/socket-api/#pipeline-pattern
https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pushpull.html
"""

import zmq
import argparse
import numpy as np
import time
import threading
import logging
from typing import Callable

from agentlace.internal.utils import make_compression_method


##############################################################################


class Producer:
    """Message producer for pipe communication."""

    def __init__(self,
                 ip: str = "localhost", 
                 port: int = 5547,
                 compression: str = 'lz4'):
        logging.debug(f"Initializing pipe consumer [{ip}:{port}]")
        if ip == "localhost":
            ip = "127.0.0.1"
        url = f"tcp://{ip}:{port}"
        context = zmq.Context()
        self.zmq_socket = context.socket(zmq.PUSH)
        self.zmq_socket.connect(url)
        self.compress, self.decompress = make_compression_method(compression)

    def send_msg(self, msg):
        msg = self.compress(msg)
        self.zmq_socket.send(msg)

##############################################################################


class Consumer:
    """Message consumer for pipe communication."""

    def __init__(self,
                 callback_fn: Callable,
                 port: int = 5547,
                 compression: str = 'lz4'):
        logging.debug(f"Initializing pipe consumer [localhost:{port}]")
        context = zmq.Context()
        self.results_receiver = context.socket(zmq.PULL)
        url = f"tcp://*:{port}"
        self.results_receiver.bind(url)
        self.callback_fn = callback_fn
        self.compress, self.decompress = make_compression_method(compression)

    def start(self):
        """blocking start method."""
        self.is_kill = False
        while self.is_kill is False:
            message = self.results_receiver.recv()
            message = self.decompress(message)
            self.callback_fn(message)

    def async_start(self):
        """non-blocking start method."""
        self.thread = threading.Thread(target=self.start)
        self.thread.start()

    def stop(self):
        self.is_kill = True
        if self.thread:
            self.thread.join()
        self.results_receiver.close()


##############################################################################

if __name__ == "__main__":
    # NOTE: This is just for Testing
    parser = argparse.ArgumentParser(description='zmq producer')
    parser.add_argument('--producer', action='store_true',
                        help='run as producer')
    parser.add_argument('--consumer', action='store_true',
                        help='run as consumer')
    args = parser.parse_args()

    if args.consumer:
        def callback_fn(result):
            name = result["name"]
            id = result["id"]
            print("received ", name, id)

        consumer = Consumer(callback_fn)
        consumer.async_start()
        time.sleep(20)
        print("Stopping consumer")
        consumer.stop()

    elif args.producer:
        producer = Producer()
        payload = np.zeros(100)
        for i in range(10):
            time.sleep(0.5)
            producer.send_msg({"name": "test", "id": i, "value": payload})
    else:
        print("Please specify --producer or --consumer")
        parser.print_help()
        exit(1)

