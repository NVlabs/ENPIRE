# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import queue
import threading
import cv2


def put_latest_image(image_queue, img_array):
    try:
        image_queue.put_nowait(img_array)
    except queue.Full:
        try:
            image_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            image_queue.put_nowait(img_array)
        except queue.Full:
            pass


class ImageDisplayer(threading.Thread):
    def __init__(self, image_queue, name, size=128):
        threading.Thread.__init__(self)
        self.queue = image_queue
        self.daemon = True  # make this a daemon thread
        self.name = name
        self._frame_size = size
        self._last_window_shape = None

    def run(self):
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break
            while True:
                try:
                    latest_img_array = self.queue.get_nowait()
                except queue.Empty:
                    break
                if latest_img_array is None:
                    return
                img_array = latest_img_array

            frame_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            window_shape = (w, h)
            if window_shape != self._last_window_shape:
                cv2.resizeWindow(self.name, w, h)
                self._last_window_shape = window_shape
            cv2.imshow(self.name, frame_rgb)
            cv2.waitKey(1)
