"""Sim adapters: wrap an EnvProtocol backend as CapServer arm/camera clients.

These are drop-in replacements for the real hardware _ArmClient and _CameraClient
used by CapServer when running in simulation mode.
"""

from enpire.env.forge.cap.server.sim_backend import SimArmClient as SimArmAdapter, SimCameraClient as SimCameraAdapter  # re-export

__all__ = ["SimArmAdapter", "SimCameraAdapter"]
