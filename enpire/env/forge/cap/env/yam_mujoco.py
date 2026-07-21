"""YAM MuJoCo simulation environment (--env yam).

Implements EnvProtocol and SceneProtocol for the YAM bimanual station.
GPU-based EGL rendering on a dedicated thread for OpenGL context affinity.
"""

from enpire.env.forge.cap.server.sim_backend import SimBackend as YamMuJoCoEnv  # re-export

__all__ = ["YamMuJoCoEnv"]
