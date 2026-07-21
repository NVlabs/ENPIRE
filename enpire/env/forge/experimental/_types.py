from dataclasses import dataclass, field
from typing import Any

import numpy as np
from enpire.env.forge.experimental.embodiment_tags import EmbodimentTag

@dataclass
class VLAStepData:
    """
    Represents a single step of VLA (Vision-Language-Action) data.

    This is the core data structure returned by datasets, containing raw observation
    and action data that will be processed by the SequenceVLAProcessor.
    """

    # Core data
    images: dict[str, list[np.ndarray]]  # view_name -> list[np.ndarray] (for temporal stacking)
    states: dict[
        str, np.ndarray
    ]  # state_name -> np.ndarray (dim,) for single step or (horizon, dim) for trajectory
    actions: dict[str, np.ndarray]  # action_name -> np.ndarray (horizon, dim) for action chunk
    text: str | None = None  # Optional task description or instruction
    rl_info: dict[str, np.ndarray] | None = None  # Optional RL info data
    embodiment: EmbodimentTag = (
        EmbodimentTag.XDOF
    )  # Optional embodiment tag for cross-embodiment training
    is_demonstration: bool = (
        False  # Whether the step is a demonstration. If True, no loss should be computed for this step.
    )

    # Flexible metadata that can be extended by users
    metadata: dict[str, Any] = field(default_factory=dict)
