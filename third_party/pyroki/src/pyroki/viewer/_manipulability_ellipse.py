from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp
import trimesh.creation
import viser
from jax.typing import ArrayLike
from loguru import logger

from .._robot import Robot


class ManipulabilityEllipse:
    """Helper class to visualize the manipulability ellipsoid for a robot link."""

    def __init__(
        self,
        server: viser.ViserServer | viser.ClientHandle,
        robot: Robot,
        root_node_name: str = "/manipulability",
        target_link_name: Optional[str] = None,
        scaling_factor: float = 0.2,
        visible: bool = True,
        wireframe: bool = True,
        color: Tuple[int, int, int] = (200, 200, 255),
    ):
        """Initializes the manipulability ellipsoid visualizer.

        Args:
            server: The Viser server or client handle.
            robot: The Pyroki robot model.
            root_node_name: The base name for the ellipsoid mesh in the Viser scene.
            target_link_name: Optional name of the link to visualize the ellipsoid for initially.
            scaling_factor: Scaling factor applied to the ellipsoid dimensions.
            visible: Initial visibility state.
            wireframe: Whether to render the ellipsoid as a wireframe.
            color: The color of the ellipsoid mesh.
        """
        self._server = server
        self._robot = robot
        self._root_node_name = root_node_name
        self._target_link_name = target_link_name
        self._scaling_factor = scaling_factor
        self._visible = visible
        self._wireframe = wireframe
        self._color = color

        self._base_manip_sphere = trimesh.creation.icosphere(radius=1.0)
        self._mesh_handle: Optional[viser.MeshHandle] = None
        self._target_link_index: Optional[int] = None
        self._last_joints: Optional[jnp.ndarray] = None

        # Initial creation of the mesh handle (hidden if not visible)
        self._create_mesh_handle()

        # Set initial target link if provided
        self.set_target_link(target_link_name)

        self.manipulability = 0.0

    def _create_mesh_handle(self):
        """Creates or recreates the mesh handle in the Viser scene."""
        if self._mesh_handle is not None:
            self._mesh_handle.remove()

        # Create with dummy data initially, will be updated
        self._mesh_handle = self._server.scene.add_mesh_simple(
            self._root_node_name,
            vertices=onp.zeros((1, 3), dtype=onp.float32),
            faces=onp.zeros((1, 3), dtype=onp.uint32),
            color=self._color,
            wireframe=self._wireframe,
            visible=self._visible,
        )

        # Viser version compatibility.
        if hasattr(self._mesh_handle, "cast_shadow"):
            self._mesh_handle.cast_shadow = (  # type: ignore[attr-defined]
                False  # Ellipsoids usually don't need shadows
            )

    def set_target_link(self, link_name: Optional[str]):
        """Sets the target link for which to display the ellipsoid.

        Args:
            link_name: The name of the target link, or None to disable.
        """
        if link_name is None:
            self._target_link_index = None
            self.set_visibility(False)  # Hide if no target link
        else:
            try:
                self._target_link_index = self._robot.links.names.index(link_name)
                # If we previously hid because of no target, make visible again
                # if the user hasn't explicitly set visibility to False.
                if self._mesh_handle is not None and self._visible:
                    self._mesh_handle.visible = True
            except ValueError:
                logger.warning(f"Link name '{link_name}' not found in robot model.")
                self._target_link_index = None
                self.set_visibility(False)  # Hide if link not found

    def update(self, joints: ArrayLike):
        """Updates the ellipsoid based on the current joint configuration.

        Args:
            joints: The current joint angles of the robot.
        """
        if (
            self._target_link_index is None
            or not self._visible
            or self._mesh_handle is None
        ):
            # Ensure mesh is hidden if it shouldn't be shown
            if self._mesh_handle is not None and self._mesh_handle.visible:
                self._mesh_handle.visible = False
            return

        # Ensure mesh is visible if it should be
        if not self._mesh_handle.visible:
            self._mesh_handle.visible = True

        joints = jnp.asarray(joints)
        self._last_joints = joints  # Store for potential future updates

        try:
            # --- Jacobian Calculation ---
            jacobian = jax.jacfwd(
                lambda q: jaxlie.SE3(self._robot.forward_kinematics(q)).translation()
            )(joints)[self._target_link_index]
            assert jacobian.shape == (3, self._robot.joints.num_actuated_joints)

            # --- Manipulability Calculation ---
            JJT = jacobian @ jacobian.T
            assert JJT.shape == (3, 3)
            self.manipulability = jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(JJT))).item()

            # --- Covariance and Eigen decomposition ---
            cov_matrix = jacobian @ jacobian.T
            assert cov_matrix.shape == (3, 3)
            # Use numpy for Eigh as it might be more stable for visualization
            vals, vecs = onp.linalg.eigh(onp.array(cov_matrix))
            vals = onp.maximum(vals, 1e-9)  # Clamp small eigenvalues for stability

            # --- Get Target Link Pose ---
            Ts_link_world_array = self._robot.forward_kinematics(joints)
            target_pose_array = Ts_link_world_array[self._target_link_index]
            target_pose = jaxlie.SE3(target_pose_array)
            target_pos = onp.array(target_pose.translation().squeeze())

            # --- Create and Transform Ellipsoid Mesh ---
            ellipsoid_mesh = self._base_manip_sphere.copy()
            tf = onp.eye(4)
            tf[:3, :3] = onp.array(vecs)  # Rotation from eigenvectors
            tf[:3, 3] = target_pos  # Translation to link origin

            # Apply scaling according to eigenvalues and the scaling factor
            ellipsoid_mesh.apply_scale(onp.sqrt(vals) * self._scaling_factor)
            # Apply the final transform
            ellipsoid_mesh.apply_transform(tf)

            # --- Update Viser Mesh ---
            self._mesh_handle.vertices = onp.array(
                ellipsoid_mesh.vertices, dtype=onp.float32
            )
            self._mesh_handle.faces = onp.array(ellipsoid_mesh.faces, dtype=onp.uint32)

        except Exception as e:
            logger.warning(f"Failed to update manipulability ellipsoid: {e}")
            # Hide the mesh on failure
            if self._mesh_handle is not None:
                self._mesh_handle.visible = False

    def set_visibility(self, visible: bool):
        """Sets the visibility of the ellipsoid mesh."""
        self._visible = visible
        if self._mesh_handle is not None:
            # If visibility is being turned on, and we have a target link and joints,
            # trigger an update to ensure the geometry is correct.
            if (
                visible
                and self._target_link_index is not None
                and self._last_joints is not None
            ):
                self.update(self._last_joints)  # Recalculate and show
            # Otherwise, just set the visibility flag on the handle
            elif self._mesh_handle.visible != visible:
                self._mesh_handle.visible = visible

    def remove(self):
        """Removes the ellipsoid mesh from the Viser scene."""
        if self._mesh_handle is not None:
            self._mesh_handle.remove()
            self._mesh_handle = None
        self._target_link_index = None  # Clear target when removed
