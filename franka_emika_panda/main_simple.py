# import numpy as np
# from scipy.spatial import transform
# import matplotlib.pyplot as plt
#
# import jax
# import jax.numpy as jnp
#
# import mediapy
#
# import collimator
# from collimator import library
# from collimator.backend import io_callback
#
# import mujoco
#
# mjmodel = library.mujoco.MuJoCo(
#     file_name="scene.xml",
#     key_frame_0="home",
#     enable_video_output=True,
# )
#
# frame = mjmodel.render()
#
# mediapy.show_image(frame)
import mujoco
import mujoco.viewer
import os

# Find the mujoco package path
model_path = "scene.xml"
# Load the model
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Viewer loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()