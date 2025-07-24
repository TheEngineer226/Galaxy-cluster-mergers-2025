import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from yt.visualization.volume_rendering.transfer_functions import ColorTransferFunction, TransferFunction, MultiVariateTransferFunction
from yt.visualization.volume_rendering.api import Scene, create_volume_source
from matplotlib import colormaps
from unyt import unyt_array
import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("OPENBLAS_NUM_THREADS:", os.environ.get("OPENBLAS_NUM_THREADS"))
def find_CoM(dataset):
    ad = dataset.all_data()
    com_x = ad.mean(('io', 'particle_position_x'))
    com_y = ad.mean(('io', 'particle_position_y'))
    com_z = ad.mean(('io', 'particle_position_z'))
    return unyt_array([com_x, com_y, com_z], com_x.units)
# Load dataset
ds = yt.load("sims_data/R1.5_v2400_b250/Data_000000")
print("Loaded Dataset...")
# Field bounds
bounds_temp = (2.44e7, 4.50e8)
bounds_density = (3.5e-30, 4.2e-26)
# Gets CoM from particles (DM and stars)
center = find_CoM(dataset = ds)
print("Found CoM...")
# Look at gas/particles within a sphere (not full simulation domain)
radius = (3.5, "Mpc")
sp = ds.sphere(center, radius)
print("Created Sphere...")
# Create transfer functions with proper setup
tf_temp = ColorTransferFunction(bounds_temp, nbins=256)
tf_temp.map_to_colormap(bounds_temp[0], bounds_temp[1], colormap="seismic")
print("Temp_tf Created...")
tf_density = ColorTransferFunction(bounds_density, nbins=256)
# For density (alpha channel), create a simple ramp
tf_density.add_layers(4, w=0.01, colormap="viridis")
print("Density_tf Created...")
# Multivariate transfer function
mv = MultiVariateTransferFunction()
# Add field tables with explicit bounds
mv.add_field_table(tf_temp, field_id=0, weight_field_id=1, weight_table_id=-1)
mv.link_channels(0, [0, 1, 2])
#mv.add_field_table(tf_temp, field_id=0, weight_field_id=-1, weight_table_id=-1)      # temperature → RGB
#mv.add_field_table(tf_density, field_id=1, weight_field_id=-1, weight_table_id=-1)   # density → Alpha
# Link channels
#mv.link_channels(0, [0, 1, 2])  # temperature controls RGB channels
#mv.link_channels(1, [3])        # density controls alpha channel ########## BRACKETS AROUND THE 3?????
print("Multi_tf Created...")
# Create scene with proper field specification
fields = [("gas", "temperature"), ("gas", "density")]
sc = yt.create_scene(sp, fields)
source = sc[0]
# Apply the multivariate transfer function
source.tfh.tf = mv
# Set bounds for both fields
source.tfh.set_bounds(bounds_temp)  # Set bounds for the primary field (temperature)
# Set log scaling - temperature linear, density log
source.set_log([False, True])  # [temp_log, density_log]
source.set_use_ghost_zones(False)
print("Scene Created...")
# Camera settings
sc.camera.resolution = (256, 256)  # Higher resolution for better quality
#im = sc.render()
#plt.imshow(im)
#plt.axis('off')  # optional, hides axis ticks
#plt.show()
# Save and render scene
file_location = "ethan/multivariate_render.png"
sc.save(file_location, sigma_clip=4, render=True)
print("Saving Render...")
# Display render result
img = mpimg.imread(file_location)
plt.figure(figsize=(10, 5))
plt.imshow(img)
plt.axis("off")
plt.title("Multivariate Rendering: Temperature (turbo colormap) weighted by Density")
plt.tight_layout()
plt.show()
