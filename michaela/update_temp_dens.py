import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from yt.visualization.volume_rendering.api import Scene, create_volume_source
from yt.visualization.volume_rendering.transfer_functions import TransferFunction, MultiVariateTransferFunction, ColorTransferFunction

# Load dataset
ds = yt.load("sims_data/R1.5_v2400_b250/Data_000044")

# Define a derived field: temperature weighted by density
def temp_density_weighted(field, data):
    return data[("gas", "temperature")] * data[("gas", "density")]

# Add the derived field
yt.add_field(
    ("gas", "temp_density_weighted"),
    function=temp_density_weighted,
    units="K*g/cm**3",
    sampling_type="cell"
)

# Create scene
sc = Scene()
source = create_volume_source(ds, ("gas", "temp_density_weighted"))

# Clear and set new transfer function
tf = source.transfer_function
tf.clear()

# Determine bounds and flatten extreme values
ad = ds.all_data()
min_val, max_val = ad.quantities.extrema(("gas", "temp_density_weighted"))
min_val = max(min_val.to_value(), 1e-30)
max_val = max_val.to_value()

# Define tighter rendering bounds (for better contrast)
vmin = np.log10(min_val)
vmax = np.log10(max_val)
clip_lo = vmin + 0.5 * (vmax - vmin) * 0.1  # clip lowest 10%
clip_hi = vmax - 0.5 * (vmax - vmin) * 0.1  # clip highest 10%

# Define custom linear ramp for color mapping
def linramp(vals, minval, maxval):
    return (vals - vals.min()) / (vals.max() - vals.min())

# Apply smooth color blending
tf.map_to_colormap(
    clip_lo,
    clip_hi,
    colormap="turbo",
    scale_func=linramp,
)

# Attach volume source and render
sc.add_source(source)
cam = sc.add_camera(ds)
sc.camera = cam
sc.save("michaela/temp_dens_weighted.png", sigma_clip=4.0)

# save transfer function plot
source.tfh.tf = tf
source.tfh.bounds = min_val, max_val
source.tfh.plot("michaela/weighted_transfer_func.png", profile_field=("gas", "temp_density_weighted"))

# Show image
render_img = mpimg.imread("michaela/temp_dens_weighted.png")
tf_img = mpimg.imread("michaela/weighted_transfer_func.png")

fig, axs = plt.subplots(1, 2, figsize=(14,6))
axs[0].imshow(render_img)
axs[0].axis("off")
axs[0].set_title("Temperature & Density")

axs[1].imshow(tf_img)
axs[1].axis("off")
axs[1].set_title("Transfer Function")

plt.tight_layout()
plt.show()
