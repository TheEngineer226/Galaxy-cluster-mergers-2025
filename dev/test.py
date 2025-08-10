import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from yt.visualization.volume_rendering.api import Scene, create_volume_source

# Load dataset
ds = yt.load("sims_data/R1.5_v2400_b250/Data_000000")

# Define a derived field: temperature weighted by density
def temp_density_weighted(field, data):
    return data[("gas", "temperature")] * data[("gas", "density")]

# Add the derived field to yt
yt.add_field(
    ("gas", "temp_density_weighted"),
    function=temp_density_weighted,
    units="K*g/cm**3",
    sampling_type="cell"
)

# Create scene using the new field
sc = yt.create_scene(ds, field=("gas", "temp_density_weighted"))

# Clear default sources cleanly
for key in list(sc.sources.keys()):
    sc.sources.pop(key)

# Create volume source for the derived field
source = create_volume_source(ds, field=("gas", "temp_density_weighted"))
tf = source.transfer_function
tf.clear

# Set field bounds (adjust based on your data)
ad = ds.all_data()
min_val, max_val = ad.quantities.extrema(("gas", "temp_density_weighted"))
min_val = min_val.to_value()
max_val = max_val.to_value()

# Create 8 log-spaced transfer function points between min and max
vals = np.logspace(np.log10(max(min_val, 1e-30)), np.log10(max_val), 8)

# Use matplotlib colormap
cmap = plt.get_cmap("plasma")

# Add each Gaussian manually with color
for val in vals:
    norm = (val - min_val) / (max_val - min_val)
    opacity = 0.05 + 0.5 * norm

    rgba = list(cmap(norm))
    rgba[3] = opacity  # set alpha

    tf.add_gaussian(val, width=0.1, height=rgba)

# Add source to the scene
sc.add_source(source)

# Save the rendered image
sc.save("rendered_temp_density_weighted.png", sigma_clip=4.0)

img = mpimg.imread("michaela/rendered_temp_density_weighted.png")
plt.imshow(img)
plt.show()
