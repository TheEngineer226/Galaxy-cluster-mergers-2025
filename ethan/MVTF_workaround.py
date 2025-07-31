import yt
import numpy as np
from yt.visualization.volume_rendering.transfer_functions import ColorTransferFunction
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array
# --- Load dataset
ds = yt.load("sims_data/R1.5_v2400_b250/Data_000044")
rgb_field   = ("gas", "temperature")
alpha_field = ("gas", "density")
ad          = ds.all_data()
# --- Percentile bounds for robust scaling
T0, T_hi = np.percentile(ad[rgb_field], [5, 97])   # use 99th instead of 100
T_clip   = T_hi                                   # any value above this gets clipped
D0, D1   = np.percentile(ad[alpha_field], [5, 100])
# --- Encoding function with temperature clipping
n_layers = 512
num_rgb_bins = n_layers
bin_width    = 1. / num_rgb_bins
scale_factor = bin_width / 8  # alpha scaling
def encoded_field(field, data):
    # --- Clip temperature at T_clip
    temp = np.minimum(data[rgb_field], T_clip)
    tnorm = (temp - T0) / (T_clip - T0)
    tnorm = np.clip(tnorm, 0, 1)
    dnorm = (np.log10(data[alpha_field]) - np.log10(D0)) / (np.log10(D1) - np.log10(D0))
    dnorm = np.clip(dnorm, 0, 1)
    return tnorm + dnorm * scale_factor
ds.add_field(
    ("gas", "enc_temp_density"),
    function=encoded_field,
    units="dimensionless",
    sampling_type="cell",
    force_override=True
)
def find_CoM(dataset):
    ad = dataset.all_data()
    com_x = ad.mean(('io', 'particle_position_x'))
    com_y = ad.mean(('io', 'particle_position_y'))
    com_z = ad.mean(('io', 'particle_position_z'))
    return unyt_array([com_x, com_y, com_z], com_x.units)
center = find_CoM(dataset = ds)
print("Found CoM...")
radius = (4, "Mpc")
sp = ds.sphere(center, radius)
sc  = yt.create_scene(sp, ("gas", "enc_temp_density"), lens_type="perspective")
# --- Scene setup
#scene  = yt.create_scene(ds, ("gas", "enc_temp_density"), lens_type="perspective")
source = sc[0]
source.set_log(False)
bounds = (float(ad["gas", "enc_temp_density"].min()), float(ad["gas", "enc_temp_density"].max()))
# --- Build custom TF
ctf = ColorTransferFunction(bounds)
cmap = plt.get_cmap("inferno")
encoded_vals = np.linspace(bounds[0], bounds[1], n_layers)
for val in encoded_vals:
    t_part = (val // scale_factor) * scale_factor
    d_part = (val - t_part) / scale_factor
    tnorm  = t_part
    dnorm  = np.clip(d_part, 0, 1)
    alpha  = 0.05 + 0.7 * dnorm
    r, g, b, _ = cmap(tnorm)
    ctf.add_gaussian(float(val), width=bin_width*2, height=[r, g, b, alpha])
    #ctf.add_gaussian(float(val), width=bin_width*2, height=[1, 1, 1, alpha])
source.tfh.tf = ctf
L = np.asarray([0.94, -0.10, 0.31]) / np.linalg.norm(np.asarray([0.94, -0.10, 0.31]))
N = np.array([L[1], -L[0], 0])
cam = sc.camera
cam.position = L
cam.north_vector = N
cam.resolution = (256, 256)
# --- Render
sc.save("ethan/encoded.png", render=True)
img = mpimg.imread("ethan/encoded.png")
plt.imshow(img)
plt.axis("off")
plt.title("Temperature (rgb) & Density (alpha)")
plt.tight_layout()
plt.show()
print(f"Clipping applied: all T > {T_clip:.3e} are mapped to max color")
