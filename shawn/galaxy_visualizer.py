import numpy as np
import yt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


ds = yt.load("./sims_data/R1.5_v2400_b250/Data_000000")

sc = yt.create_scene(ds, lens_type="perspective")

source = sc[0]

source.set_field(("gas", "density"))
source.set_log(True)

bounds = (3e-31, 5e-27)

# Since this rendering is done in log space, the transfer function needs
# to be specified in log space.
tf = yt.ColorTransferFunction(np.log10(bounds))


def linramp(vals, minval, maxval):
    return (vals - vals.min()) / (vals.max() - vals.min())


tf.map_to_colormap(
    np.log10(3e-31), np.log10(5e-27), colormap="cmyt.arbre", scale_func=linramp
)

source.tfh.tf = tf
source.tfh.bounds = bounds

source.tfh.plot("./shawn/transfer_function.png", profile_field=("gas", "density"))

sc.save("./shawn/rendering.png", sigma_clip=6)


img = mpimg.imread("./shawn/rendering.png")
plt.imshow(img)
plt.axis('off')
plt.show()

try:
	os.remove("./shawn/rendering.png")
except:
	print("An error has occured")
