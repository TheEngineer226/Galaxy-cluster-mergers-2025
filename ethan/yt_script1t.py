import yt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
ds = yt.load("./sims_data/R1.5_v2400_b250/Data_000005")
sc = yt.create_scene(ds, lens_type="perspective")
source = sc[0]
bounds = (5e-30, 5e-27)
source.tfh.bounds = bounds
sc.save("ethan/rendered_output.png", sigma_clip=8.0)
img = mpimg.imread("ethan/rendered_output.png")
plt.imshow(img)
plt.axis('off')
plt.show()

