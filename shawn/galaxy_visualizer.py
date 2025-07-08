# Run this in the home folder

import numpy as np
import yt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def save_and_show_img(file_location: str, s_clip: float, subplot_cords: tuple, title: str, is_transfer_function: bool, p_field: str = None):
    # Render and save the scene
    if is_transfer_function:
        source.tfh.plot(file_location, profile_field=p_field)
    else:
        sc.save(file_location, sigma_clip=s_clip)

    # Load rendered image and display
    render_img = mpimg.imread(file_location)

    plt.subplot(*subplot_cords)
    plt.imshow(render_img)
    plt.axis('off')
    plt.title(title)


def setup_source_properties(source, field: tuple, is_log: bool, is_grey_opacity: bool, bounds: tuple = None):
    source.set_field(field)
    source.set_log(is_log)
    source.grey_opacity = is_grey_opacity
    if bounds:
        source.tfh.set_bounds(bounds)


ds = yt.load("sims_data/R1.5_v2400_b250/Data_000000")

# Create a scene with a perspective camera
sc = yt.create_scene(ds, lens_type="perspective")
source = sc[0]
plt.figure(figsize=(10, 5))


# Density
setup_source_properties(source, ("gas", "density"), True, True, (3e-31, 5e-27))

save_and_show_img("shawn/density_transfer_function.png", 0, (2, 2, 1), "Transfer Function Density", True, ("gas", "density"))
save_and_show_img("shawn/density_rendering.png", 6, (2, 2, 2), "Rendered Scene Density", False)


# Temperature
setup_source_properties(source, ("gas", "temperature"), True, True)

save_and_show_img("shawn/temperature_transfer_function.png", 0, (2, 2, 3), "Transfer Function Temperature", True, ("gas", "temperature"))
save_and_show_img("shawn/temperature_rendering.png", 6, (2, 2, 4), "Rendered Scene Temperature", False)


plt.tight_layout()
plt.show()
