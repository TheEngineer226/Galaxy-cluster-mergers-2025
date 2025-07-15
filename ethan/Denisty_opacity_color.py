import yt, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array
import numpy as np


# Dictionary of the default parameters
DEFAULT_PARAMETERS = {
     "data_location" : "sims_data/R1.5_v2400_b250/Data_000044",
     "sphere_radius" :  3,
     "sphere_radius_units" : "Mpc",
     "field" : ("gas", "density")
      }


# Returns the CoM from all the particles (DM and stars)
def find_CoM(dataset):
    ad = dataset.all_data()
    com_x = ad.mean(('io', 'particle_position_x'))
    com_y = ad.mean(('io', 'particle_position_y'))
    com_z = ad.mean(('io', 'particle_position_z'))
    return unyt_array([com_x, com_y, com_z], com_x.units)


def _total_density(field, data):
    return data['gas', 'density'] + data['gas', 'particle_density_on_grid']


# Sets up the source
def setup_source_properties(source, field: tuple, is_log: bool, is_grey_opacity: bool, use_ghost_zones: bool, bounds: tuple = None):
    source.tfh.set_field(field)
    source.set_log(is_log)
    source.grey_opacity = is_grey_opacity
    source.set_use_ghost_zones(use_ghost_zones) #Looks better but way slower
    if bounds:
        source.tfh.set_bounds(bounds)


# Saves the rendering and then utilizes Matplotlib to display the render 
# Saving images should pass in a VolumeSource, Transfer Functions should pass in scene 
def save_and_show_img(file_location: str, s_clip: float, subplot_cords: tuple, title: str, is_transfer_function: bool, p_field: tuple = None, source = None, scene = None):
    if is_transfer_function:
        source.tfh.plot(file_location, profile_field=p_field)
    else:
        scene.save(file_location, sigma_clip=s_clip)

    render_img = mpimg.imread(file_location)

    plt.subplot(*subplot_cords)
    plt.imshow(render_img)
    plt.axis('off')
    plt.title(title)


def main(data: str = DEFAULT_PARAMETERS['data_location'], sphere_radius: float = DEFAULT_PARAMETERS['sphere_radius'], sphere_radius_units: str = DEFAULT_PARAMETERS['sphere_radius_units'], field: tuple = DEFAULT_PARAMETERS['field']):
    # Get user folder for saving images
    user_prefix = input("Enter your username (for saving images to a folder): ").strip()
    if not user_prefix.endswith('/'):
        user_prefix += '/'

    # Load the dataset and define region
    ds = yt.load(data)
    c = find_CoM(dataset=ds)
    radius = (sphere_radius, sphere_radius_units)
    sp = ds.sphere(c, radius)

    # 1. Create the scene and get the source
    sc = yt.create_scene(sp, field=field, lens_type='perspective')
    source = sc[0]

    # --- The map_to_colormap Workflow ---

    # 2. Define the data bounds and set the rendering to log scale
    source.set_log(True)
    bounds = (10e-29, 5e-26)
    log_bounds = np.log10(bounds)

    # 3. Create a ColorTransferFunction object in log space
    tf = yt.ColorTransferFunction(log_bounds)

    # 4. Define a function that maps data values to alpha (opacity) values.
    # This function creates a linear ramp for opacity.
    def alpha_func(vals, min_val, max_val):
        # Linearly map values from their range [min_val, max_val]
        # to a new range of [0.0 (transparent), 1.0 (opaque)]
        return (vals - min_val) / (max_val - min_val)

    # 5. Use map_to_colormap to apply the colormap and opacity scale in one step.
    tf.map_to_colormap(
        log_bounds[0],
        log_bounds[1],
        colormap="viridis",
        scale_func=alpha_func
    )

    # 6. Assign our custom-built transfer function back to the scene source.
    source.tfh.tf = tf
    source.tfh.bounds = bounds

    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    save_and_show_img(user_prefix + "density_transfer_function.png", 0, (1, 2, 1), "Custom Transfer Function", True, p_field=field, source=source)
    save_and_show_img(user_prefix + "density_rendering.png", 4.0, (1, 2, 2), "Rendered Scene (Density-Weighted Opacity)", False, scene=sc)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
