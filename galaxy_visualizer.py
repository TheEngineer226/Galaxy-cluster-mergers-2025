# imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yt
from unyt import unyt_array
import numpy as np


# Dictionary of default parameters
DEFAULT_PARAMETERS = {
     "data_location" : "sims_data/R1.5_v2400_b250/Data_000044",
     "sphere_radius" :  3,
     "sphere_radius_units" : "Mpc",
     "field" : ("gas", "density")
      }


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


# Sets up the source
def setup_source_properties(source, field: tuple, is_log: bool, is_grey_opacity: bool, use_ghost_zones: bool, bounds: tuple = None):
    source.tfh.set_field(field)
    source.set_log(is_log)
    source.grey_opacity = is_grey_opacity
    source.set_use_ghost_zones(use_ghost_zones) #looks better but way slower
    if bounds:
        source.tfh.set_bounds(bounds)


# Loads in and volume renders galaxy data, then volume renders them
def main(data = DEFAULT_PARAMETERS['data_location'], sphere_radius = DEFAULT_PARAMETERS['sphere_radius'], sphere_radius_units = DEFAULT_PARAMETERS['sphere_radius_units'], field = DEFAULT_PARAMETERS['field']):
    ds = yt.load(data)

    # get CoM from particles (DM and stars) ------------------------------------------------------------
    ad = ds.all_data()
    com_x = ad.mean(('io', 'particle_position_x'))
    com_y = ad.mean(('io', 'particle_position_y'))
    com_z = ad.mean(('io', 'particle_position_z'))
    c = unyt_array([com_x, com_y, com_z], com_x.units)

    # how to just look at gas/particles within a sphere (not full simulation domain) -------------------
    radius = (sphere_radius, sphere_radius_units)
    sp = ds.sphere(c, radius) # c = CoM defined above, Tuple is required here as parameter

    # add a field for the total (DM + stars + gas) density
    def _total_density(field, data):
        return data['gas', 'density'] + data['gas', 'particle_density_on_grid']

    ds.add_field(('gas', 'total_density'), _total_density, units='g/cm**3', sampling_type='local')

    #total_density_in_sphere = sp[('gas', 'total_density')]  # Will probably be used in the future but not sure just yet
    #temperature_in_sphere = sp[('gas', 'temperature')]  # same as above

    plt.figure(figsize=(10, 5))

    sc = yt.create_scene(sp, field=field, lens_type='perspective')
    source = sc[0]

    bounds = (3e-31, 5e-27)

    setup_source_properties(source, field, True, True, False, bounds)
    save_and_show_img("density_transfer_function.png", 0, (1, 2, 1), "Transfer Function Density", True, field, source = source)
    save_and_show_img("density_rendering.png", 6, (1, 2, 2), "Rendered Scene Density", False, scene = sc)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
        main()



