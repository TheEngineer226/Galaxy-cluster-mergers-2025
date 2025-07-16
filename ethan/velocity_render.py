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
      }


# Returns the CoM from all the particles (DM and stars)
def find_CoM(dataset):
    ad = dataset.all_data()
    com_x = ad.mean(('io', 'particle_position_x'))
    com_y = ad.mean(('io', 'particle_position_y'))
    com_z = ad.mean(('io', 'particle_position_z'))
    return unyt_array([com_x, com_y, com_z], com_x.units)


# Function to define velocity fields
def define_velocity_fields(ds):
    """Defines derived velocity fields from momentum and density."""
    if ("gas", "velocity_magnitude") in ds.field_list:
        return

    def _velocity_magnitude(field, data):
        vx = data[("gamer", "MomX")] / data[("gamer", "Dens")]
        vy = data[("gamer", "MomY")] / data[("gamer", "Dens")]
        vz = data[("gamer", "MomZ")] / data[("gamer", "Dens")]
        return (vx**2 + vy**2 + vz**2)**0.5

    ds.add_field(("gas", "velocity_magnitude"), function=_velocity_magnitude, units="cm/s", sampling_type="cell")


# Saves the rendering and then utilizes Matplotlib to display the render
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


def main(data: str = DEFAULT_PARAMETERS['data_location'], sphere_radius: float = DEFAULT_PARAMETERS['sphere_radius'], sphere_radius_units: str = DEFAULT_PARAMETERS['sphere_radius_units']):
    # Get user folder for saving images
    user_prefix = input("Enter your username (for saving images to a folder): ").strip()
    if not user_prefix.endswith('/'):
        user_prefix += '/'
    if not os.path.isdir(user_prefix):
        os.makedirs(user_prefix)

    # Load the dataset
    ds = yt.load(data)
    
    # Define the velocity fields for this dataset
    define_velocity_fields(ds)
    
    # Set the field to our derived velocity field
    field = ("gas", "velocity_magnitude")

    # The rest of the setup
    c = find_CoM(dataset=ds)
    radius = (sphere_radius, sphere_radius_units)
    sp = ds.sphere(c, radius)

    # Create the scene and get the source
    sc = yt.create_scene(sp, field=field, lens_type='perspective')
    source = sc[0]

    # --- The map_to_colormap Workflow ---

    # Set rendering to a linear scale for velocity
    source.set_log(False)
    
    # --- CHANGED: Manually define the data bounds for velocity in cm/s ---
    bounds = (0.75e8, 2.1e8)
    print(f"Using manual velocity bounds: {bounds[0]/1e5:.2f} km/s to {bounds[1]/1e5:.2f} km/s")

    # Create a ColorTransferFunction object
    tf = yt.ColorTransferFunction(bounds)

    # Define a function for opacity
    def alpha_func(vals, min_val, max_val):
        return (vals - min_val) / (max_val - min_val)

    # Use map_to_colormap to apply color and opacity
    tf.map_to_colormap(
        bounds[0],
        bounds[1],
        colormap="viridis",
        scale_func=alpha_func
    )

    # Assign the transfer function to the source
    source.tfh.tf = tf
    source.tfh.bounds = bounds

    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    save_and_show_img(user_prefix + "velocity_transfer_function.png", 0, (1, 2, 1), "Velocity Transfer Function", True, p_field=field, source=source)
    save_and_show_img(user_prefix + "velocity_render.png", 4.0, (1, 2, 2), "Volume Rendered Velocity", False, scene=sc)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
