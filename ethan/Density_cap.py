import yt
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array
import numpy as np

# --- Default Parameters ---
DEFAULT_PARAMETERS = {
     "data_location" : "sims_data/R1.5_v2400_b250/Data_000044",
     "sphere_radius" :  3,
     "sphere_radius_units" : "Mpc",
     }


# --- Helper Functions ---
def find_CoM(dataset):
    """Returns the CoM from all the particles (DM and stars)."""
    ad = dataset.all_data()
    com_x = ad.mean(('io', 'particle_position_x'))
    com_y = ad.mean(('io', 'particle_position_y'))
    com_z = ad.mean(('io', 'particle_position_z'))
    return unyt_array([com_x, com_y, com_z], com_x.units)

def save_and_show_img(file_location: str, s_clip: float, subplot_cords: tuple, title: str, is_transfer_function: bool, p_field: tuple = None, source = None, scene = None):
    """Saves a plot and displays it in a subplot."""
    if is_transfer_function:
        source.tfh.plot(file_location, profile_field=p_field)
    else:
        scene.save(file_location, sigma_clip=s_clip)

    render_img = mpimg.imread(file_location)
    plt.subplot(*subplot_cords)
    plt.imshow(render_img)
    plt.axis('off')
    plt.title(title)


# --- Main Rendering Function ---
def main(data: str = DEFAULT_PARAMETERS['data_location'], sphere_radius: float = DEFAULT_PARAMETERS['sphere_radius'], sphere_radius_units: str = DEFAULT_PARAMETERS['sphere_radius_units']):
    """Main function to render the galaxy cluster data."""
    user_prefix = input("Enter your username (for saving images to a folder): ").strip()
    if not user_prefix.endswith('/'):
        user_prefix += '/'
    if not os.path.isdir(user_prefix):
        os.makedirs(user_prefix)
        print(f"Created directory: {user_prefix}")

    # --- Tuning Parameter for Density ---
    cap_density_val = 1e-27 

    # --- Field Definitions ---
    base_density_field = ('gas', 'density')
    capped_density_field = ('gas', 'capped_density') 

    # --- Data Loading & Field Creation ---
    ds = yt.load(data)
    cap_density = ds.quan(cap_density_val, "g/cm**3")
    
    c = find_CoM(dataset=ds)
    radius = ds.quan(sphere_radius, sphere_radius_units)
    sp = ds.sphere(c, radius)

    min_dens, _ = sp.min(base_density_field), sp.max(base_density_field)

    if cap_density <= min_dens:
        print(f"\nERROR: Your cap_density ({cap_density}) is lower than the minimum density in the data ({min_dens:.2e}).")
        print("Please choose a higher value for the cap.\n")
        return 

    # --- CORRECTED: Define the new "capped" density field with unit conversion ---
    if capped_density_field not in ds.field_list:
        def _capped_density(field, data):
            # First, convert the raw data to consistent units (g/cm**3)
            density_in_cgs = data[base_density_field].in_units("g/cm**3")
            # Now, clip the data. All arguments have the same units.
            return np.clip(density_in_cgs, min_dens, cap_density)
        ds.add_field(name=capped_density_field, sampling_type="cell", function=_capped_density, units="g/cm**3")

    min_capped_dens, max_capped_dens = sp.min(capped_density_field), sp.max(capped_density_field)
    log_dens_bounds = [np.log10(min_capped_dens.v), np.log10(max_capped_dens.v)]
    
    # --- Rendering Workflow ---
    sc = yt.create_scene(sp, field=capped_density_field, lens_type='perspective')
    source = sc[0]
    source.set_log(True)
    
    tf = yt.ColorTransferFunction(log_dens_bounds)
    
    def alpha_func(vals, min_val, max_val):
        return (vals - min_val) / (max_val - min_val)

    tf.map_to_colormap(
        log_dens_bounds[0],
        log_dens_bounds[1],
        colormap="viridis", 
        scale_func=alpha_func
    )

    source.tfh.tf = tf
    source.tfh.bounds = (min_capped_dens.v, max_capped_dens.v)

    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    save_and_show_img(user_prefix + "dens_tf_capped_data.png", 0, (1, 2, 1), "Capped Density TF", is_transfer_function=True, p_field=capped_density_field, source=source)
    save_and_show_img(user_prefix + "dens_render_capped_data.png", 4.0, (1, 2, 2), "Capped Density Render", is_transfer_function=False, scene=sc)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
