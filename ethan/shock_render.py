import yt
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array, unyt_quantity
import numpy as np

# --- Default Parameters ---
DEFAULT_PARAMETERS = {
    "data_location": "sims_data/R1.5_v2400_b250/Data_000044",
    "sphere_radius": 3,
    "sphere_radius_units": "Mpc",
}

# --- Helper Functions ---
def find_CoM(dataset):
    """Returns the CoM from all the particles (DM and stars)."""
    ad = dataset.all_data()
    com_x = ad.mean(('particle_position_x'))
    com_y = ad.mean(('particle_position_y'))
    com_z = ad.mean(('particle_position_z'))
    return unyt_array([com_x, com_y, com_z], com_x.units)

def save_and_show_img(file_location: str, s_clip: float, subplot_cords: tuple, title: str, is_transfer_function: bool, p_field = None, source = None, scene = None):
    """Saves a plot as a PNG and displays it in a subplot."""
    if is_transfer_function:
        # This function saves the transfer function plot
        source.tfh.plot(file_location, profile_field=p_field)
    else:
        # This saves the main 3D rendering
        scene.save(file_location, sigma_clip=s_clip)

    # Read the saved PNG file back to display it in the subplot
    render_img = mpimg.imread(file_location)
    plt.subplot(*subplot_cords)
    plt.imshow(render_img)
    plt.axis('off')
    plt.title(title)

# --- Main Rendering Function ---
def main(data: str = DEFAULT_PARAMETERS['data_location'], sphere_radius: float = DEFAULT_PARAMETERS['sphere_radius'], sphere_radius_units: str = DEFAULT_PARAMETERS['sphere_radius_units']):
    """Main function to render the galaxy cluster data."""
    output_dir = "ethan/"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # --- Data Loading ---
    ds = yt.load(data)

    # --- 1. Define Shock Fields ---
    # Define temperature threshold, converting keV to K using thermal equivalence
    shock_temp_threshold = ds.quan(10, "keV").to("K", equivalence="thermal")

    # Define field names for clarity
    shock_mask_field = ('gas', 'shock_temperature_mask')
    shock_velocity_field = ('gas', 'shock_velocity')

    # Define the binary temperature mask field if it doesn't exist
    if shock_mask_field not in ds.field_list:
        def _shock_mask(field, data):
            temp = data[('gas', 'temperature')]
            mask_array = (temp >= shock_temp_threshold).astype("int")
            # Return as a yt array with dimensionless units to prevent errors
            return data.ds.arr(mask_array, "")
        ds.add_field(name=shock_mask_field,
                     sampling_type="cell",
                     function=_shock_mask,
                     units="dimensionless")

    # Define the final shock velocity field if it doesn't exist
    if shock_velocity_field not in ds.field_list:
        def _shock_velocity(field, data):
            mask = data[shock_mask_field]
            vel_mag = data[('gas', 'velocity_magnitude')]
            return mask * vel_mag
        ds.add_field(name=shock_velocity_field,
                     sampling_type="cell",
                     function=_shock_velocity,
                     units="km/s")

    # --- 2. Create Data Sphere and Get Bounds ---
    c = find_CoM(dataset=ds)
    radius = ds.quan(sphere_radius, sphere_radius_units)
    sp = ds.sphere(c, radius)

    # --- FIX: Correctly find the non-zero minimum velocity ---
    # The .min() shortcut doesn't accept 'non_zero'. We get the data array
    # and filter it manually for a more robust calculation.
    max_shock_vel = sp.max(shock_velocity_field)
    
    all_shock_vels = sp[shock_velocity_field]
    nonzero_vels = all_shock_vels[all_shock_vels > 0]
    
    min_shock_vel_nonzero = nonzero_vels.min() if nonzero_vels.size > 0 else yt.YTQuantity(0, 'km/s')

    # Set the logarithmic bounds for the color map
    if max_shock_vel > 0 and min_shock_vel_nonzero > 0:
        log_min = np.log10(min_shock_vel_nonzero.to('km/s').v)
        log_max = np.log10(max_shock_vel.to('km/s').v)
        log_vel_bounds = (log_min, log_max)
    else:
        # Fallback if there's no shock velocity in the sphere
        log_vel_bounds = (-1, 0)

    # --- 3. Rendering Workflow ---
    sc = yt.create_scene(sp, field=shock_velocity_field, lens_type='perspective')
    source = sc[0]
    source.set_log(True)

    # Setup the color transfer function
    tf = yt.ColorTransferFunction(log_vel_bounds)
    tf.map_to_colormap(log_vel_bounds[0], log_vel_bounds[1], colormap="magma")
    source.tfh.tf = tf
    
    # Set the bounds to the actual non-zero data range
    if max_shock_vel > 0 and min_shock_vel_nonzero > 0:
        source.tfh.bounds = (min_shock_vel_nonzero.v, max_shock_vel.v)
    else:
        # Fallback bounds if no velocity data
        source.tfh.bounds = (0.1, 1.0) 

    # --- 4. Visualization ---
    plt.figure(figsize=(10, 5))
    
    # Save plots as PNG to be readable by matplotlib's imread
    save_and_show_img(output_dir + "shock_tf.png", 0, (1, 2, 1), "Shock Velocity TF", is_transfer_function=True, p_field=shock_velocity_field, source=source)
    save_and_show_img(output_dir + "shock_render.png", 4.0, (1, 2, 2), "Shock Velocity Map (>10 keV)", is_transfer_function=False, scene=sc)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
