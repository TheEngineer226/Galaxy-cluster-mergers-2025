import yt
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array, unyt_quantity
import numpy as np
from yt.utilities.orientation import Orientation

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
        # This correctly sets the transfer function plot to a linear scale.
        source.tfh.set_log(False)
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
    output_dir = "ethan/"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # --- Data Loading ---
    print("Starting render process...")
    ds = yt.load(data)

    # --- 1. Define Observer's Viewpoint ---
    L = [1, 0, 0] 
    N = [0, 0, 1] 
    orient = Orientation(L, north_vector=N)
    z_hat = orient.unit_vectors[2] 

    # --- 2. Define Custom Line-of-Sight Velocity Field ---
    los_velocity_field = ('gas', 'los_velocity_side_on')
    
    if los_velocity_field not in ds.field_list:
        def _los_velocity(field, data):
            v_los = (data['gas', 'velocity_x'] * z_hat[0] +
                     data['gas', 'velocity_y'] * z_hat[1] +
                     data['gas', 'velocity_z'] * z_hat[2])
            return v_los
        ds.add_field(name=los_velocity_field, sampling_type="cell", function=_los_velocity, units="km/s")

    # --- 3. Create Data Sphere and Get Symmetrical Bounds ---
    c = find_CoM(dataset=ds)
    radius = ds.quan(sphere_radius, sphere_radius_units)
    sp = ds.sphere(c, radius)
    
    # Use the uncapped field to determine the bounds
    min_vlos, max_vlos = sp.min(los_velocity_field), sp.max(los_velocity_field)
    max_abs_vlos = max(abs(min_vlos), abs(max_vlos))
    color_bounds = [-max_abs_vlos.v, max_abs_vlos.v]

    # --- 4. Rendering Workflow ---
    # Render using the original, uncapped field
    sc = yt.create_scene(sp, field=los_velocity_field, lens_type='perspective')
    source = sc[0]
    source.set_log(False)
    
    tf = yt.ColorTransferFunction(color_bounds)
    
    # --- CHANGE: Re-add the custom inverted alpha function ---
    def inverted_centered_alpha_func(vals, min_val, max_val):
        mid = 0.5 * (min_val + max_val)
        half_range = 0.5 * (max_val - min_val)
        if half_range == 0:
            return np.zeros_like(vals)
        return np.abs(vals - mid) / half_range

    tf.map_to_colormap(color_bounds[0], color_bounds[1], colormap="coolwarm", scale_func=inverted_centered_alpha_func)
    
    source.tfh.tf = tf
    source.tfh.bounds = color_bounds

    # --- 5. Visualization ---
    plt.figure(figsize=(10, 5))
    
    tf_path = os.path.join(output_dir, "vlos_side_on_tf.png")
    render_path = os.path.join(output_dir, "vlos_side_on_render.png")

    save_and_show_img(tf_path, 0, (1, 2, 1), "Side-On LoS Velocity TF", is_transfer_function=True, p_field=los_velocity_field, source=source)
    save_and_show_img(render_path, 4.0, (1, 2, 2), "Side-On LoS Velocity", is_transfer_function=False, scene=sc)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
