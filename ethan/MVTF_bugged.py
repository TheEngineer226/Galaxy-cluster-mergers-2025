import yt
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array
import numpy as np

# These specific imports are needed for this advanced method
from yt.visualization.volume_rendering.transfer_functions import ColorTransferFunction, MultiVariateTransferFunction, TransferFunction

# --- Default Parameters ---
DEFAULT_PARAMETERS = {
     "data_location" : "sims_data/R1.5_v2400_b250/Data_000000",
     "sphere_radius" :  4,
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

# --- Main Rendering Function ---
def main(data: str, sphere_radius: float, sphere_radius_units: str):
    """Main function to render the galaxy cluster data."""
    output_dir = "ethan/"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # --- Field Definitions ---
    color_field = ('gas', 'temperature')
    opacity_field = ('gas', 'density')

    # --- Data Loading & Field Creation ---
    ds = yt.load(data)
    if color_field not in ds.field_list:
        def _temperature(field, data):
            # A common proxy for temperature is Energy / Density
            return data[('gamer', 'Engy')] / data[('gamer', 'Dens')]
        ds.add_field(name=color_field, sampling_type="cell", function=_temperature, units="K")

    c = find_CoM(dataset=ds)
    radius = ds.quan(sphere_radius, sphere_radius_units)
    sp = ds.sphere(c, radius)

    # --- Calculate Bounds for the Transfer Functions ---
    min_temp, max_temp = sp.min(color_field), sp.max(color_field)
    log_temp_bounds = [np.log10(min_temp.v), np.log10(max_temp.v)]
    
    min_dens, max_dens = sp.min(opacity_field), sp.max(opacity_field)
    log_dens_bounds = [np.log10(min_dens.v), np.log10(max_dens.v)]
    
    # --- The "Two-Table" MVTF Workflow ---
    
    # 1. Create a transfer function for COLOR, based on Temperature.
    tf_color = ColorTransferFunction(log_temp_bounds)
    tf_color.add_layers(10, colormap="gray", w=0.5)

    # 2. Create a separate, simple transfer function for OPACITY, based on Density.
    tf_opacity = TransferFunction(log_dens_bounds)
    # This ramp uses a non-linear curve to make faint features visible and is amplified.
    normalized_ramp = np.linspace(0.0, 1.0, tf_opacity.nbins)
    curved_ramp = normalized_ramp**0.5 
    tf_opacity.y = 20.0 * curved_ramp

    # 3. Create the MultiVariateTransferFunction object.
    mvtf = MultiVariateTransferFunction()
    
    # 4. Add two separate tables to the MVTF.
    # Table 0: Opacity, driven by field_id=1 (density).
    mvtf.add_field_table(tf_opacity, field_id=1)
    # Table 1: Color, driven by field_id=0 (temperature).
    mvtf.add_field_table(tf_color, field_id=0)
    
    # 5. Link the renderer's output channels to the correct tables.
    # The final color (RGB) comes from the temperature recipe (table_id=1).
    mvtf.link_channels(table_id=1, channels=[0, 1, 2])
    # The final opacity (Alpha) comes from the density recipe (table_id=0).
    mvtf.link_channels(table_id=0, channels=[3])

    # 6. Create a scene with a LIST of the two fields we need.
    # The order MUST match the field_id's above.
    fields_to_render = [color_field, opacity_field]
    sc = yt.create_scene(sp, fields_to_render)
    
    # 7. Use the direct shortcut to get the source.
    source = sc[0]
    
    # 8. Assign our custom MVTF to the transfer function helper.
    source.tfh.tf = mvtf
    
    # 9. Set the bounds and log scaling on the helper.
    source.tfh.set_bounds((min_temp.v, max_temp.v))
    source.set_log([True, True]) # Temp=log, Density=log

    # --- Visualization ---
    output_filename = os.path.join(output_dir, "mvtf_render.pdf")
    print(f"Rendering image to {output_filename}...")
    sc.save(output_filename, sigma_clip=4.0)
    print("Render complete!")
    
    # --- Display Result ---
    image_array = sc.render()
    plt.figure(figsize=(8,8))
    plt.imshow(image_array)
    plt.axis("off")
    plt.title(f"Color: Temperature | Opacity: Density")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(DEFAULT_PARAMETERS['data_location'], DEFAULT_PARAMETERS['sphere_radius'], DEFAULT_PARAMETERS['sphere_radius_units'])
