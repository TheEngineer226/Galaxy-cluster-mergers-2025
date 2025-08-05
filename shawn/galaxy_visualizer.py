import yt
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array, unyt_quantity
import numpy as np
from pdf2image import convert_from_path
from yt.visualization.volume_rendering.render_source import VolumeSource


# --- Default Parameters ---
DEFAULT_PARAMETERS = {
    "data_location" : "sims_data/R1.5_v2400_b250/Data_000044", 
    "save_path" : "shawn/",
    "sphere_radius" : 3, #
    "sphere_radius_units": "Mpc",
    "temp_threshold_value" : 10, #
    "temp_threshold_unit" : "keV",
    "camera_position" : "los", # Input "los" to use the line of sight, 
    "colormap" : "cmr.prinsenvlag_r"
}


# --- Helper Functions ---
def find_CoM(dataset):
    """
    Returns the CoM from all the particles (DM and stars).
    
    Parameters
    ----------
    dataset : ???
        STUFF HERE
    """
    ad = dataset.all_data()
    com_x = ad.mean(('particle_position_x'))
    com_y = ad.mean(('particle_position_y'))
    com_z = ad.mean(('particle_position_z'))
    return unyt_array([com_x, com_y, com_z], com_x.units)


def create_shock_field(ds, temp_threshold_value, temp_threshold_unit):
    """
    Description
    
    Parameters
    ----------
    dataset : ???
        STUFF HERE
    """

    # Define temperature threshold, converting keV to K using thermal equivalence
    shock_temp_threshold = ds.quan(temp_threshold_value, temp_threshold_unit).to("K", equivalence="thermal")
    
    def _shock_mask(field, data):
        temp = data[('gas', 'temperature')]
        mask_array = (temp >= shock_temp_threshold).astype("int")
        # Return as a yt array with dimensionless units to prevent errors
        return data.ds.arr(mask_array, "")

    shock_mask_field = ('gas', 'shock_temperature_mask')
    shock_velocity_field = ('gas', 'shock_velocity')

    # Define the binary temperature mask field if it doesn't exist
    if shock_mask_field not in ds.field_list:
        ds.add_field(
            name=shock_mask_field,
            sampling_type="cell",
            function=_shock_mask,
            units="dimensionless"
        )
        
    # Define the final shock velocity field if it doesn't exist
    if shock_velocity_field not in ds.field_list:
        def _shock_velocity(field, data):
            mask = data[shock_mask_field]
            vel_mag = data[('gas', 'velocity_magnitude')]
            return mask * vel_mag
        
        ds.add_field(
            name=shock_velocity_field,
            sampling_type="cell",
            function=_shock_velocity,
            units="km/s"
        )
    return shock_velocity_field


def find_shock_bounds(sp, shock_velocity_field):
    """
    Description
    
    Parameters
    ----------
    dataset : ???
        STUFF HERE
    """

    # Correctly find the non-zero minimum velocity
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
    return log_vel_bounds, max_shock_vel, min_shock_vel_nonzero


# This creates a linear ramp for opacity from 0.0 to 1.0.
def alpha_func(vals, min_val, max_val):
    return (vals - min_val) / (max_val - min_val)


# Sets up the source
def setup_source_properties(
    sp, 
    field: tuple, 
    render_bounds: tuple = None,
    tf_bounds: tuple = None,
    colormap: str = None,
    alpha_function = None,
    use_log_space: bool = True
):
    """
    STUFF.
    
    Parameters
    ----------
    save_location : str
        Path to save the image (ignored for TF unless customized).
    s_clip : float
        Not currently used, but could control scaling or clipping.
    """
    sc = yt.create_scene(sp, field=field, lens_type='perspective')
    source = sc[0]
    source.set_log(use_log_space)

    tf = yt.ColorTransferFunction(render_bounds)

    # 3c. Use map_to_colormap with our custom alpha function
    tf.map_to_colormap(
        render_bounds[0],
        render_bounds[1],
        colormap=colormap,
        scale_func=alpha_function
    )

    source.tfh.tf = tf
    source.tfh.bounds = tf_bounds
    return sc, source

def setup_camera(
    sc, 
    position, 
    north_vector, 
    resolution
    ):
    """
    Stuff

    Parameters
    ----------
    stuff : str
        stuff
    """
    # Set the camera to look at the region of interest
    cam = sc.camera
    cam.position = position
    cam.north_vector = north_vector
    cam.resolution = (resolution, resolution) # can lower resolution to 128 to prioritize faster rendering. 512 is the standard quality

def save_and_prep_img(
        save_location: str, 
        s_clip: float, 
        subplot_cords: tuple, 
        title: str,
        is_transfer_function: bool, 
        p_field=None, 
        source=None, 
        scene=None
    ):
    
    """
    Renders and displays either a transfer function or a volume render image.
    
    Parameters
    ----------
    save_location : str
        Path to save the image (ignored for TF unless customized).
    s_clip : float
        Not currently used, but could control scaling or clipping.
    subplot_cords : tuple
        Tuple like (rows, cols, index) for subplot layout.
    title : str
        Title to show on the subplot.
    is_transfer_function : bool
        Whether to plot the transfer function instead of the volume render.
    p_field : str, optional
        Field to use for transfer function profile.
    source : yt volume source, optional
        Used for transfer function plotting.
    scene : yt Scene, optional
        Used for rendering the volume.
    """
    
    plt.subplot(*subplot_cords)
    
    if is_transfer_function:
        # Save transfer function plot as PDF
        source.tfh.plot(save_location, profile_field=p_field)

        # Convert first page of PDF to image (PIL format)
        images = convert_from_path(save_location, dpi=200)
        img = images[0]

        # Show in matplotlib
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    else:
        img = scene.render()

        # Composite alpha over white background if alpha channel exists
        if img.shape[2] == 4:
            rgb = img[..., :3]
            alpha = img[..., 3:4]
            img = rgb * alpha + (1 - alpha) * 1.0  # Composite over white

        # Normalize and apply gamma correction
        img = np.clip(img / img.max(), 0, 1)
        gamma = 1.0 / 1.4
        img = img ** gamma

        # Show and save the image
        #plt.imshow(img, origin='lower', cmap='magma') ###### CHECK TO SEE WHETHER CMAP IS EVEN NEEDED and whether bounds are correct
        plt.imshow(img, origin='lower')
        plt.axis('off')
        plt.title(title)
        plt.colorbar(label = "Velocity")
        plt.savefig(save_location, bbox_inches='tight', pad_inches=0)


# --- Main Rendering Function ---
def main(
        data: str = DEFAULT_PARAMETERS['data_location'], 
        save_folder: str = DEFAULT_PARAMETERS['save_path'],
        sphere_radius: float = DEFAULT_PARAMETERS['sphere_radius'], 
        sphere_radius_units: str = DEFAULT_PARAMETERS['sphere_radius_units'],
        temp_threshold_value: float = DEFAULT_PARAMETERS['temp_threshold_value'],
        temp_threshold_unit: str = DEFAULT_PARAMETERS['temp_threshold_unit'],
        colormap: str = DEFAULT_PARAMETERS["colormap"]
    ):
    """
    Main function to render the galaxy cluster data.

    Parameters
    ----------
    data : str
        STUFF HERE
    sphere_radius : float, optional
        STUFF HERE
    sphere_radius_units : str, optional
        STUFF HERE
    """

    # --- Data Loading ---
    ds = yt.load(data)

    # --- 1. Define Shock Fields ---
    shock_velocity_field = create_shock_field(ds, temp_threshold_value, temp_threshold_unit)
    
    # --- 2. Create Data Sphere and Get Bounds ---
    c = find_CoM(dataset=ds)
    radius = ds.quan(sphere_radius, sphere_radius_units)
    sp = ds.sphere(c, radius)

    log_vel_bounds, max_shock_vel, min_shock_vel_nonzero = find_shock_bounds(sp, shock_velocity_field)

    # --- 3. Setup Render ---
    if max_shock_vel > 0 and min_shock_vel_nonzero > 0:
        tf_bounds = (min_shock_vel_nonzero.v, max_shock_vel.v)
    else:
        # Fallback bounds if no velocity data
        print("Warning: No valid shock velocities found, using fallback tf_bounds (0.1, 1.0)")
        tf_bounds = (0.1, 1.0)

    sc, source = setup_source_properties(sp, shock_velocity_field, log_vel_bounds, tf_bounds, colormap, alpha_func, True)

    L = np.asarray([0.94, -0.10, 0.31]) / np.linalg.norm(np.asarray([0.94, -0.10, 0.31]))
    N = np.array([L[1], -L[0], 0])
    setup_camera(sc, L, N, 256)

    # --- 4. Visualization ---
    plt.figure(figsize=(10, 5))

    # Save plots and saves the plots as PDF
    save_and_prep_img(save_folder + "transfer_function.pdf", 0, (1, 2, 1), "Shock Velocity TF", is_transfer_function=True, p_field=shock_velocity_field, source=source)
    save_and_prep_img(save_folder + "shock_render.pdf", 4.0, (1, 2, 2), "Shock Velocity Map (>10 keV)", is_transfer_function=False, scene=sc)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
