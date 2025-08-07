import yt
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array, unyt_quantity
import numpy as np
from pdf2image import convert_from_path
from yt.visualization.volume_rendering.render_source import VolumeSource
import cmasher as cmr
import matplotlib.colors as mcolors

# --- Default Parameters ---
DEFAULT_PARAMETERS = {
    "data_location": "sims_data/R1.5_v2400_b250/Data_000330", #Data_000000 to Data_000419
    "save_path": "shawn/",
    "sphere_radius": 3, #
    "sphere_radius_units": "Mpc",
    "temp_threshold_value": 10, #
    "temp_threshold_unit": "keV",
    "camera_position": "los", # Input "los" to use the line of sight,
    "colormap": "cmr.prinsenvlag_r",
    "contrast": 1.4,
    "resolution": 256,
    "is_animation": False,
    "frame_number": None
}


z_hat = None # Line-of-sight direction (unit vector)


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
    Create shock-related fields in the dataset:
      - shock_temperature_mask: binary mask for T >= threshold
      - shock_velocity: LOS velocity in masked regions (can be negative)
    """

    # Convert threshold to Kelvin using thermal equivalence
    shock_temp_threshold = ds.quan(temp_threshold_value, temp_threshold_unit)\
                             .to("K", equivalence="thermal")

    shock_mask_field = ('gas', 'shock_temperature_mask')
    los_velocity_field = ('gas', 'los_velocity')
    shock_velocity_field = ('gas', 'shock_velocity')

    # Binary mask for hot gas above threshold
    def _shock_mask(field, data):
        temp = data[('gas', 'temperature')]
        mask_array = (temp >= shock_temp_threshold).astype("int")
        return data.ds.arr(mask_array, "dimensionless")

    # LOS velocity (can be positive or negative)
    def _los_velocity(field, data):
        return (data[('gas', 'velocity_x')] * z_hat[0] +
                data[('gas', 'velocity_y')] * z_hat[1] +
                data[('gas', 'velocity_z')] * z_hat[2])

    # Masked LOS velocity with clipping
    def _shock_velocity(field, data):
        mask = data[shock_mask_field]
        vel = data[los_velocity_field]
        vel_clipped = np.clip(vel, -2150, 2150)  # <-- Clip here
        return mask * vel_clipped


    if shock_mask_field not in ds.field_list:
        ds.add_field(shock_mask_field, sampling_type="cell",
                     function=_shock_mask, units="dimensionless")

    if los_velocity_field not in ds.field_list:
        ds.add_field(los_velocity_field, sampling_type="cell",
                     function=_los_velocity, units="km/s")

    if shock_velocity_field not in ds.field_list:
        ds.add_field(shock_velocity_field, sampling_type="cell",
                     function=_shock_velocity, units="km/s")

    return shock_velocity_field

# NOTE: You have two functions named `find_shock_bounds`. Python will only use the second one.
# I am leaving this as-is since you didn't ask to debug it, but you may want to review it.
def find_shock_bounds(sp, shock_velocity_field):
    """
    Find symmetric bounds for shock velocity centered at 0.
    """
    all_vels = sp[shock_velocity_field]
    if all_vels.size == 0:
        return (-1, 1), yt.YTQuantity(1, 'km/s'), yt.YTQuantity(-1, 'km/s')

    max_abs_vel = np.max(np.abs(all_vels))
    bound = max_abs_vel.to('km/s').v if hasattr(max_abs_vel, "to") else max_abs_vel
    return (-bound, bound), yt.YTQuantity(bound, 'km/s'), yt.YTQuantity(-bound, 'km/s')



def find_shock_bounds(sp, shock_velocity_field):
    """
    Find min/max bounds for shock velocity (linear scale).

    Parameters
    ----------
    sp : yt.data_objects.data_containers.YTRegion
        Data container (e.g., sphere) to search in.
    shock_velocity_field : tuple
        Field name tuple for the shock velocity.

    Returns
    -------
    bounds : tuple
        (min, max) bounds in linear scale.
    max_vel : YTQuantity
        Maximum shock velocity found.
    min_vel_nonzero : YTQuantity
        Minimum non-zero shock velocity found.
    """
    all_shock_vels = sp[shock_velocity_field]
    nonzero_vels = all_shock_vels[all_shock_vels > 0]

    if nonzero_vels.size > 0:
        min_vel_nonzero = nonzero_vels.min()
        max_vel = all_shock_vels.max()
        bounds = (min_vel_nonzero.to('km/s').v,
                  max_vel.to('km/s').v)
    else:
        # Fallback if no shocks present
        min_vel_nonzero = yt.YTQuantity(0, 'km/s')
        max_vel = yt.YTQuantity(0, 'km/s')
        bounds = (0.0, 1.0)

    return bounds, max_vel, min_vel_nonzero


def alpha_func(vals, min_val, max_val, slope=2.0, early_peak=0.6):
    """
    Alpha rises from center to edges with adjustable slope and early saturation.

    slope > 1   -> steeper rise, more focus on extremes
    slope = 1   -> linear slope
    slope < 1   -> flatter
    early_peak  -> fraction of the distance from center where alpha reaches 1.0
                   (e.g., 0.6 means 60% of the way to the edge is already max alpha)
    """
    # Normalize to 0..1
    norm = (vals - min_val) / (max_val - min_val)

    # Distance from center (0 at center, 1 at edges)
    dist = np.abs(norm - 0.5) * 2

    # Rescale so that early_peak distance maps to 1.0 alpha
    dist_scaled = dist / early_peak

    # Apply slope
    alpha = (dist_scaled ** slope)

    # Clip to max=1.0 so it stays saturated at the edges
    alpha = np.clip(alpha, 0, 1)

    return alpha


# Sets up the source
def setup_source_properties(
    sp,
    field: tuple,
    render_bounds: tuple = None,
    tf_bounds: tuple = None,
    colormap: str = None,
    alpha_function=None,
    use_log_space: bool = False
):
    """
    Setup volume rendering source with symmetric transfer function for signed velocities.
    """
    sc = yt.create_scene(sp, field=field, lens_type='perspective')
    source = sc[0]
    source.set_log(use_log_space)

    vmin, vmax = render_bounds
    bound = max(abs(vmin), abs(vmax))
    symmetric_bounds = (-bound, bound)

    tf = yt.ColorTransferFunction(symmetric_bounds)

    # Symmetric mapping (negative to positive) with optional alpha scaling
    tf.map_to_colormap(
        symmetric_bounds[0],
        symmetric_bounds[1],
        colormap=colormap,
        scale_func=alpha_function
    )

    source.tfh.tf = tf
    source.tfh.bounds = symmetric_bounds
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


def plot_velocity_histogram(sp, shock_velocity_field, mask_field=None, bins=100, range=None, save_path=None):
    vel = sp[shock_velocity_field].to('km/s').v
    if mask_field:
        mask = sp[mask_field].v.astype(bool)
        vel = vel[mask]  # Keep only masked cells

    plt.figure(figsize=(6,4))
    plt.hist(vel, bins=bins, range=range, histtype='step', color='k')
    plt.xlabel("Shock Velocity (km/s)")
    plt.ylabel("Cell Count")
    plt.title("Shock Velocity Distribution")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def save_and_prep_transfer_function(
        save_location: str,
        subplot_cords: tuple,
        title: str,
        p_field=None,
        source=None,
        save_as_png = False,
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

    # Save transfer function plot as PDF
    source.tfh.plot(save_location, profile_field=p_field)

    if save_as_png:
        img = mpimg.imread(save_location)
    else:
        # Convert first page of PDF to image (PIL format)
        images = convert_from_path(save_location, dpi=200)
        img = images[0]

    # Show in matplotlib
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')



def save_and_prep_render(
    save_location: str,
    subplot_cords: tuple,
    title: str,
    contrast: float,
    scene,
    colormap,
    bounds,
    save_as_png = False
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
    
    img = scene.render()

    # Composite alpha over white background if alpha channel exists
    if img.shape[2] == 4:
        rgb = img[..., :3]
        alpha = img[..., 3:4]
        img = rgb * alpha + (1 - alpha) * 1.0  # Composite over white

    # Normalize and apply gamma correction
    img = np.clip(img / img.max(), 0, 1)
    gamma = 1.0 / contrast
    img = img ** gamma
    
    bound = max(abs(bounds[0]), abs(bounds[1]))
    norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    plt.imshow(img, origin='lower', cmap=colormap, norm=norm)

    plt.axis('off')
    plt.title(title)
    plt.colorbar(label = "LOS Velocity (km/s)")
    plt.savefig(save_location, bbox_inches='tight', pad_inches=0)


# --- Main Rendering Function ---
def main(
        data: str = DEFAULT_PARAMETERS['data_location'],
        save_folder: str = DEFAULT_PARAMETERS['save_path'],
        sphere_radius: float = DEFAULT_PARAMETERS['sphere_radius'],
        sphere_radius_units: str = DEFAULT_PARAMETERS['sphere_radius_units'],
        temp_threshold_value: float = DEFAULT_PARAMETERS['temp_threshold_value'],
        temp_threshold_unit: str = DEFAULT_PARAMETERS['temp_threshold_unit'],
        colormap: str = DEFAULT_PARAMETERS['colormap'],
        contrast: float = DEFAULT_PARAMETERS['contrast'],
        resolution: int = DEFAULT_PARAMETERS['resolution'],
        is_animation: bool = DEFAULT_PARAMETERS['is_animation'],
        frame_number: int = DEFAULT_PARAMETERS['frame_number']
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


    # Build filenames
    if frame_number is not None:
        tf_file = os.path.join(save_folder, f"transfer_function_{frame_number:04d}.png")
        render_file = os.path.join(save_folder, f"shock_render_{frame_number:04d}.png")
    else:
        tf_file = os.path.join(save_folder, "transfer_function.pdf")
        render_file = os.path.join(save_folder, "shock_render.pdf")


    # Use line of sight as the camera position vector
    L = np.asarray([0.94, -0.10, 0.31])
    L = L / np.linalg.norm(L)
    global z_hat
    z_hat = L  # Set the global line-of-sight unit vector

    # --- 1. Define Shock Fields ---
    shock_velocity_field = create_shock_field(ds, temp_threshold_value, temp_threshold_unit)
    
    # --- 2. Create Data Sphere and Get Bounds ---
    c = find_CoM(dataset=ds)
    radius = ds.quan(sphere_radius, sphere_radius_units)
    sp = ds.sphere(c, radius)

    # Example use
    if not is_animation:
        plot_velocity_histogram(sp, shock_velocity_field, mask_field=('gas', 'shock_temperature_mask'), bins=200)

    linear_bounds, max_shock_vel, min_shock_vel_nonzero = find_shock_bounds(sp, shock_velocity_field)
    tf_bounds = linear_bounds

    # Force bounds to the clipped range
    linear_bounds = (-2150, 2150)
    tf_bounds = linear_bounds

    # --- 3. Setup Render ---
    if max_shock_vel > 0 and min_shock_vel_nonzero > 0:
        tf_bounds = (min_shock_vel_nonzero.v, max_shock_vel.v)
    else:
        # Fallback bounds if no velocity data
        print("Warning: No valid shock velocities found, using fallback tf_bounds (0.1, 1.0)")
        tf_bounds = (0.1, 1.0)

    def alpha_scale_func(val, min_val, max_val):
        return alpha_func(val, min_val, max_val, slope=0.75, early_peak=0.30)

    sc, source = setup_source_properties(sp=sp, field=shock_velocity_field, render_bounds=linear_bounds, tf_bounds=tf_bounds, colormap=colormap, alpha_function=alpha_scale_func, use_log_space=False)

    N = np.array([L[1], -L[0], 0])
    setup_camera(sc, L, N, resolution)

    # --- 4. Visualization ---
    plt.figure(figsize=(10, 5), dpi=300)

    # Save plots and saves the plots as PDF
    save_and_prep_transfer_function(save_location = tf_file, subplot_cords = (1, 2, 1), title = "Shock Velocity TF", p_field = shock_velocity_field, source = source, save_as_png=is_animation)
    save_and_prep_render(save_location = render_file, subplot_cords = (1, 2, 2), title = "Shock Velocity Map (>12 keV)", contrast = contrast, scene=sc, colormap = colormap, bounds = linear_bounds, save_as_png=is_animation)

    plt.tight_layout()
    if not is_animation:
        plt.show()


if __name__ == "__main__":
    # Single Image Render
    #main()
    
    # === EDITED SECTION START ===

    # Animation frames Render
    start_frame = 1      # Your desired starting frame
    end_frame = 125       # The frame to end on (loop goes to end_frame - 1)
    step = 2

    print(f"Starting animation render from frame {start_frame} to {end_frame - 1}.")
    
    # Loop from your specified start frame (60) to the end of your data (419)
    for i in range(start_frame, end_frame, step): #Data_000000 to Data_000419
        
    
        print(f"--- Processing frame: {i} ---") 
        
        main(data=f"sims_data/R1.5_v2400_b250/Data_{i:06d}", 
             save_folder=f"ethan/10kevframes/", 
             is_animation=True, 
             frame_number=i)

    print("--- Animation rendering complete. ---")
    
  
