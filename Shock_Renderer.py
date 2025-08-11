"""
Volume Rendering of Shock Velocities using yt
----------------------------------------------
This script generates 3D volume renderings of shock velocity fields
from simulation datasets. It can produce a single render or an animation.

Key features:
- Customizable camera position, colormap, and rendering parameters
- Optional animation with step control (e.g., render every 2 frames)
- Transfer function visualization

Author: Shawn Cheng, Ethan Tang, Michaela Lau
"""

# --- Imports ---
import os
import glob
import numpy as np
import yt
from yt.visualization.volume_rendering.render_source import VolumeSource
from unyt import unyt_array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from pdf2image import convert_from_path
import cmasher as cmr  # Extra colormaps


# --- Default Parameters ---
DEFAULT_PARAMETERS = {
    "data_path": "sims_data/*/Data_*=",  # Single file or glob pattern
    "save_path": "",                     # Output directory ("" means current dir)
    "sphere_radius": 3,                  # Radius of spherical selection (in Mpc)
    "sphere_radius_units": "Mpc",
    "temp_threshold_value": 12,          # Temperature threshold for shock mask (keV)
    "temp_threshold_unit": "keV",
    "camera_position": [0.94, -0.10, 0.31],  # Camera LOS vector
    "colormap": "cmr.prinsenvlag_r",    # Colormap for rendering
    "bounds": (-2150, 2150),             # Velocity clipping bounds (km/s)
    "contrast": 1.4,                     # Gamma contrast for render
    "resolution": 256,                   # Image resolution (px)
    "is_animation": False,               # Enable animation mode
    "frame_number": None,                # Frame index for single mode (None → no number in filename)
    "animation_frame_range": (74, 76),  # Frame start/end for animation
    "animation_step": 1,                 # Render every Nth frame in animation
    "render_title": "Shock Velocity Map (>12 keV)",
    "tf_title": "Shock Velocity Transfer Function",
    "alpha_func_slope": 0.75,            # Controls transfer function alpha slope
    "alpha_func_peak": 0.30,             # Early peak position for alpha function
    "display_renders": True,             # Pauses code and displays results 
}


# --- Helper Functions ---

def find_CoM(dataset):
    """Return the center-of-mass position as a unyt_array."""
    ad = dataset.all_data()
    # Mean of particle positions along each axis
    com_x = ad.mean(('particle_position_x'))
    com_y = ad.mean(('particle_position_y'))
    com_z = ad.mean(('particle_position_z'))
    return unyt_array([com_x, com_y, com_z], com_x.units)


def create_shock_field(ds, temp_threshold_value, temp_threshold_unit, z_hat, bounds):
    """
    Add derived fields for shock detection and velocity.
    Returns the field name tuple for shock velocity.
    """
    shock_temp_threshold = ds.quan(temp_threshold_value, temp_threshold_unit).to("K", equivalence="thermal")

    shock_mask_field = ('gas', 'shock_temperature_mask')
    los_velocity_field = ('gas', 'los_velocity')
    shock_velocity_field = ('gas', 'shock_velocity')

    # Shock mask: cells with temperature >= threshold
    def _shock_mask(field, data):
        temp = data[('gas', 'temperature')]
        return data.ds.arr((temp >= shock_temp_threshold).astype("int"), "dimensionless")

    # Line-of-sight velocity projection
    def _los_velocity(field, data):
        return (data[('gas', 'velocity_x')] * z_hat[0] +
                data[('gas', 'velocity_y')] * z_hat[1] +
                data[('gas', 'velocity_z')] * z_hat[2])

    # Shock velocity masked & clipped to bounds
    def _shock_velocity(field, data):
        mask = data[shock_mask_field]
        vel = data[los_velocity_field]
        return mask * np.clip(vel, bounds[0], bounds[1])

    # Add fields if not already present
    if shock_mask_field not in ds.field_list:
        ds.add_field(shock_mask_field, sampling_type="cell", function=_shock_mask, units="dimensionless")
    if los_velocity_field not in ds.field_list:
        ds.add_field(los_velocity_field, sampling_type="cell", function=_los_velocity, units="km/s")
    if shock_velocity_field not in ds.field_list:
        ds.add_field(shock_velocity_field, sampling_type="cell", function=_shock_velocity, units="km/s")

    return shock_velocity_field


def find_shock_bounds(sp, shock_velocity_field):
    """Return (max_velocity, min_nonzero_velocity) from a sphere object."""
    all_shock_vels = sp[shock_velocity_field]
    nonzero_vels = all_shock_vels[all_shock_vels > 0]

    if nonzero_vels.size > 0:
        min_vel_nonzero = nonzero_vels.min()
        max_vel = all_shock_vels.max()
    else:
        print("Warning: No valid shock velocities found, using fallback bounds (0.1, 1.0 km/s)")
        min_vel_nonzero = yt.YTQuantity(0.1, 'km/s')
        max_vel = yt.YTQuantity(1.0, 'km/s')

    return max_vel, min_vel_nonzero


def alpha_func(vals, min_val, max_val, slope, early_peak):
    """
    Generate alpha values for transfer function mapping.
    `early_peak` shifts transparency peak earlier.
    """
    norm = (vals - min_val) / (max_val - min_val)
    dist = np.abs(norm - 0.5) * 2
    alpha = (dist / early_peak) ** slope
    return np.clip(alpha, 0, 1)


def setup_source_properties(sp, field, render_bounds, colormap, alpha_function):
    """Set up a yt scene and color transfer function."""
    sc = yt.create_scene(sp, field=field, lens_type='perspective')
    source = sc[0]
    source.set_log(False)  # using linear scale here

    bound = max(abs(render_bounds[0]), abs(render_bounds[1]))
    symmetric_bounds = (-bound, bound)

    tf = yt.ColorTransferFunction(symmetric_bounds)
    tf.map_to_colormap(symmetric_bounds[0], symmetric_bounds[1], colormap=colormap, scale_func=alpha_function)

    source.tfh.tf = tf
    source.tfh.bounds = symmetric_bounds
    return sc, source


def setup_camera(sc, position, north_vector, resolution):
    """Configure camera properties."""
    cam = sc.camera
    cam.position = position
    cam.north_vector = north_vector
    cam.resolution = (resolution, resolution)


def make_output_filename(save_path, prefix, frame_number, save_as_pdf):
    """
    Construct output filename.
    If `frame_number` is None → omit frame number.
    """
    ext = ".pdf" if save_as_pdf else ".png"
    if frame_number is not None:
        filename = f"{prefix}_{frame_number:06d}{ext}"
    else:
        filename = f"{prefix}{ext}"
    return os.path.join(save_path, filename)


# --- Core Rendering ---
def render_frame(file_path, frame_number, save_as_pdf, params):
    """Render one frame and its transfer function."""
    print(f"Rendering: {file_path} (frame={frame_number})")

    ds = yt.load(file_path)

    # Normalize LOS vector
    L = np.asarray(params["camera_position"])
    L /= np.linalg.norm(L)
    z_hat = L

    # Create derived field
    shock_velocity_field = create_shock_field(
        ds,
        params["temp_threshold_value"],
        params["temp_threshold_unit"],
        z_hat,
        params["bounds"]
    )

    # Select sphere region around center of mass
    center = find_CoM(ds)
    sp = ds.sphere(center, ds.quan(params["sphere_radius"], params["sphere_radius_units"]))

    # Find bounds for TF scaling
    max_vel, min_vel = find_shock_bounds(sp, shock_velocity_field)
    tf_bounds = (min_vel.v, max_vel.v)

    # Alpha scaling function for transfer function
    def alpha_scale_func(val, min_val, max_val):
        return alpha_func(val, min_val, max_val,
                          slope=params["alpha_func_slope"],
                          early_peak=params["alpha_func_peak"])

    # Setup rendering source and transfer function
    sc, source = setup_source_properties(sp, shock_velocity_field,
                                         params["bounds"], params["colormap"], alpha_scale_func)

    # Configure camera with north vector perpendicular to LOS
    north_vector = np.array([L[1], -L[0], 0])
    setup_camera(sc, L, north_vector, params["resolution"])

    # Determine save directory
    save_path = params["save_path"] or os.getcwd()
    if not os.path.isdir(save_path):
        raise FileNotFoundError(f"Save path '{save_path}' does not exist.")

    # Create output filenames for transfer function and render
    tf_file = make_output_filename(save_path, "transfer_function", frame_number, save_as_pdf)
    render_file = make_output_filename(save_path, "shock_render", frame_number, save_as_pdf)

    # Plot and save results
    plt.figure(figsize=(10, 5), dpi=300, constrained_layout=True)

    # Plot transfer function and load it as image for subplot
    source.tfh.plot(tf_file, profile_field=shock_velocity_field)
    img = convert_from_path(tf_file, dpi=200)[0] if save_as_pdf else mpimg.imread(tf_file)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(params["tf_title"])
    plt.axis('off')

    # Render volume and prepare image with alpha compositing and contrast
    img = sc.render()
    rgb, alpha = img[..., :3], img[..., 3:4]
    img = np.clip((rgb * alpha + (1 - alpha)), 0, 1) ** (1.0 / params["contrast"])

    # Normalize colors around zero with TwoSlopeNorm
    bound = max(abs(params["bounds"][0]), abs(params["bounds"][1]))
    norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)

    plt.subplot(1, 2, 2)
    plt.imshow(img, origin='lower', cmap=params["colormap"], norm=norm)
    plt.title(params["render_title"])
    plt.colorbar(label="LOS Velocity (km/s)")
    plt.axis('off')

    plt.savefig(render_file, bbox_inches='tight', pad_inches=0.5)

    if params["display_renders"]:
        plt.show()
    plt.close()


# --- Main ---
def main(**kwargs):
    """Main execution: decides between single render and animation."""
    params = DEFAULT_PARAMETERS.copy()
    params.update(kwargs)

    # Determine if data_path is a file or glob pattern
    if os.path.isfile(params["data_path"]):
        files = [params["data_path"]]
    else:
        files = sorted(glob.glob(params["data_path"]))

    if not files:
        raise FileNotFoundError(f"No files found for '{params['data_path']}'")

    print(f"Found {len(files)} file(s) matching pattern.")

    if params["is_animation"]:
        start, end = params["animation_frame_range"]
        step = params["animation_step"]
        for frame_index in range(start, end, step):
            render_frame(files[frame_index], frame_index, save_as_pdf=False, params=params)
    else:
        frame_index = params["frame_number"]
        if frame_index is None:
            # Render first file without frame number in filename
            render_frame(files[0], None, save_as_pdf=True, params=params)
        else:
            render_frame(files[frame_index], frame_index, save_as_pdf=True, params=params)


# --- Entry Point ---
if __name__ == "__main__":
    main()
