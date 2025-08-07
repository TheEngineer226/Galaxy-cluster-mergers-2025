
# --- Imports ---
import os
import numpy as np
import yt
from yt.visualization.volume_rendering.render_source import VolumeSource
from unyt import unyt_array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from pdf2image import convert_from_path
import cmasher as cmr


# --- Default Parameters ---
DEFAULT_PARAMETERS = {
    "data_location": "sims_data/R1.5_v2400_b250/Data_000074",  # Path to the simulation dataset
    "save_path": "shawn/",  # Output directory for renders and transfer functions
    "sphere_radius": 3,  # Radius for the spherical data selection (in Mpc)
    "sphere_radius_units": "Mpc",
    "temp_threshold_value": 12,  # Temperature threshold for shocks (in keV)
    "temp_threshold_unit": "keV",
    "camera_position": "los",  # Camera position vector or 'los' to use the default line-of-sight
    "colormap": "cmr.prinsenvlag_r",  # Colormap for visualization
    "bounds": (-2150, 2150),  # Velocity clipping bounds (in km/s)
    "contrast": 1.4,  # Gamma contrast for rendering
    "resolution": 256,  # Image resolution (pixels)
    "is_animation": False,  # Flag to enable animation mode
    "frame_number": None,  # Optional frame number for saving outputs
    "animation_frame_range": (20, 130),  # Range of frames for animation
    "render_title": "Shock Velocity Map (>12 keV)",
    "tf_title": "Shock Velocity Transfer Function"
}


# --- Helper Functions ---
def find_CoM(dataset):
    """
    Compute the center of mass (CoM) from all particles (DM and stars).
    """
    ad = dataset.all_data()
    com_x = ad.mean(('particle_position_x'))
    com_y = ad.mean(('particle_position_y'))
    com_z = ad.mean(('particle_position_z'))
    return unyt_array([com_x, com_y, com_z], com_x.units)


def create_shock_field(ds, temp_threshold_value, temp_threshold_unit, z_hat, bounds):
    """
    Adds fields to the dataset related to shock structure and LOS velocity.
    """
    shock_temp_threshold = ds.quan(temp_threshold_value, temp_threshold_unit).to("K", equivalence="thermal")

    shock_mask_field = ('gas', 'shock_temperature_mask')
    los_velocity_field = ('gas', 'los_velocity')
    shock_velocity_field = ('gas', 'shock_velocity')

    def _shock_mask(field, data):
        temp = data[('gas', 'temperature')]
        return data.ds.arr((temp >= shock_temp_threshold).astype("int"), "dimensionless")

    def _los_velocity(field, data):
        return (data[('gas', 'velocity_x')] * z_hat[0] +
                data[('gas', 'velocity_y')] * z_hat[1] +
                data[('gas', 'velocity_z')] * z_hat[2])

    def _shock_velocity(field, data):
        mask = data[shock_mask_field]
        vel = data[los_velocity_field]
        return mask * np.clip(vel, bounds[0], bounds[1])

    if shock_mask_field not in ds.field_list:
        ds.add_field(shock_mask_field, sampling_type="cell", function=_shock_mask, units="dimensionless")

    if los_velocity_field not in ds.field_list:
        ds.add_field(los_velocity_field, sampling_type="cell", function=_los_velocity, units="km/s")

    if shock_velocity_field not in ds.field_list:
        ds.add_field(shock_velocity_field, sampling_type="cell", function=_shock_velocity, units="km/s")

    return shock_velocity_field


def find_shock_bounds(sp, shock_velocity_field):
    """
    Compute min/max bounds for shock velocity in linear scale.
    """
    all_shock_vels = sp[shock_velocity_field]
    nonzero_vels = all_shock_vels[all_shock_vels > 0]

    if nonzero_vels.size > 0:
        min_vel_nonzero = nonzero_vels.min()
        max_vel = all_shock_vels.max()
    else:
        print("Warning: No valid shock velocities found, using fallback bounds (0.1, 1.0)")
        min_vel_nonzero = yt.YTQuantity(0.1, 'km/s')
        max_vel = yt.YTQuantity(1.0, 'km/s')

    return max_vel, min_vel_nonzero


def alpha_func(vals, min_val, max_val, slope=2.0, early_peak=0.6):
    """
    Compute alpha (opacity) values based on distance from center of value range.
    """
    norm = (vals - min_val) / (max_val - min_val)
    dist = np.abs(norm - 0.5) * 2
    alpha = (dist / early_peak) ** slope
    return np.clip(alpha, 0, 1)


def setup_source_properties(sp, field, render_bounds, tf_bounds, colormap, alpha_function, use_log_space=False):
    """
    Initialize the yt volume rendering source and apply a transfer function.
    """
    sc = yt.create_scene(sp, field=field, lens_type='perspective')
    source = sc[0]
    source.set_log(use_log_space)

    bound = max(abs(render_bounds[0]), abs(render_bounds[1]))
    symmetric_bounds = (-bound, bound)

    tf = yt.ColorTransferFunction(symmetric_bounds)
    tf.map_to_colormap(symmetric_bounds[0], symmetric_bounds[1], colormap=colormap, scale_func=alpha_function)

    source.tfh.tf = tf
    source.tfh.bounds = symmetric_bounds
    return sc, source


def setup_camera(sc, position, north_vector, resolution):
    """
    Configure the yt camera for rendering.
    """
    cam = sc.camera
    cam.position = position
    cam.north_vector = north_vector
    cam.resolution = (resolution, resolution)


def save_and_prep_transfer_function(save_location, subplot_cords, title, p_field=None, source=None, save_as_png=False):
    """
    Render and display the transfer function.
    """
    plt.subplot(*subplot_cords)
    source.tfh.plot(save_location, profile_field=p_field)
    img = mpimg.imread(save_location) if save_as_png else convert_from_path(save_location, dpi=200)[0]
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')


def save_and_prep_render(save_location, subplot_cords, title, contrast, scene, colormap, bounds, save_as_png=False):
    """
    Render and display the volume rendering result.
    """
    plt.subplot(*subplot_cords)
    img = scene.render()

    if img.shape[2] == 4:
        rgb = img[..., :3]
        alpha = img[..., 3:4]
        img = rgb * alpha + (1 - alpha) * 1.0

    img = np.clip(img / img.max(), 0, 1)
    img = img ** (1.0 / contrast)

    bound = max(abs(bounds[0]), abs(bounds[1]))
    norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    plt.imshow(img, origin='lower', cmap=colormap, norm=norm, interpolation="nearest")
    plt.axis('off')
    plt.title(title)
    plt.colorbar(label="LOS Velocity (km/s)")
    plt.savefig(save_location, bbox_inches='tight', pad_inches=0)


# --- Main Function ---
def main(
    data=DEFAULT_PARAMETERS['data_location'],
    save_folder=DEFAULT_PARAMETERS['save_path'],
    sphere_radius=DEFAULT_PARAMETERS['sphere_radius'],
    sphere_radius_units=DEFAULT_PARAMETERS['sphere_radius_units'],
    temp_threshold_value=DEFAULT_PARAMETERS['temp_threshold_value'],
    temp_threshold_unit=DEFAULT_PARAMETERS['temp_threshold_unit'],
    camera_position=DEFAULT_PARAMETERS['camera_position'],
    colormap=DEFAULT_PARAMETERS['colormap'],
    bounds=DEFAULT_PARAMETERS['bounds'],
    contrast=DEFAULT_PARAMETERS['contrast'],
    resolution=DEFAULT_PARAMETERS['resolution'],
    is_animation=DEFAULT_PARAMETERS['is_animation'],
    frame_number=DEFAULT_PARAMETERS['frame_number'],
    animation_frame_range=DEFAULT_PARAMETERS['animation_frame_range'],
    render_title=DEFAULT_PARAMETERS['render_title'],
    tf_title=DEFAULT_PARAMETERS['tf_title']
):
    """
    Load data, generate fields, perform volume rendering, and save outputs.
    """
    L = np.asarray([0.94, -0.10, 0.31])
    L = L / np.linalg.norm(L)
    z_hat = L

    if is_animation:
        for i in range(*animation_frame_range):
            print(f"Rendering frame {i}...")
            main(
                data=f"sims_data/R1.5_v2400_b250/Data_{i:06d}",
                save_folder=save_folder,
                sphere_radius=sphere_radius,
                sphere_radius_units=sphere_radius_units,
                temp_threshold_value=temp_threshold_value,
                temp_threshold_unit=temp_threshold_unit,
                camera_position=camera_position,
                colormap=colormap,
                bounds=bounds,
                contrast=contrast,
                resolution=resolution,
                is_animation=False,
                frame_number=i
            )
        return

    ds = yt.load(data)

    tf_file = os.path.join(save_folder, f"transfer_function_{frame_number:04d}.png" if frame_number is not None else "transfer_function.pdf")
    render_file = os.path.join(save_folder, f"shock_render_{frame_number:04d}.png" if frame_number is not None else "shock_render.pdf")

    shock_velocity_field = create_shock_field(ds, temp_threshold_value, temp_threshold_unit, z_hat, bounds)
    center = find_CoM(ds)
    sp = ds.sphere(center, ds.quan(sphere_radius, sphere_radius_units))

    max_vel, min_vel = find_shock_bounds(sp, shock_velocity_field)
    tf_bounds = (min_vel.v, max_vel.v)

    def alpha_scale_func(val, min_val, max_val):
        return alpha_func(val, min_val, max_val, slope=0.75, early_peak=0.30)

    sc, source = setup_source_properties(sp, shock_velocity_field, bounds, tf_bounds, colormap, alpha_scale_func)

    north_vector = np.array([L[1], -L[0], 0])
    setup_camera(sc, L if camera_position == "los" else camera_position, north_vector, resolution)

    plt.figure(figsize=(10, 5), dpi=300)
    save_and_prep_transfer_function(tf_file, (1, 2, 1), tf_title, shock_velocity_field, source, is_animation)
    save_and_prep_render(render_file, (1, 2, 2), render_title, contrast, sc, colormap, bounds, is_animation)

    plt.tight_layout()
    if not is_animation:
        plt.show()


# --- Entry Point ---
if __name__ == "__main__":
    main()
