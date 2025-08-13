"""
Volume Rendering of Shock Velocities using yt
----------------------------------------------
This script generates 3D volume renderings of shock velocity fields
from simulation datasets. It can produce either a single render or 
a sequence of frames for an animation (the movie itself is not created 
by this script).

Key features:
- Customizable camera position, colormap, and rendering parameters
- Optional frame-by-frame output for animation, with step control 
  (e.g., render every 2 frames)
- Transfer function visualization
"""


# --- Imports ---
import os
import glob
import numpy as np
import yt
from unyt import unyt_array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from pdf2image import convert_from_path
import cmasher as cmr


# --- Default Parameters ---
DEFAULT_PARAMETERS = {
    # Data handling
    "data_path": "sims_data/*/Data_*",       # "data_path": single file or glob pattern (needed for animation) 
    "save_path": "",                         # Output directory ("" means current dir)

    # Selection & thresholds
    "sphere_radius": 4,                      # Radius of spherical selection
    "sphere_radius_units": "Mpc",            # Units for sphere_radius
    "temp_threshold_value": 12,              # Temperature threshold for shock mask
    "temp_threshold_unit": "keV",            # Units for temp_threshold_value

    # Camera & rendering
    "camera_position": [0.94, -0.10, 0.31],  # Camera LOS vector
    "colormap": "cmr.prinsenvlag_r",         # Colormap for rendering
    "bounds": (-2150, 2150),                 # Velocity clipping bounds (km/s)
    "contrast": 1.4,                         # Gamma contrast for render
    "resolution": 256,                       # Image resolution

    # Animation / single render
    "frame_number": 74,                      # Frame index (None = no number in filename)
    "animation_frame_range": None,           # (start, end) or None for single render
    "animation_step": 1,                     # Render every Nth frame in animation

    # Plot titles
    "render_title": "Shock Velocity Map (>12 keV)", 
    "tf_title": "Shock Velocity Transfer Function",

    # Transfer function alpha shape
    "alpha_func_slope": 0.75,                # Controls alpha slope
    "alpha_func_peak": 0.30,                 # Early peak position for alpha

    # Output behavior
    "display_renders": True,                 # Pause & show results after rendering (disable for animation)
}


# --- Helper Functions ---
def find_CoM(dataset):
    """
    Compute the center-of-mass (CoM) position of the dataset.

    Parameters
    ----------
    dataset : yt.Dataset
        The loaded yt dataset to analyze.

    Returns
    -------
    unyt_array
        A length-3 unyt_array representing the (x, y, z) position of the center-of-mass
        in the same spatial units as the particle positions.
    """
    # --- Access all data in the simulation ---
    # This creates a data container with all gas cells and/or particles.
    ad = dataset.all_data()

    # --- Compute mean positions along each axis ---
    # These fields typically come from particle data (dark matter, gas, stars, etc.).
    # Using 'mean' on these fields yields the average spatial coordinate.
    com_x = ad.mean(('particle_position_x'))
    com_y = ad.mean(('particle_position_y'))
    com_z = ad.mean(('particle_position_z'))

    # --- Return the CoM as a unyt_array ---
    # We ensure it carries units by using com_x.units (same for y, z).
    return unyt_array([com_x, com_y, com_z], com_x.units)


def create_shock_field(ds, temp_threshold_value, temp_threshold_unit, z_hat, bounds):
    """
    Add derived fields for shock detection and shock line-of-sight velocity.

    Parameters
    ----------
    ds : yt.Dataset
        The loaded dataset in yt.
    temp_threshold_value : float
        The shock temperature threshold (numerical value).
    temp_threshold_unit : str
        Units for the temperature threshold (e.g., "keV").
    z_hat : array-like of length 3
        The normalized line-of-sight (LOS) direction vector.
    bounds : tuple of float
        (min_velocity, max_velocity) bounds for clipping the shock velocity.

    Returns
    -------
    tuple
        The field name tuple ('gas', 'shock_velocity').
    """

    # --- Convert temperature threshold to Kelvin ---
    # The dataset's temperature may be in Kelvin internally,
    # but here we allow specifying e.g. 12 keV and convert using thermal equivalence.
    shock_temp_threshold = ds.quan(temp_threshold_value, temp_threshold_unit).to(
        "K", equivalence="thermal"
    )

    # --- Define field name tuples ---
    # Using ('gas', field_name) makes it easy to avoid naming collisions in yt.
    shock_mask_field = ('gas', 'shock_temperature_mask')  # 1 if cell is shocked, else 0
    los_velocity_field = ('gas', 'los_velocity')          # Velocity projected along LOS
    shock_velocity_field = ('gas', 'shock_velocity')      # LOS velocity only in shocked cells

    # --- Define the derived field functions ---

    def _shock_mask(field, data):
        """
        Binary mask: 1 where cell temperature >= threshold, else 0.
        """
        temp = data[('gas', 'temperature')]
        mask_array = (temp >= shock_temp_threshold).astype("int")
        return data.ds.arr(mask_array, "dimensionless")

    def _los_velocity(field, data):
        """
        Project the 3D velocity vector onto the LOS vector z_hat.
        This yields a scalar velocity along the viewing direction.
        """
        return (data[('gas', 'velocity_x')] * z_hat[0] +
                data[('gas', 'velocity_y')] * z_hat[1] +
                data[('gas', 'velocity_z')] * z_hat[2])

    def _shock_velocity(field, data):
        """
        LOS velocity in shocked cells only.
        Non-shocked cells are set to 0 velocity.
        Values are clipped to the given bounds to remove outliers.
        """
        mask = data[shock_mask_field]
        vel = data[los_velocity_field]
        return mask * np.clip(vel, bounds[0], bounds[1])

    # --- Register the fields with yt ---
    # We only add them if they don't already exist in ds.field_list to avoid duplication.
    if shock_mask_field not in ds.field_list:
        ds.add_field(shock_mask_field, sampling_type="cell",
                     function=_shock_mask, units="dimensionless")

    if los_velocity_field not in ds.field_list:
        ds.add_field(los_velocity_field, sampling_type="cell",
                     function=_los_velocity, units="km/s")

    if shock_velocity_field not in ds.field_list:
        ds.add_field(shock_velocity_field, sampling_type="cell",
                     function=_shock_velocity, units="km/s")

    # Return the field name tuple so it can be passed directly into yt plotting/rendering functions
    return shock_velocity_field


def find_shock_bounds(sp, shock_velocity_field):
    """
    Determine the maximum and minimum non-zero shock velocities in a sphere.

    Parameters
    ----------
    sp : yt.data_objects.selection_objects.sphere.Sphere
        A spherical data container from yt.
    shock_velocity_field : tuple
        Field name tuple for the shock velocity (e.g., ('gas', 'shock_velocity')).

    Returns
    -------
    tuple
        (max_velocity, min_nonzero_velocity) as yt.YTQuantity objects.
        Units are preserved from the dataset.

    Notes
    -----
    - The minimum velocity returned ignores zero values (non-shocked cells).
    - If no non-zero values are found, fallback values are used.
    """

    # --- Extract all shock velocity values from the sphere ---
    # This is typically a 1D unyt_array of velocities for each cell in the sphere.
    all_shock_vels = sp[shock_velocity_field]

    # --- Select only the positive (non-zero) velocities ---
    # This avoids including unshocked cells, which have a value of 0 by design.
    nonzero_vels = all_shock_vels[all_shock_vels > 0]

    # --- If we have at least one shocked cell ---
    if nonzero_vels.size > 0:
        # Minimum velocity among shocked cells
        min_vel_nonzero = nonzero_vels.min()
        # Maximum velocity across all cells (shocked or not)
        max_vel = all_shock_vels.max()

    # --- If no shocked cells are found ---
    else:
        print("Warning: No valid shock velocities found, using fallback bounds (0.1, 1.0 km/s)")
        min_vel_nonzero = yt.YTQuantity(0.1, 'km/s')  # Arbitrary small positive number
        max_vel = yt.YTQuantity(1.0, 'km/s')          # Arbitrary upper bound

    return max_vel, min_vel_nonzero


def alpha_func(vals, min_val, max_val, slope, early_peak):
    """
    Generate alpha (opacity) values for use in a transfer function.

    Parameters
    ----------
    vals : array-like
        Input data values to map into alpha.
    min_val : float
        Minimum value of the input range (fully normalized to 0).
    max_val : float
        Maximum value of the input range (fully normalized to 1).
    slope : float
        Controls how sharply alpha rises towards the extremes.
        - slope > 1 : steeper rise (more contrast)
        - slope = 1 : linear change
        - slope < 1 : flatter, more gradual rise
    early_peak : float
        Fractional position (0 < early_peak <= 1) where alpha reaches 1.
        Smaller values push the high-opacity region closer to the center.

    Returns
    -------
    alpha : ndarray
        Array of opacity values in [0, 1].
    
    Notes
    -----
    - Normalization maps `vals` into a 0–1 range before applying the formula.
    - `dist` measures distance from the midpoint (0.5) so alpha is symmetric.
    - Result is clipped to [0, 1] to avoid out-of-bound alpha values.
    """

    # Normalize values to the [0, 1] range
    norm = (vals - min_val) / (max_val - min_val)

    # Distance from center (0.5), scaled to 0–1
    dist = np.abs(norm - 0.5) * 2

    # Apply early_peak scaling + slope shaping
    alpha = (dist / early_peak) ** slope

    # Ensure alpha stays between 0 and 1
    return np.clip(alpha, 0, 1)


def setup_source_properties(sp, field, render_bounds, colormap, alpha_function):
    """
    Create a yt Scene and configure the transfer function for rendering.

    Parameters
    ----------
    sp : yt.data_objects.selection_data_containers.YTSphere (or similar)
        Data selection (e.g., sphere, box) to visualize.
    field : tuple or str
        The field to render (e.g., ('gas', 'density')).
    render_bounds : tuple of float
        Value range (min, max) for the rendering. Used to set symmetric bounds.
    colormap : str
        Matplotlib colormap name to use for RGB mapping.
    alpha_function : callable
        Function of the form alpha_function(vals, min_val, max_val).
        This controls the opacity curve.

    Returns
    -------
    sc : yt.visualization.volume_rendering.scene.Scene
        The configured yt Scene.
    source : yt.visualization.volume_rendering.render_source.VolumeSource
        The source object with transfer function and bounds applied.

    Notes
    -----
    - The `max(abs(...))` step enforces symmetry around zero, useful for
      bipolar fields like velocity where negative and positive values should
      be equally visible.
    - `set_log(False)` forces a linear scale; remove if log scaling is desired.
    - `tf.map_to_colormap` applies both the colormap and the alpha mapping
      via `scale_func`.
    """
    # Create scene
    sc = yt.create_scene(sp, field=field, lens_type='perspective')
    source = sc[0]

    # Use linear scale for transfer function
    source.set_log(False)

    # Make bounds symmetric about zero
    bound = max(abs(render_bounds[0]), abs(render_bounds[1]))
    symmetric_bounds = (-bound, bound)

    # Create color transfer function
    tf = yt.ColorTransferFunction(symmetric_bounds)

    # Map full range to colormap + alpha curve
    tf.map_to_colormap(
        symmetric_bounds[0], symmetric_bounds[1],
        colormap=colormap,
        scale_func=alpha_function
    )

    # Apply to source
    source.tfh.tf = tf
    source.tfh.bounds = symmetric_bounds

    return sc, source


def setup_camera(sc, position, north_vector, resolution):
    """
    Configure the camera for a yt scene.

    Parameters
    ----------
    sc : yt.visualization.volume_rendering.scene.Scene
        The yt Scene containing the camera.
    position : array-like of float
        3D coordinates for the camera position in code units.
    north_vector : array-like of float
        Vector defining the "up" direction in the image.
    resolution : int
        Output image resolution (square).
    focus : array-like of float, optional
        Point in space for the camera to focus on.
        If None, keeps the existing camera focus.
    width : float or yt.units.yt_quantity, optional
        Width of the camera view. If None, keeps existing width.

    Notes
    -----
    - `north_vector` defines image orientation.
    - Without setting `focus`, the scene may use its default (usually dataset center).
    - `width` controls zoom; smaller width = more zoom.
    """
    cam = sc.camera
    cam.position = position
    cam.north_vector = north_vector
    cam.resolution = (resolution, resolution)


def make_output_filename(save_path, prefix, frame_number, save_as_pdf):
    """
    Construct the output filename for saved images.

    Parameters
    ----------
    save_path : str
        Directory where the file will be saved.
    prefix : str
        Base name of the file (e.g., 'transfer_function' or 'shock_render').
    frame_number : int or None
        Frame index for animations; if None, the filename will not contain a frame number.
    save_as_pdf : bool
        If True, save as .pdf; otherwise, save as .png.

    Returns
    -------
    str
        Full path to the output file.
    """
    # Decide file extension based on user preference
    ext = ".pdf" if save_as_pdf else ".png"

    # If frame_number is given → zero-pad to 6 digits
    if frame_number is not None:
        filename = f"{prefix}_{frame_number:06d}{ext}"
    else:
        # Single render without frame index
        filename = f"{prefix}{ext}"

    return os.path.join(save_path, filename)


# --- Core Rendering ---
def render_frame(
    file_path,
    frame_number,
    save_as_pdf,
    camera_position,
    temp_threshold_value,
    temp_threshold_unit,
    bounds,
    sphere_radius,
    sphere_radius_units,
    alpha_func_slope,
    alpha_func_peak,
    colormap,
    resolution,
    save_path,
    render_title,
    tf_title,
    contrast,
    display_renders,
):
    """
    Render a single frame of the shock velocity map along with its transfer function.

    Parameters
    ----------
    file_path : str
        Path to the dataset file to be rendered.
    frame_number : int or None
        Index of the frame being rendered (used for naming output files).
        If None, the output will not include a frame number in the filename.
    save_as_pdf : bool
        Whether to save the outputs as PDF (True) or PNG (False).
    camera_position : array-like of float
        3D vector specifying the camera line-of-sight direction.
    temp_threshold_value : float
        Temperature threshold value for shock mask.
    temp_threshold_unit : str
        Unit of the temperature threshold (e.g., "keV").
    bounds : tuple of float
        Min and max velocity clipping bounds (km/s).
    sphere_radius : float
        Radius of spherical region to select.
    sphere_radius_units : str
        Units of the sphere radius (e.g., "Mpc").
    alpha_func_slope : float
        Controls the slope parameter of the alpha transfer function.
    alpha_func_peak : float
        Early peak position for alpha transfer function.
    colormap : str
        Colormap name for rendering.
    resolution : int
        Image resolution (pixels).
    save_path : str
        Directory path to save output files.
    render_title : str
        Title for the volume render subplot.
    tf_title : str
        Title for the transfer function subplot.
    contrast : float
        Gamma contrast adjustment for final image.
    display_renders : bool
        Whether to display the final figure interactively.

    """
    print(f"Rendering: {file_path} (frame={frame_number})")

    ds = yt.load(file_path)

    # Normalize the camera position to get LOS unit vector
    L = np.asarray(camera_position, dtype=float)
    L /= np.linalg.norm(L)
    z_hat = L

    # Create derived shock velocity field
    shock_velocity_field = create_shock_field(
        ds,
        temp_threshold_value,
        temp_threshold_unit,
        z_hat,
        bounds
    )

    # Select sphere region around center-of-mass
    center = find_CoM(ds)
    sp = ds.sphere(center, ds.quan(sphere_radius, sphere_radius_units))

    # Find min/max shock velocity for transfer function scaling
    max_vel, min_vel = find_shock_bounds(sp, shock_velocity_field)
    tf_bounds = (min_vel.v, max_vel.v)

    # Define alpha scaling function for transfer function
    def alpha_scale_func(val, min_val, max_val):
        return alpha_func(val, min_val, max_val,
                          slope=alpha_func_slope,
                          early_peak=alpha_func_peak)

    # Setup yt scene and transfer function
    sc, source = setup_source_properties(sp, shock_velocity_field, bounds, colormap, alpha_scale_func)

    # Configure camera with north vector perpendicular to LOS
    north_vector = np.array([L[1], -L[0], 0])
    setup_camera(sc, L, north_vector, resolution)

    # Verify save path exists
    save_dir = save_path or os.getcwd()
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"Save path '{save_dir}' does not exist.")

    # Generate output filenames
    tf_file = make_output_filename(save_dir, "transfer_function", frame_number, save_as_pdf)
    render_file = make_output_filename(save_dir, "shock_render", frame_number, save_as_pdf)

    # Create figure with 2 subplots
    plt.figure(figsize=(10, 5), dpi=300, constrained_layout=True)

    # Plot transfer function and load image
    source.tfh.plot(tf_file, profile_field=shock_velocity_field)
    if save_as_pdf:
        img = convert_from_path(tf_file, dpi=200)[0]
    else:
        img = mpimg.imread(tf_file)

    # globally smaller text
    plt.rcParams.update({'font.size': 8})  

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(tf_title)
    plt.axis('off')

    img = sc.render()

    # Composite alpha over white background if alpha channel exists
    if img.shape[2] == 4:
        rgb = img[..., :3]
        alpha = img[..., 3:4]
        img = rgb * alpha + (1 - alpha) * 1.0  # Composite over white

    # Normalize image intensity to [0, 1]
    img = np.clip(img / img.max(), 0, 1)

    # Apply gamma correction for contrast
    img = img ** (1.0 / contrast)

    # Normalize colors symmetrically around zero
    bound = max(abs(bounds[0]), abs(bounds[1]))
    norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)

    plt.subplot(1, 2, 2)
    plt.imshow(img, origin='lower', cmap=colormap, norm=norm)
    plt.title(render_title)
    plt.colorbar(label="LOS Velocity (km/s)")
    plt.axis('off')

    # Save final figure
    plt.savefig(render_file, bbox_inches='tight', pad_inches=0.5)

    # Display if requested
    if display_renders:
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

    def call_render(file_path, frame_number, save_as_pdf):
        render_frame(
            file_path=file_path,
            frame_number=frame_number,
            save_as_pdf=save_as_pdf,
            camera_position=params["camera_position"],
            temp_threshold_value=params["temp_threshold_value"],
            temp_threshold_unit=params["temp_threshold_unit"],
            bounds=params["bounds"],
            sphere_radius=params["sphere_radius"],
            sphere_radius_units=params["sphere_radius_units"],
            alpha_func_slope=params["alpha_func_slope"],
            alpha_func_peak=params["alpha_func_peak"],
            colormap=params["colormap"],
            resolution=params["resolution"],
            save_path=params["save_path"],
            render_title=params["render_title"],
            tf_title=params["tf_title"],
            contrast=params["contrast"],
            display_renders=params["display_renders"],
        )

    # Animation mode if animation_frame_range is set
    if params["animation_frame_range"] is not None:
        start, end = params["animation_frame_range"]
        step = params["animation_step"]
        for frame_index in range(start, end, step):
            if frame_index >= len(files):
                print(f"Skipping frame {frame_index}: index out of range.")
                continue
            print(f"Rendering animation frame {frame_index}")
            call_render(files[frame_index], frame_index, save_as_pdf=False)

    else:
        frame_index = params["frame_number"]
        if frame_index is None:
            print(f"Rendering first file: {files[0]}")
            call_render(files[0], None, save_as_pdf=True)
        else:
            if frame_index >= len(files):
                raise IndexError(f"frame_number {frame_index} is out of range (0-{len(files)-1}).")
            print(f"Rendering single frame {frame_index}")
            call_render(files[frame_index], frame_index, save_as_pdf=True)


# --- Entry Point ---
if __name__ == "__main__":
    main()
