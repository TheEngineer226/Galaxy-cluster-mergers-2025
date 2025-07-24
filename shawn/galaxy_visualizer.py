# imports
import yt
from yt.visualization.volume_rendering.transfer_functions import MultiVariateTransferFunction, ColorTransferFunction
from yt.visualization.volume_rendering.render_source import VolumeSource
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array
import numpy as np
import cmasher as cmr
from unyt import unyt_array
import pyxsim, subprocess
import numpy as np
#import astropy.units as u
#from astropy.cosmology import FlatLambdaCDM
from unyt import unyt_array
from yt.utilities.orientation import Orientation
import os

os.environ["OMP_NUM_THREADS"] = "4"

print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("OPENBLAS_NUM_THREADS:", os.environ.get("OPENBLAS_NUM_THREADS"))

# Maps data values to alpha (opacity) values.
# This function creates a linear ramp for opacity.
def rising_alpha_func(vals, min_val, max_val):
    # Linearly map values from their range [min_val, max_val]
    # to a new range of [0.0 (transparent), 1.0 (opaque)]
    return (vals - min_val) / (max_val - min_val)

def falling_alpha_func(vals, min_val, max_val):
    # Linearly map values from their range [min_val, max_val]
    # to a new range of [0.0 (transparent), 1.0 (opaque)]
    return 1 - (vals - min_val) / (max_val - min_val)

def flat_alpha_func(vals, min_val, max_val, alpha=1.0):
    # Returns a constant alpha value for all inputs
    return np.full_like(vals, fill_value=alpha, dtype=float)

def centered_alpha_func(vals, min_val, max_val):
    mid = 0.5 * (min_val + max_val)
    half_range = 0.5 * (max_val - min_val)
    return 1.0 - np.abs(vals - mid) / half_range

def inverted_centered_alpha_func(vals, min_val, max_val):
    mid = 0.5 * (min_val + max_val)
    half_range = 0.5 * (max_val - min_val)
    return np.abs(vals - mid) / half_range

def alpha_zero_at_zero(vals, min_val, max_val):
    return np.clip(np.abs(vals) / max(abs(min_val), abs(max_val)), 0.0, 1.0)


# The parameters pass for each field
field_parameters = [
    {
        "field" : ("gas", "density"),
        # Use "percentile" to get a percentile and "value" when you want to direct put in a value for the bound. 
        # "min" gives you the smallest number and "max" gives you the biggest(smallest, largest)
        "bounds" : [("min", None), ("max", None)], 
        "use_grey_opacity" : True, # Make underdense regions appear opaque
        "use_ghost_zones" : False, # Uses interpolated data around grid boundaries to smooth out visual artifacts. But will come at the cost of performance
        "colormap" : "turbo", 
        "alpha_function" : rising_alpha_func,
        "use_log_space" : True, 
        "file_location" : "shawn/", # The path to the folder you want the file to be saved at
        "label" : "Density", 
        "sigma_clip" : 5, # Removing values that are more than N standard deviations brighter than the mean of your image. Typically, a choice of 4 to 6.
        "interpolation" : "bilinear" # "nearest" has no smoothing, "billinear" makes it smoother
    },
    {
        "field" : ("gas", "temperature"),
        "bounds" : [("min", None), ("max", None)],  
        "use_grey_opacity" : True, 
        "use_ghost_zones" : False, 
        "colormap" : "cmr.viola",
        "alpha_function" : falling_alpha_func,
        "use_log_space" : False,
        "file_location" : "shawn/", 
        "label" : "Temperature", 
        "sigma_clip" : 6, 
        "interpolation" : "bilinear"
    },
    {
        "field" : ("gamer", "signed_velocity"),
        "bounds" : [("min", None), ("max", None)],
        "use_grey_opacity" : True, 
        "use_ghost_zones" : False, 
        "colormap" : "cmr.prinsenvlag_r",
        "alpha_function" : centered_alpha_func,
        "use_log_space" : False,
        "file_location" : "shawn/", 
        "label" : "Velocity", 
        "sigma_clip" : 4, 
        "interpolation" : "bilinear"
    },
    {
        "field" : ("gas", "total_density"),
        "bounds" : [("min", None), ("max", None)],
        "use_grey_opacity" : True, 
        "use_ghost_zones" : False, 
        "colormap" : "turbo",
        "alpha_function" : rising_alpha_func,
        "use_log_space" : True,
        "file_location" : "shawn/", 
        "label" : "Total_Density", 
        "sigma_clip" : 5, 
        "interpolation" : "bilinear"
    }
]


# Dictionary of the default parameters
DEFAULT_PARAMETERS = {
     "data_location" : "sims_data/R1.5_v2400_b250/Data_000044",
     "field_list" : field_parameters,
     "subplot_cords" : (1, 2),
     "sphere_radius" :  4,
     "sphere_radius_units" : "Mpc",
     "camera_resolution" : 512,
     "interpolation" : "nearest"
}


# Returns the CoM from all the particles (DM and stars)
def find_CoM(all_data):
    com_x = all_data.mean(('io', 'particle_position_x'))
    com_y = all_data.mean(('io', 'particle_position_y'))
    com_z = all_data.mean(('io', 'particle_position_z'))
    return unyt_array([com_x, com_y, com_z], com_x.units)


# Returns the total density (combination of both gas and particles)
def _total_density(field, data):
    return data['gas', 'density'] + data['gas', 'particle_density_on_grid']


# Uses 
def define_velocity_fields():
    def _velocity_x(field, data):
        return data[("gamer", "MomX")] / data[("gamer", "Dens")]

    def _velocity_y(field, data):
        return data[("gamer", "MomY")] / data[("gamer", "Dens")]

    def _velocity_z(field, data):
        return data[("gamer", "MomZ")] / data[("gamer", "Dens")]

    def _velocity_magnitude(field, data):
        vx = data[("gamer", "velocity_x")]
        vy = data[("gamer", "velocity_y")]
        vz = data[("gamer", "velocity_z")]
        return (vx**2 + vy**2 + vz**2)**0.5

    yt.add_field(("gamer", "velocity_x"), function=_velocity_x, units="cm/s", sampling_type="cell", force_override=True)
    yt.add_field(("gamer", "velocity_y"), function=_velocity_y, units="cm/s", sampling_type="cell", force_override=True)
    yt.add_field(("gamer", "velocity_z"), function=_velocity_z, units="cm/s", sampling_type="cell", force_override=True)
    yt.add_field(("gamer", "velocity_magnitude"), function=_velocity_magnitude, units="cm/s", sampling_type="cell", force_override=True)


def define_dark_matter_velocity_fields(all_data, L, N, c):
    # configure orientation of observer
    orient = Orientation(L, north_vector=N)
    x_hat, y_hat, z_hat = orient.unit_vectors[0], orient.unit_vectors[1], orient.unit_vectors[2]

    # get x, y positions and velocities of particles from observer's perspective (uses all_data object, not sphere; change if you like)
    x_sky_all = ((all_data['particle_position_x']-c[0])*x_hat[0] + (all_data['particle_position_y']-c[1])*x_hat[1] + (all_data['particle_position_z']-c[2])*x_hat[2]).in_units('kpc')
    y_sky_all = ((all_data['particle_position_x']-c[0])*y_hat[0] + (all_data['particle_position_y']-c[1])*y_hat[1] + (all_data['particle_position_z']-c[2])*y_hat[2]).in_units('kpc')
    v_los_all = (all_data['particle_velocity_x']*z_hat[0] + all_data['particle_velocity_y']*z_hat[1] + all_data['particle_velocity_z']*z_hat[2])

    # now filter on just the dark matter particles
    particle_ids = all_data[('nbody', 'ParType')] # 2=DM, 3=stars
    x_sky = x_sky_all[particle_ids==2.]
    y_sky = y_sky_all[particle_ids==2.]
    v_los = v_los_all[particle_ids==2.].in_units('km/s').value

    def dark_matter_los_velocity(field, data):
        # Assume 'particle_velocity_x', etc. exist in the dataset
        # and that you have access to `z_hat` somehow (might need to pass it in globally or via closure)
        vx = data['particle_velocity_x']
        vy = data['particle_velocity_y']
        vz = data['particle_velocity_z']
        # Dot product with z_hat
        v_los = vx * z_hat[0] + vy * z_hat[1] + vz * z_hat[2]
        return v_los.in_units('cm/s')

    yt.add_field(("gamer", "dark_matter_velocity"), 
             function=dark_matter_los_velocity,
             units="cm/s",
             sampling_type="cell",
             force_override=True)


def signed_velocity_magnitude(field, data):
    sign = np.sign(data[("gas", "velocity_x")])  # Source of sign
    magnitude = data[("gas", "velocity_magnitude")]  # Precomputed field
    return sign * magnitude


# Sets up the source
def setup_source_properties(
    source: VolumeSource, 
    field: tuple, 
    bounds: tuple = None,
    use_grey_opacity: bool = False, 
    use_ghost_zones: bool = False, 
    colormap: str = None,
    alpha_function = flat_alpha_func,
    use_log_space: bool = True
):
    
    source.set_field(field)
    source.set_log(use_log_space)
    source.grey_opacity = use_grey_opacity
    source.set_use_ghost_zones(use_ghost_zones)
    
    if bounds:
        source.tfh.set_bounds(bounds)

    if use_log_space:
        bounds = np.log10(bounds)

    if colormap:
        tf = yt.ColorTransferFunction(bounds)

        # Uses map_to_colormap to apply the colormap and opacity scale
        tf.map_to_colormap(
            bounds[0],
            bounds[1],
            colormap=colormap,
            scale_func=alpha_function
        )

        source.tfh.tf = tf
        

# Saves the rendering and then utilizes Matplotlib to display the render 
# Saving images should pass in a VolumeSource, Transfer Functions should pass in source 
def save_and_show_img(
    file_location: str, 
    subplot_cords: tuple, 
    title: str, 
    is_transfer_function: bool, 
    scene_or_source = None, 
    profile_field: tuple = None, 
    sigma_clip: float = None, 
    interpolation: str = "none"
):
    
    if is_transfer_function:
        scene_or_source.tfh.plot(file_location, profile_field = profile_field)
    else:
        scene_or_source.save(file_location, sigma_clip = sigma_clip, render = True) # Render tells the program to re-render the scene even if a previously rendered image exists 

    render_img = mpimg.imread(file_location)
    plt.subplot(*subplot_cords)
    plt.imshow(render_img, interpolation=interpolation) # "bilinear" makes it smoother, "nearest" has no smoothing
    plt.title(title)
    plt.axis('off')


# Loads in and volume renders galaxy data, then volume renders them
def main(
    data: str = DEFAULT_PARAMETERS["data_location"], 
    field_list = DEFAULT_PARAMETERS["field_list"], 
    subplot_cords: tuple = DEFAULT_PARAMETERS["subplot_cords"],
    sphere_radius: float = DEFAULT_PARAMETERS["sphere_radius"], 
    sphere_radius_units: str = DEFAULT_PARAMETERS["sphere_radius_units"], 
    resolution: int = DEFAULT_PARAMETERS["camera_resolution"], 
):
    
    ds = yt.load(data)
    ad = ds.all_data()

    L = np.asarray([0.94, -0.10, 0.31]) / np.linalg.norm(np.asarray([0.94, -0.10, 0.31]))
    N = np.array([L[1], -L[0], 0])

    # Gets CoM from particles (DM and stars) 
    center = find_CoM(ad)

    # Define velocity fields
    define_velocity_fields()
    ds.add_field(("gamer", "signed_velocity"), function=signed_velocity_magnitude, units="cm/s", sampling_type="cell")
    #define_dark_matter_velocity_fields(ad, L, N, center)

    # Look at gas/particles within a sphere (not full simulation domain) 
    radius = (sphere_radius, sphere_radius_units)
    sp = ds.sphere(center, radius) # center = CoM defined above, Tuple is required here as parameter

    ds.add_field(('gas', 'total_density'), _total_density, units='g/cm**3', sampling_type='local')
    
    #total_density_in_sphere = sp[('gas', 'total_density')]  # Will probably be used in the future but not sure just yet
    #temperature_in_sphere = sp[('gas', 'temperature')]  # Same as above

    #sc = yt.create_scene(sp, field=None, lens_type="perspective")
    sc = yt.create_scene(sp, field=None, lens_type="plane-parallel")
    source = sc[0]

    # Set the camera to look at the region of interest
    cam = sc.camera
    cam.position = L
    cam.north_vector = N
    cam.resolution = (resolution, resolution) # can lower resolution to 128 to prioritize faster rendering. 512 is the standard quality
    
    plt.figure(figsize=(10, 5))

    for i, field_params in enumerate(field_list):
        bounds = []
        for value in field_params["bounds"]:
            if value[0] == "min":
                bounds.append(ds.find_min(field_params["field"])[0].to_value())

            elif value[0] == "max":
                bounds.append(ds.find_max(field_params["field"])[0].to_value())

            elif value[0] == "percentile":
                bounds.append(np.percentile(sp[field_params["field"]], [value[1]])[0].to_value())

            elif value[0] == "value":
                bounds.append(value[1])

        print(bounds)

        setup_source_properties(
            source = source, 
            field = field_params["field"], 
            bounds = bounds,
            use_grey_opacity = field_params["use_grey_opacity"], 
            use_ghost_zones = field_params["use_ghost_zones"], 
            colormap = field_params["colormap"],
            alpha_function = field_params["alpha_function"],
            use_log_space = field_params["use_log_space"]
        )

        save_and_show_img(
            file_location = field_params["file_location"] + field_params["label"].lower() + "_transfer.png", 
            subplot_cords = (subplot_cords[0], subplot_cords[1], i * 2 + 2), 
            title = field_params["label"] + " Transfer Function", 
            is_transfer_function = True,
            scene_or_source = source,
            profile_field = field_params["field"]
        )

        save_and_show_img(
            file_location = field_params["file_location"] + field_params["label"].lower() + "_image.png", 
            subplot_cords = (subplot_cords[0], subplot_cords[1], i * 2 + 1), 
            title = field_params["label"] + " Image", 
            is_transfer_function = False,
            scene_or_source = sc,
            sigma_clip = field_params["sigma_clip"], 
            interpolation = field_params["interpolation"] 
        )


    # Ensures labels and titles don't overlap
    plt.tight_layout() 
    plt.show() 


if __name__ == "__main__":
        main(sphere_radius= 3.5, subplot_cords = (3, 4), resolution= 128)
        #main()
