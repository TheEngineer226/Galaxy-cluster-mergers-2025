# imports
import yt
from yt.visualization.volume_rendering.transfer_functions import MultiVariateTransferFunction, ColorTransferFunction
from yt.visualization.volume_rendering.render_source import VolumeSource
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array
import numpy as np
import cmasher as cmr


# The parameters pass for each field
field_parameters = [
    {
        "field" : ("gas", "density"),
        "bounds" : [("percentile", 5), ("percentile", 100)], # Use "percentile" to get a percentile and "value" when you want to direct put in a value for the bound. (smallest, largest)
        "use_grey_opacity" : False, # Make underdense regions appear opaque
        "use_ghost_zones" : False, # Uses interpolated data around grid boundaries to smooth out visual artifacts. But will come at the cost of performance
        "colormap" : "turbo", 
        "use_log_space" : True, 
        "file_location" : "shawn/", # The path to the folder you want the file to be saved at
        "label" : "Density", 
        "sigma_clip" : 6, # Removing values that are more than N standard deviations brighter than the mean of your image. Typically, a choice of 4 to 6.
        "interpolation" : "nearest" # "nearest" has no smoothing, "billinear" makes it smoother at the cost of performance
    },
    {
        "field" : ("gas", "temperature"),
        "bounds" : [("percentile", 5), ("value", 2.6e8)], 
        "use_grey_opacity" : False, 
        "use_ghost_zones" : False, 
        "colormap" : "cmr.viola",
        "use_log_space" : False,
        "file_location" : "shawn/", 
        "label" : "Temperature", 
        "sigma_clip" : 6, 
        "interpolation" : "nearest"
    },
    {
        "field" : ("gamer", "velocity_magnitude"),
        "bounds" : [("percentile", 5), ("percentile", 100)],
        "use_grey_opacity" : False, 
        "use_ghost_zones" : False, 
        "colormap" : "cmr.prinsenvlag_r",
        "use_log_space" : True,
        "file_location" : "shawn/", 
        "label" : "Velocity", 
        "sigma_clip" : 6, 
        "interpolation" : "nearest"
    },
    {
        "field" : ("gas", "total_density"),
        "bounds" : [("percentile", 5), ("percentile", 100)],
        "use_grey_opacity" : False, 
        "use_ghost_zones" : False, 
        "colormap" : "turbo",
        "use_log_space" : True,
        "file_location" : "shawn/", 
        "label" : "Total_Density", 
        "sigma_clip" : 6, 
        "interpolation" : "nearest"
    }
]


"""
    bounds_density = (4.13e-29, 4.17e-26)
    #bounds_temp = (2.44e7, 4.50e8) ### EXTREMA
    bounds_temp = (2.44e7, 2.6e8) ######### NOT EXTREMA, but its better ####################################################################################
    #bounds_velocity = (9.81e4, 2.43e8) ### EXTREMA
    bounds_velocity = (1e7, 2.43e8) #### NOT EXTREMA
    bounds_total_density = (4.13e-29, 3.80e-24)
"""

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
def find_CoM(dataset):
    ad = dataset.all_data()
    com_x = ad.mean(('io', 'particle_position_x'))
    com_y = ad.mean(('io', 'particle_position_y'))
    com_z = ad.mean(('io', 'particle_position_z'))
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


# Sets up the source
def setup_source_properties(
    source: VolumeSource, 
    field: tuple, 
    bounds: tuple = None,
    use_grey_opacity: bool = False, 
    use_ghost_zones: bool = False, 
    colormap: str = None,
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

        # Maps data values to alpha (opacity) values.
        # This function creates a linear ramp for opacity.
        def alpha_func(vals, min_val, max_val):
            # Linearly map values from their range [min_val, max_val]
            # to a new range of [0.0 (transparent), 1.0 (opaque)]
            return (vals - min_val) / (max_val - min_val)

        # Uses map_to_colormap to apply the colormap and opacity scale
        tf.map_to_colormap(
            bounds[0],
            bounds[1],
            colormap=colormap,
            scale_func=alpha_func
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
    plt.imshow(render_img, interpolation=interpolation) # bilinear makes it smoother
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

    # Define velocity fields
    define_velocity_fields()

    # Gets CoM from particles (DM and stars) 
    center = find_CoM(dataset = ds)

    # Look at gas/particles within a sphere (not full simulation domain) 
    radius = (sphere_radius, sphere_radius_units)
    sp = ds.sphere(center, radius) # center = CoM defined above, Tuple is required here as parameter

    ds.add_field(('gas', 'total_density'), _total_density, units='g/cm**3', sampling_type='local')
    
    #total_density_in_sphere = sp[('gas', 'total_density')]  # Will probably be used in the future but not sure just yet
    #temperature_in_sphere = sp[('gas', 'temperature')]  # Same as above

    plt.figure(figsize=(10, 5))

    fields = []
    for params in field_list:
        fields.append(params["field"])

    print(fields)

    sc = yt.create_scene(sp, field=fields, lens_type="perspective")
    source = sc[0]

    # Set the camera to look at the region of interest
    cam = sc.camera
    cam.set_focus(center)
    cam.resolution = (resolution, resolution) # can lower resolution to 128 to prioritize faster rendering. 512 is the standard quality
    

    for i, field_params in enumerate(field_list):
        bounds = []
        for value in field_params["bounds"]:
            if value[0] == "percentile":
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
        main(sphere_radius= 2.5, subplot_cords = (3, 4), resolution= 128)
        #main()
