import yt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array
import os

# Default Parameters
DEFAULT_PARAMETERS = {
    "data_location": "sims_data/R1.5_v2400_b250/Data_000044",
    "sphere_radius": 3,
    "sphere_radius_units": 'Mpc'
}


def define_velocity_fields(ds):
    """
    Defines derived velocity fields from momentum and density if they don't already exist.
    """
    # Check if the final field exists; if so, assume all are defined and skip.
    if ("gamer", "velocity_magnitude") in ds.field_list:
        return

    def _velocity_x(field, data):
        return data[("gamer", "MomX")] / data[("gamer", "Dens")]

    def _velocity_y(field, data):
        return data[("gamer", "MomY")] / data[("gamer", "Dens")]

    def _velocity_z(field, data):
        return data[("gamer", "MomZ")] / data[("gamer", "Dens")]

    def _velocity_magnitude(field, data):
        vx = data[("gamer", "MomX")] / data[("gamer", "Dens")]
        vy = data[("gamer", "MomY")] / data[("gamer", "Dens")]
        vz = data[("gamer", "MomZ")] / data[("gamer", "Dens")]
        return (vx**2 + vy**2 + vz**2)**0.5

    # Use ds.add_field to attach the fields to this specific dataset instance
    ds.add_field(("gamer", "velocity_x"), function=_velocity_x, units="cm/s", sampling_type="cell")
    ds.add_field(("gamer", "velocity_y"), function=_velocity_y, units="cm/s", sampling_type="cell")
    ds.add_field(("gamer", "velocity_z"), function=_velocity_z, units="cm/s", sampling_type="cell")
    ds.add_field(("gamer", "velocity_magnitude"), function=_velocity_magnitude, units="cm/s", sampling_type="cell")


def generate_velocity_slice(data_path, radius, radius_units):
    """Generates and displays a weighted and annotated slice plot."""
    # Get user folder for saving images
    user_prefix = input("Enter your username (for saving images to a folder): ").strip()
    if not user_prefix.endswith('/'):
        user_prefix += '/'
    if not os.path.isdir(user_prefix):
        os.makedirs(user_prefix)
        print(f"Created directory: {user_prefix}")

    # Load data
    ds = yt.load(data_path)

    # Define velocity fields for this dataset
    define_velocity_fields(ds)

    # Calculate each axis for the center of mass separately.
    ad = ds.all_data()
    com_x = ad.mean(('io', 'particle_position_x'))
    com_y = ad.mean(('io', 'particle_position_y'))
    com_z = ad.mean(('io', 'particle_position_z'))
    center = unyt_array([com_x, com_y, com_z], com_x.units)

# Make a 2D slice colored by velocity magnitude
    slc = yt.SlicePlot(ds, 'z', ('gamer', 'velocity_magnitude'), center=center)
    slc.set_cmap(('gamer', 'velocity_magnitude'), 'turbo')

# --- Recommended Changes for a Clearer Image ---
# 1. Set the plot to use a logarithmic scale for velocity
    slc.set_log(('gamer', 'velocity_magnitude'), True)

# 2. Set the data limits to focus on the interesting velocity range (in cm/s)
# This ignores the noisy data below 10^6 cm/s (10 km/s)
    slc.set_zlim(('gamer', 'velocity_magnitude'), 1e6, 3e8)
# ---

# Annotate with streamlines for flow direction and contours for density
    slc.annotate_streamlines(("gamer", "velocity_x"), ("gamer", "velocity_y"), factor=8)
    slc.annotate_contour(("gas", "density"), ncont=8, clim=(1e-29, 1e-26))

    # Save and display image
    output_file = user_prefix + "velocity_slice_with_contours.png"
    slc.save(output_file)
    
    img = mpimg.imread(output_file)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Velocity Magnitude with Density Contours")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_velocity_slice(
        DEFAULT_PARAMETERS['data_location'],
        DEFAULT_PARAMETERS['sphere_radius'],
        DEFAULT_PARAMETERS['sphere_radius_units']
    )
