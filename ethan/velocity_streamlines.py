import yt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array

# Default Parameters
DEFAULT_PARAMETERS = {
     "data_location" : "sims_data/R1.5_v2400_b250/Data_000044",
     "sphere_radius" : 3,
     "sphere_radius_units" : 'Mpc'
}

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

def generate_velocity_streamlines(data_path, radius, radius_units):
    # Load data
    ds = yt.load(data_path)

    # Define velocity fields
    define_velocity_fields()

    # Center of mass
    ad = ds.all_data()
    com_x = ad.mean(('io', 'particle_position_x'))
    com_y = ad.mean(('io', 'particle_position_y'))
    com_z = ad.mean(('io', 'particle_position_z'))
    center = unyt_array([com_x, com_y, com_z], com_x.units)

    # Sphere for region (not strictly necessary for 2D slice, but kept for consistency)
    sp = ds.sphere(center, (radius, radius_units))

    # Make a 2D slice with streamlines
    slc = yt.SlicePlot(ds, 'z', ('gamer', 'velocity_magnitude'), center=center)
    slc.set_cmap(('gamer', 'velocity_magnitude'), 'turbo')  # Use advisor's suggested colormap
    slc.annotate_streamlines(("gamer", "velocity_x"), ("gamer", "velocity_y"), factor=8)

    # Save and display image
    output_file = "shawn/velocity_streamlines_slice.png"
    slc.save(output_file)
    img = mpimg.imread(output_file)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Velocity Magnitude with Streamlines")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_velocity_streamlines(
        DEFAULT_PARAMETERS['data_location'],
        DEFAULT_PARAMETERS['sphere_radius'],
        DEFAULT_PARAMETERS['sphere_radius_units']
    )
