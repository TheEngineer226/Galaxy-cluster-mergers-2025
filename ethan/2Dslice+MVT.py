import yt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array

# Default Parameters
DEFAULT_PARAMETERS = {
    "data_location": "sims_data/R1.5_v2400_b250/Data_000044",
    "sphere_radius": 3,
    "sphere_radius_units": 'Mpc'
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

      # --- 2D SLICE PLOT ---
    slc = yt.SlicePlot(ds, 'z', ('gamer', 'velocity_magnitude'), center=center)
    slc.set_cmap(('gamer', 'velocity_magnitude'), 'turbo')
    slc.annotate_streamlines(("gamer", "velocity_x"), ("gamer", "velocity_y"), factor=8)
    output_file = "ethan/velocity_streamlines_slice.png"
    slc.save(output_file)

    # Display
    img = mpimg.imread(output_file)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Velocity Magnitude with Streamlines")
    plt.tight_layout()
    plt.show()

    # --- VOLUME RENDERING (Density-Weighted X Velocity) ---
    sc = yt.create_scene(ds, field=("gamer", "velocity_x"))
    source = sc[0]
    source.set_log(False)
    source.set_weight_field(("gamer", "Dens"))
    source.tfh.set_bounds([-2e8, 2e8])
    source.tfh.map_to_colormap(-2e8, 2e8, scale=5.0, colormap="turbo")
    source.tfh.build_transfer_function()
    sc.camera.set_width(ds.quan(3, "Mpc"))
    sc.save("ethan/x_velocity_density_weighted.png", sigma_clip=4)

if __name__ == "__main__":
    generate_velocity_streamlines(
        DEFAULT_PARAMETERS['data_location'],
        DEFAULT_PARAMETERS['sphere_radius'],
        DEFAULT_PARAMETERS['sphere_radius_units']
    )
