import yt
import numpy as np

# Default Parameters
DEFAULT_PARAMETERS = {
    "data_location": "sims_data/R1.5_v2400_b250/Data_000044",
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

def render_density_weighted_velocity(data_path):
    ds = yt.load(data_path)
    define_velocity_fields()

    # Create scene with velocity_x as field to render
    sc = yt.create_scene(ds, field=("gamer", "velocity_x"))
    source = sc[0]

    source.set_log(False)  # Use linear scale for velocity
    source.set_weight_field(("gamer", "Dens"))  # Weight by density

    # Setup transfer function helper
    tfh = yt.TransferFunctionHelper(ds)
    tfh.set_bounds(bounds)
    tfh.add_colormap("turbo")
    tfh.add_gaussian(np.mean(bounds), 0.1 * (bounds[1] - bounds[0]), 0.8, 0.1)
    source.tfh = tfh


    # Set camera width (adjust as needed)
    sc.camera.set_width(ds.quan(3, "Mpc"))

    # Save image
    sc.save("ethan/x_velocity_density_weighted.png", sigma_clip=4)

if __name__ == "__main__":
    render_density_weighted_velocity(DEFAULT_PARAMETERS["data_location"])
