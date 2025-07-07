import yt

ds = yt.load("./sims_data/R1.5_v2400_b250/Data_000044")

# Define velocity fields
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

yt.add_field(("gamer", "velocity_x"), function=_velocity_x, units="cm/s", sampling_type="cell")
yt.add_field(("gamer", "velocity_y"), function=_velocity_y, units="cm/s", sampling_type="cell")
yt.add_field(("gamer", "velocity_z"), function=_velocity_z, units="cm/s", sampling_type="cell")
yt.add_field(("gamer", "velocity_magnitude"), function=_velocity_magnitude, units="cm/s", sampling_type="cell")

# Create a velocity-colored slice
slc = yt.SlicePlot(ds, 'z', ('gamer', 'velocity_magnitude'))
slc.annotate_streamlines(("gamer", "velocity_x"), ("gamer", "velocity_y"), factor=8)
slc.set_cmap(('gamer', 'velocity_magnitude'), 'viridis')
slc.save("ethan/velocity_streamlines.png")
