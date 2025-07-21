import yt
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array

# Default Parameters
DEFAULT_PARAMETERS = {
    "data_location" : "sims_data/R1.5_v2400_b250/", # Path to the directory containing the data files
    "resolution" : 512, # The pixel resolution of each frame
}

# --- Movie Control Parameters ---
output_dir = "frames_2d_movie/" # A dedicated folder for these new frames
start_frame = 0    # The first data file number to use
end_frame = 101     # The data file number to stop BEFORE (will render up to 49)
frame_step = 2     # The step size (1 = every frame, 2 = every other frame)


# --- Helper Functions ---
def define_velocity_fields(ds):
    """Defines derived velocity fields from momentum and density."""
    if ("gas", "velocity_magnitude") in ds.field_list:
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

    ds.add_field(("gas", "velocity_x"), function=_velocity_x, units="cm/s", sampling_type="cell")
    ds.add_field(("gas", "velocity_y"), function=_velocity_y, units="cm/s", sampling_type="cell")
    ds.add_field(("gas", "velocity_z"), function=_velocity_z, units="cm/s", sampling_type="cell")
    ds.add_field(("gas", "velocity_magnitude"), function=_velocity_magnitude, units="cm/s", sampling_type="cell")

def find_CoM(dataset):
    """Returns the CoM from all the particles (DM and stars)."""
    ad = dataset.all_data()
    com_x = ad.mean(('io', 'particle_position_x'))
    com_y = ad.mean(('io', 'particle_position_y'))
    com_z = ad.mean(('io', 'particle_position_z'))
    return unyt_array([com_x, com_y, com_z], com_x.units)


# --- Main Movie-Making Function ---
def generate_movie_frames(base_path, resolution):
    """Loops through datasets to generate frames for a movie."""
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    output_frame_counter = 0

    for data_file_number in range(start_frame, end_frame, frame_step):
        ds_path = os.path.join(base_path, f"Data_{data_file_number:06d}")
        print(f"Rendering output frame {output_frame_counter} from {ds_path}...")
        
        ds = yt.load(ds_path)
        define_velocity_fields(ds)
        center = find_CoM(ds)

        # Create the 2D Slice Plot
        slc = yt.SlicePlot(ds, 'z', ('gas', 'velocity_magnitude'), center=center)
        slc.set_buff_size((resolution, resolution))
        slc.set_cmap(('gas', 'velocity_magnitude'), 'viridis')
        slc.set_log(('gas', 'velocity_magnitude'), True)
        slc.set_zlim(('gas', 'velocity_magnitude'), (10, 'km/s'), (3000, 'km/s'))
        
        # Add all annotations
        slc.annotate_contour(("gas", "density"), ncont=8, clim=(1e-29, 1e-26), plot_args={"colors": "white", "linewidths": 0.5})
        slc.annotate_streamlines(("gas", "velocity_x"), ("gas", "velocity_y"), factor=8)
        slc.annotate_timestamp(corner='upper_left', time_unit='Gyr', text_args={'color': 'white'})

        # Save the frame sequentially
        output_file = os.path.join(output_dir, f"slice_{output_frame_counter:04d}.png")
        slc.save(output_file)
        
        output_frame_counter += 1

    print(f"All {output_frame_counter} frames rendered successfully!")


if __name__ == "__main__":
    generate_movie_frames(
        DEFAULT_PARAMETERS['data_location'],
        DEFAULT_PARAMETERS['resolution']
    )
