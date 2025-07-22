import yt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unyt import unyt_array
import os

# --- Default Parameters ---
DEFAULT_PARAMETERS = {
    "data_location": "sims_data/R1.5_v2400_b250/Data_000000",
}

# --- Helper Functions ---
def define_velocity_fields(ds):
    """Defines derived velocity fields from momentum and density."""
    if ("gas", "velocity_x") in ds.field_list: 
        return

    def _velocity_x(field, data):
        return data[("gamer", "MomX")] / data[("gamer", "Dens")]
    def _velocity_y(field, data):
        return data[("gamer", "MomY")] / data[("gamer", "Dens")]
    def _velocity_z(field, data):
        return data[("gamer", "MomZ")] / data[("gamer", "Dens")]

    ds.add_field(("gas", "velocity_x"), function=_velocity_x, units="cm/s", sampling_type="cell")
    ds.add_field(("gas", "velocity_y"), function=_velocity_y, units="cm/s", sampling_type="cell")
    ds.add_field(("gas", "velocity_z"), function=_velocity_z, units="cm/s", sampling_type="cell")

def find_CoM(dataset):
    """Returns the CoM from all the particles (DM and stars)."""
    ad = dataset.all_data()
    com_x = ad.mean(('io', 'particle_position_x'))
    com_y = ad.mean(('io', 'particle_position_y'))
    com_z = ad.mean(('io', 'particle_position_z'))
    return unyt_array([com_x, com_y, com_z], com_x.units)


# --- Main Rendering Function ---
def generate_weighted_projection(data_path):
    """Generates a custom weighted projection plot."""
    user_prefix = input("Enter your username (for saving images to a folder): ").strip() + "/"
    if not os.path.isdir(user_prefix): os.makedirs(user_prefix)
        
    ds = yt.load(data_path)
    define_velocity_fields(ds)
    center = find_CoM(ds)

    field_to_plot = ('gas', 'velocity_x')
    weight_field = ('gas', 'density')

    # We project along the 'z' axis to get a "top-down" view of the x-y plane
    proj = yt.ProjectionPlot(
        ds, 
        'z', 
        field_to_plot, 
        center=center, 
        width=(10, 'Mpc'),
        weight_field=weight_field
    )
    
    # --- Visualization Settings ---
    proj.set_unit(field_to_plot, 'km/s')
    proj.set_cmap(field_to_plot, cmap='coolwarm')
    
    # CHANGED: Color bar limits are now -1000 to +1000 km/s
    proj.set_zlim(field_to_plot, zmin=-1000, zmax=1000)
    
    proj.set_log(field_to_plot, log=False)
    
    # --- Save and Display ---
    output_file = user_prefix + "frame_0000_projection.png"
    proj.save(output_file)
    
    img = mpimg.imread(output_file)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Density-Weighted X-Velocity (Frame 0)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_weighted_projection(DEFAULT_PARAMETERS['data_location'])
