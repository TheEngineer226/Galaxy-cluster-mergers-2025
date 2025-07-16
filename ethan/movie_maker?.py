import yt
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- Parameters ---
# Base path to your simulation data
base_path = "sims_data/R1.5_v2400_b250/"
# A sub-directory to save the individual frames
output_dir = "frames/"
# The number of frames to render
num_frames = 50
# The field to render
field = ('gas', 'density')

# --- Main Movie-Making Function ---
def create_movie_frames():
    """Loops through datasets to generate frames for a movie."""
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # --- Set up a consistent Transfer Function ---
    last_ds_path = os.path.join(base_path, f"Data_{num_frames-1:06d}")
    ds_last = yt.load(last_ds_path)
    ad_last = ds_last.all_data()
    min_dens, max_dens = ad_last.min(field), ad_last.max(field)
    
    log_bounds = [np.log10(min_dens.v), np.log10(max_dens.v)]
    
    tf = yt.ColorTransferFunction(log_bounds)
    def alpha_func(vals, min_val, max_val):
        return ((vals - min_val) / (max_val - min_val))**2.0
    tf.map_to_colormap(log_bounds[0], log_bounds[1], colormap="viridis", scale_func=alpha_func)

    # --- Loop through datasets and render each frame ---
    camera_state = {} # Use a dictionary to store camera properties

    for i in range(num_frames):
        ds_path = os.path.join(base_path, f"Data_{i:06d}")
        print(f"Rendering frame {i+1}/{num_frames} from {ds_path}...")
        
        ds = yt.load(ds_path)
        sc = yt.create_scene(ds, field=field)
        
        # --- CORRECTED: Use a consistent Camera STATE ---
        if i == 0:
            # On the first frame, save the camera's properties
            camera = sc.camera
            camera_state['position'] = camera.position
            camera_state['focus'] = camera.focus
            camera_state['width'] = camera.width
            camera_state['north_vector'] = camera.north_vector
        else:
            # For all other frames, apply the saved properties to the new camera
            sc.camera.position = camera_state['position']
            sc.camera.focus = camera_state['focus']
            sc.camera.width = camera_state['width']
            sc.camera.north_vector = camera_state['north_vector']
            
        # Get the render source and apply our fixed transfer function
        source = sc[0]
        source.set_log(True)
        source.tfh.tf = tf
        source.tfh.bounds = (min_dens.v, max_dens.v)
        
        # Save the frame
        sc.save(os.path.join(output_dir, f"frame_{i:04d}.png"), sigma_clip=4.0)

    print("All frames rendered successfully!")

if __name__ == "__main__":
    create_movie_frames()
