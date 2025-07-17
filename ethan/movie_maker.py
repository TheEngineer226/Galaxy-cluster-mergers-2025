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
# The field to render
field = ('gas', 'density')

# --- NEW: Movie Control Parameters ---
# Change these values to control which frames are rendered
start_frame = 1  # The first data file number to use
end_frame = 100   # The data file number to stop BEFORE (will render up to 99)
frame_step = 2    # The step size (2 means every other frame)

# --- Main Movie-Making Function ---
def create_movie_frames():
    """Loops through datasets to generate frames for a movie."""
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # --- Set up a consistent Transfer Function ---
    # We load the LAST dataset in the sequence to find the global max density.
    last_ds_path = os.path.join(base_path, f"Data_{end_frame-1:06d}")
    ds_last = yt.load(last_ds_path)
    ad_last = ds_last.all_data()
    min_dens, max_dens = ad_last.min(field), ad_last.max(field)
    
    log_bounds = [np.log10(min_dens.v), np.log10(max_dens.v)]
    
    tf = yt.ColorTransferFunction(log_bounds)
    def alpha_func(vals, min_val, max_val):
        return ((vals - min_val) / (max_val - min_val))**2.0
    tf.map_to_colormap(log_bounds[0], log_bounds[1], colormap="viridis", scale_func=alpha_func)

    # --- MODIFIED: Loop through the specified range and render each frame ---
    camera_state = {}
    output_frame_counter = 0 # This will count our output frames: 0, 1, 2, 3...

    # The range function now uses your specified start, end, and step
    for data_file_number in range(start_frame, end_frame, frame_step):
        
        ds_path = os.path.join(base_path, f"Data_{data_file_number:06d}")
        print(f"Rendering output frame {output_frame_counter} from {ds_path}...")
        
        ds = yt.load(ds_path)
        sc = yt.create_scene(ds, field=field)
        
        # Use a consistent Camera STATE
        # Checks if this is the first frame in our specified sequence
        if data_file_number == start_frame:
            camera = sc.camera
            camera_state['position'] = camera.position
            camera_state['focus'] = camera.focus
            camera_state['width'] = camera.width
            camera_state['north_vector'] = camera.north_vector
        else:
            sc.camera.position = camera_state['position']
            sc.camera.focus = camera_state['focus']
            sc.camera.width = camera_state['width']
            sc.camera.north_vector = camera_state['north_vector']
            
        source = sc[0]
        source.set_log(True)
        source.tfh.tf = tf
        source.tfh.bounds = (min_dens.v, max_dens.v)
        
        # Save the frame using the simple counter to ensure a sequential sequence
        sc.save(os.path.join(output_dir, f"frame_{output_frame_counter:04d}.png"), sigma_clip=4.0)
        
        output_frame_counter += 1 # Increment for the next output filename

    print(f"All {output_frame_counter} frames rendered successfully!")

if __name__ == "__main__":
    create_movie_frames()
