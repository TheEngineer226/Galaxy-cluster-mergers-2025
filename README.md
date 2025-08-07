3D Volume Rendering of Galaxy Cluster Mergers

This project contains a suite of Python scripts for generating scientific visualizations of galaxy cluster merger simulations using the yt analysis package. The primary focus is on visualizing shock fronts and gas kinematics from hydrodynamical simulation data of the MACS J0018.5+1626 cluster.

Features:

The scripts in this repository can generate several types of visualizations. Prominently, "Shock Maps" identify shocks in galaxy clusters by finding gas that has been heated to extremely high temperatures (e.g., >10 keV or >15 keV). These scripts create a binary mask to isolate this hot gas and generate 3D volume renders showing the location and structure of these shock fronts.

"Signed Line-of-Sight Velocity" helps to understand the dynamics of the collision by rendering the velocity of the shocked gas along a custom line of sight. A diverging colormap (coolwarm or cmr.prinsenvlag_r) is used to clearly distinguish gas moving towards the observer (blueshifted) from gas moving away (redshifted).

"Movie Generation" allows all visualization types to be rendered as sequential frames for creating animations of the merger event over time. The movie-making scripts are designed to be resumable; if a render is interrupted, it can be restarted without losing progress. With options for frame start and end, as well as step count.

Setup:

This project requires a Python environment with several scientific libraries and external software.

First, clone the repository.

Second, ensure you have the following Python packages installed, preferably in a conda environment: 
yt, numpy, matplotlib, unyt, cmasher, pdf2image (and its dependency, poppler), and subprocess.

Third, you must have ffmpeg installed and available in your system's PATH to create movies from the rendered frames.
