3D Volume Rendering of Galaxy Cluster Mergers

This project contains a Python script for generating scientific visualizations of galaxy cluster merger simulations using the yt analysis package. The primary focus is on visualizing shock fronts and gas kinematics from hydrodynamical simulation data of the MACS J0018.5+1626 cluster.

Features:

The script can generate several types of visualizations. First, "Shock Maps" identify shocks in galaxy clusters by applying a user-defined temperature threshold (for example, 12 keV by default). This creates a binary mask to isolate only gas above the threshold, producing 3D volume renders that show the location and structure of these shock fronts.

"Signed Line-of-Sight Velocity" helps reveal the dynamics of the collision by rendering the velocity of the shocked gas along a custom line of sight. A diverging colormap (by default cmr.prinsenvlag_r) is used to clearly distinguish gas moving towards the observer (blueshifted) from gas moving away (redshifted). Velocities are clipped to a user-defined range to keep the color mapping consistent.

The script can produce a single render or a set of sequential frames for animation. For single renders, the output is saved as a high-quality PDF by default. For animations, each frame is saved as a PNG, with user options to set the starting frame, ending frame, and step size between frames. The script does not directly create a movie file, but the output frames can be combined into a video using external tools such as ffmpeg. It can handle both a single dataset file and glob patterns for multiple time steps.

Setup:

First, clone the repository.

Second, ensure you have the following Python packages installed, preferably in a conda environment: yt, numpy, matplotlib, unyt, cmasher, and pdf2image. The pdf2image package requires Poppler to be installed separately. Subprocess is also used, but it is part of the Python standard library.

Third, if you plan to combine the rendered frames into movies, you must have ffmpeg installed and available in your system’s PATH.
