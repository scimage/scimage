
# SciCharImage (scimage)
This library provides algorithms for data analysis and image processing, including automation of feature extraction and characterization of image regions. 

### Example Outputs
**Results of processing a synthesized image that represents *peaks* and *valleys*:**

	Number of detected peaks: 11
	Number of detected regions around the peaks: 11
	
	Regions Characterized:
	Region 1: Avg. Thickness: 3.0572496387314114 ; Length: 20.195087901804587
	Region 2: Avg. Thickness: 5.045338744098684 ; Length: 14.345840282578262
	...

Detected peak points | Detected regions around the peak points
------------ | -------------
![Detected peak points](https://github.com/scimage/scimage/blob/main/sample-results/sample-peaks.png) | ![Detected regions around each peaks point](https://github.com/scimage/scimage/blob/main/sample-results/sample-regions.png)

\
**Results of processing an image obtained from a plasma simlation:**

	Number of detected peaks (maximas): 54
	Number of detected regions (current sheets) around the peaks: 54

	Regions Characterized:
	Current sheet 1: Avg. Thickness: 0.4704377263922215 ; Length: 8.961775377533403
	Current sheet 2: Avg. Thickness: 0.799345244773368 ; Length: 10.27921886368474
	...
	

Detected peak points | Detected regions (current sheets) around the peak points
------------ | -------------
![Detected peak points](https://github.com/scimage/scimage/blob/main/sample-results/plasma-peaks.png) | ![Detected regions around each peaks point](https://github.com/scimage/scimage/blob/main/sample-results/plasma-regions.png)

\
**Data filtering and smoothing:**

- We implemented Svizky-Gulay which is one of the digital filtering methods (convolution process). In this method, the local 
  data point is fitted by the sub-set of adjacent data point sequentially with low degree polynomial by the linear least method.\
  ![Screenshot_2021-08-05_13-33-30](https://user-images.githubusercontent.com/86779335/128354740-5bc50030-3cc1-4050-a324-b30eb652e7d3.png) 
  
  Here we implemented the data filtering for diffrent values of *n<sub>f</sub>* (degree of polynomal) and *n<sub>w</sub>* (grid point) and same time frame of simulation **<img src="https://latex.codecogs.com/svg.image?\omega_{ci}t=50" title="\omega_{ci}t=50" />**.

\
**Cropping out and characterizing some of the detected regions (the line represents length):**

Region 6 | Region 10 | Region 46
------------ | ------------- | -------------
![Plasma Region 6](https://github.com/scimage/scimage/blob/main/sample-results/plasma-region-6-r.png) | ![Plasma Region 10](https://github.com/scimage/scimage/blob/main/sample-results/plasma-region-10-r.png) | ![Plasma Region 46](https://github.com/scimage/scimage/blob/main/sample-results/plasma-region-46-r.png)

**Example of statistical analysis:**

 In the below figures, statistical analysis of extracted features (current sheets) and distribution of size geometrical 
    properties(thickness, length) from the simulation are shown.\
    ![Screenshot_2021-08-05_13-34-15](https://user-images.githubusercontent.com/86779335/128373691-5944f873-b373-4792-96e8-0f70111d42a5.png)
   
    


#### Note
This library can be also beneficial to the astrophysics society. For example you can perform statistical analysis of thickness, length, and aspect ratio (length/half-thickness) of each current-sheet in a plasma. Examples of visualizations and results can be seen in a publication in the Journal of Physics of Plasmas: [doi.org/10.1063/5.0040692](https://doi.org/10.1063/5.0040692)


## Contributing
Pull requests are welcome!

## Installation
	pip install scimage

## Using the library
See examples ([simple-example.py](https://github.com/scimage/scimage/blob/main/simple-example.py) and [example.py](https://github.com/scimage/scimage/blob/main/example.py)) in the github repository, on how to use the library functions. Before running the examples, make sure to download the data files from the *data* folder of the repository.

### Simple Example:
    import sys
    import numpy as np
    import matplotlib.pyplot as plt

    import scimage.identification as ident # Identification of peaks and regions (image segmentation)
    import scimage.characterization as char # Characterizing each detected region (e.g. thickness and length)
    import scimage.plot_functions as scplt # Plotting functions
    from scimage.file_functions import (load_simulation_image)

    sys.setrecursionlimit(10000) # to avoid possible RecursionError


    # Prepare a 2D plane image
    values, nx, ny, lx, ly, coordinates_x, coordinates_y = load_simulation_image('data/data-512-512.npz')

    noise_threshold = 0.1
    ratio_of_boundary_to_max = 0.5
    points_upto_local_boundary = 10

    values_abs = np.abs(values)

    # Detect peak points (local maximas):
    good_indexes = ident.remove_noise(values_abs, noise_threshold)
    indexes_of_peaks, peak_values, array_with_peaks_only = \
        ident.find_local_maxima_at_selected_indexes(values_abs, good_indexes, points_upto_local_boundary)

    # Detect regions surrounding each maxima point (image segmentation)
    indexes_of_points_of_all_regions, indexes_of_valid_peaks = \
        ident.detect_regions_around_peaks(values_abs, indexes_of_peaks, ratio_of_boundary_to_max)

    print ("Number of detected peaks:" , len(indexes_of_peaks))
    print ("Number of detected regions around the peaks:" , len(indexes_of_points_of_all_regions))

    # Plot the whole image plane, together with the detected peaks and regions:
    plt.rcParams["figure.autolayout"] = True # Enable tight layout with minimum margins
    plt.rcParams["figure.figsize"] = (10, 8) # Set the desired figure size

    plt.ioff()
    scplt.plot_locations_of_local_maximas(coordinates_x, coordinates_y, values, noise_threshold, indexes_of_peaks)
    scplt.plot_locations_of_region_points(coordinates_x, coordinates_y, values, noise_threshold, indexes_of_points_of_all_regions)
    plt.show() # Show the plots. This also pauses the script here so that we can see the plots


    # Characterize one of the detected regions -------------------------------
    selected_region = 0 # choose one region as an example
    indexes_of_points_of_one_region = indexes_of_points_of_all_regions[selected_region]

    # First, cut out the selected region as a separate frame from the whole image
    coordinates_x_in_frame, coordinates_y_in_frame, values_of_frame = \
        char.build_region_frame(indexes_of_points_of_one_region, coordinates_x, coordinates_y, values)

    # Now, estimate thickness of the region
    min_val = np.max(values_of_frame) * 0.42
    half_thickness_plus_side, half_thickness_minus_side = \
        char.characterize_region(values_of_frame, coordinates_x_in_frame, coordinates_y_in_frame, min_val)

    # Also, find length with the pair-wise comparison method
    length, p1, p2 = char.find_length_by_pariwise_distance(indexes_of_points_of_one_region, coordinates_x, coordinates_y)

    print()
    print("Region", selected_region,"with frame size in pixels", values_of_frame.shape, "characterized:")
    print("\tLength:", length)
    print("\tThickness (half plus, half minus):", half_thickness_plus_side, half_thickness_minus_side)


    # Plot one region and save its image
    plt.rcParams["figure.figsize"] = (4, 4) # Set the desired figure size
    plt.ioff()
    scplt.plot_region(coordinates_x_in_frame, coordinates_y_in_frame, values_of_frame, p1, p2, region_index = selected_region)
    plt.show() # Show the plots. This also pauses the script here so that we can see the plots
