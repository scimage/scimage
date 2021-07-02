"""
This particalur example works with images obtained from PIC Hybrid simulation of plasmas, 
in which, each pixel value represents current density at that point. However, you can use 
any input data or image for your own projects.

Examples of visualizations and results can be seen at https://doi.org/10.1063/5.0040692
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

import scimage.identification as ident # Identification of peaks and regions (image segmentation)
import scimage.characterization as char # Characterizing each detected region (e.g. thickness and length)
import scimage.plot_functions as scplt # Plotting functions
from scimage.file_functions import (load_simulation_image, create_folder)

sys.setrecursionlimit(10000) # to avoid possible RecursionError
##################################

# Load the image and its corresponding values, prepare the data:

# Note: 
#   The .npz file is an example for testing purposes
#   Using the load_simulation_image() function is optional. Instead you can 
#   also fill the following variables yourself based on your own images. 
#   The jz 2D array here represents values at each pixel point.
jz, nx, ny, lx, ly, x, y = load_simulation_image('data/data-512-512.npz')

jz_magnitude = np.abs(jz) # Specific to the plasma simulation images
jz_rms = np.sqrt(np.sum(jz**2)/(nx*ny)) # Specific to the plasma simulation images simulation images

J_th = 1.0 * jz_rms # Value for background noise detection. The algorithms reduce the noise in the image data according to this threshold value

# ---------------------------------------------------------
# Call the image processing and analysis functions 

# Custom settings for the algorithms
ratio_of_jzboundary_to_jzmax = 0.5
number_of_points_upto_local_boundary = 25

""" 
Note: It is necessary to call the functions for local peak current density and 
current sheet's points, because the results are used in the characterization part.
"""

indexes_where_condition_is_satisfied = ident.remove_noise(jz_magnitude, J_th)

indexes_of_local_jzmax, values_of_pjz_peaks, array_of_peaks_only = \
    ident.find_local_maxima_at_selected_indexes(jz_magnitude,indexes_where_condition_is_satisfied,
                                                number_of_points_upto_local_boundary)

indexes_of_points_of_all_cs, indexes_of_valid_local_maxima = \
    ident.detect_regions_around_peaks(jz_magnitude, indexes_of_local_jzmax, 
                                      ratio_of_jzboundary_to_jzmax)

print ("Number of detected peaks (maximas):" , len(indexes_of_local_jzmax))
print ("Number of detected regions (current sheets) around the peaks:" , len(indexes_of_points_of_all_cs))
print()

#---------------------------------------------------------------------------
# Call the Chararcterization functions

halfthickness1 = []
halfthickness2 = []
ave_thicknesses = []
lengths_pairwise = []
aspect_ratios = []
min_half_thickness = []
region_index = 1

# Settings:
MIN_FRAME_SIZE = 3 # Minimum size of an acceptable currect-sheet (e.g. 3 x 3 pixels)
SAVE_CHARACTERIZATION_PLOTS = True

# Characterize each detected current-sheet (CS):
for indexes_of_points_of_a_cs in indexes_of_points_of_all_cs:
    
    #1. Build a local frame for each current-sheet (CS) and get data in it
    x_in_cs_frame_global_value, y_in_cs_frame_global_value, jz_in_cs_frame = \
        char.build_region_frame(indexes_of_points_of_a_cs,x,y,jz_magnitude)

    j_min = np.max(jz_in_cs_frame)*0.42 #ratio_of_jzboundary_to_jzmax
    
    print("Current sheet",region_index,"to be characterized; Frame size in pixels:", jz_in_cs_frame.shape)
    
    #2.  Characterize the current-sheet
    if jz_in_cs_frame.shape[0] >= MIN_FRAME_SIZE and jz_in_cs_frame.shape[1] >= MIN_FRAME_SIZE:
        half_thickness_plus_side,half_thickness_minus_side = \
                char.characterize_region(jz_in_cs_frame, 
                     	                 x_in_cs_frame_global_value, y_in_cs_frame_global_value, 
                     	                 j_min)
        
        average_thickness = (half_thickness_plus_side + half_thickness_minus_side)/2 # Average half thicknesses
        halfthickness1.append(half_thickness_plus_side)
        halfthickness2.append(half_thickness_minus_side)
        ave_thicknesses.append(average_thickness)
        minimum_thickness= min(half_thickness_plus_side, half_thickness_minus_side)
        min_half_thickness.append(minimum_thickness)
        
        if np.isnan(half_thickness_plus_side) or np.isnan(half_thickness_minus_side):
            print("Warning! No thickness calculated for peak point", np.max(jz_in_cs_frame))
       
        # Find length with the pair-wise comparison method
        length, p1, p2 = char.find_length_by_pariwise_distance(indexes_of_points_of_a_cs, x, y)
        aspect_ratio = length/average_thickness
        aspect_ratios.append(aspect_ratio)
        lengths_pairwise.append(length)
        
        # Save current sheet plot with the detected length lines ---------------
        if SAVE_CHARACTERIZATION_PLOTS:
            target_folder = './output/example/detected_regions_with_length'
            create_folder(target_folder)
            plt.ioff()
            scplt.plot_region(x_in_cs_frame_global_value, y_in_cs_frame_global_value, 
                              jz_in_cs_frame, p1, p2, region_index=region_index)
            plt.savefig(target_folder + '/region-'+ str(region_index)+'.png', bbox_inches="tight", dpi=90)
            plt.close()
    else:
        print("Warning! Frame too small for region_index=",region_index)
        ave_thicknesses.append(np.nan)
        lengths_pairwise.append(np.nan)
        min_half_thickness.append(np.nan)
        halfthickness1.append(np.nan)
        halfthickness2.append(np.nan)
        aspect_ratios.append(np.nan)

    #---------------
    region_index += 1

# Display some of the characterization results
print()
print("Regions Characterized:")
for index in range(0, len(ave_thicknesses)):
    print("Current sheet "+str(index+1)+": Avg. Thickness:", ave_thicknesses[index],
          "; Length:", lengths_pairwise[index])

##########################################################
# Calling the plotting functions

"""
Now, we can test the algorithm results by plotting the dectected local peak points 
and highlighting the points of each current-sheet, all painted over the image of the simulation results
"""

target_folder = './output/example'
create_folder(target_folder)
plt.ioff()
plt.rcParams["figure.autolayout"] = True # Enable tight layout with minimum margins
plt.rcParams["figure.figsize"] = (10, 8) # Set the desired figure size

# Plot the whole image with detected peak points highlighted
scplt.plot_locations_of_local_maximas(x, y, jz, J_th, indexes_of_local_jzmax)
plt.savefig(target_folder + '/detected-peaks.png', bbox_inches="tight", dpi=90)

# Plot the whole image with detected regions highlighted
scplt.plot_locations_of_region_points(x, y, jz, J_th, indexes_of_points_of_all_cs)
plt.savefig(target_folder + '/detected-regions.png', bbox_inches="tight", dpi=90)

plt.show() # Show all the plots. This also pauses the script here so that we can see the plots
