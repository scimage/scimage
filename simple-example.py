#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 20:23:25 2021

@author: mehrdad
"""

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
