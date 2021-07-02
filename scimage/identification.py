
""" 
This file contains the identification part of the project which is detecting 
local peaks (local maximas) and all the points belonging to each region that stems
from each local peak. 

For example, we can detect local peak current density and all points belong to 
each current sheets in a plasma simulation.

"""

import numpy as np

###################

# Function of detecting peak current density
def find_local_maxima_at_selected_indexes(array, selected_indexes, \
                                          number_of_points_upto_local_boundary):
    # This function checks if the values of "array" at selected indexes provided in
    # "selected_indexes" are local maxima in a surrounding box whose each boundary is away from
    # the candidate point by "number_of_points_upto_local_boundary" number of points.
    # It returns values and indexes of the so found local maxima.
    
    array_of_peaks_only = np.zeros_like(array)
    indexes_of_local_maxima = []
    values_of_peak_points=[]
    n =  number_of_points_upto_local_boundary
    
    for index in selected_indexes:
        i = index[0]
        j = index[1]
        #ignoring 'n' points close to the global boundary.   
        if i>n-1 and j>n-1 and i<array.shape[0]-n and j<array.shape[1]-n:
            # point being considered is a "candidate" point
            value_at_candidate_point = array[i, j] 
            
            # get the values at all points in the box surrounding the candidate point
            values_at_all_points_in_surrounding_box = array[i-n:i+n+1, j-n:j+n+1].copy()

            # exclude central candidate point from being considered when finding maximum value in
            # the surrounding box by assigning it the minimum possible value  
            values_at_all_points_in_surrounding_box[n,n] = -np.inf

            # Find maximum value at the surrounding points.
            maximum_value_at_surrounding_points = \
                np.max(values_at_all_points_in_surrounding_box)

            if (value_at_candidate_point > maximum_value_at_surrounding_points):
                array_of_peaks_only[i,j] = value_at_candidate_point
                values_of_peak_points.append(value_at_candidate_point)
                indexes_of_local_maxima.append(index)
                
    return indexes_of_local_maxima, values_of_peak_points, array_of_peaks_only


###########--------------------------------------------------------------------------
#It should be mentioned that in the above function, in one of two paramters in the argument, 
# "selected_indexes" we passe  another out put of function which is indicated below:

def get_indexes_where_array_exceeds_threshold(array, threshold):
    indexes_where_array_exceeds_threshold = []
    for index, value in np.ndenumerate (array):
        if (value > threshold):
            indexes_where_array_exceeds_threshold.append(index)
    return indexes_where_array_exceeds_threshold

# wrapper function
def remove_noise(array, threshold):
    return get_indexes_where_array_exceeds_threshold(array, threshold)
        
##########----------------------------------------------------------------------------

#Functions for detection of all points belong to each current sheets
# FIrst we check the condtion for each adjucent point for each detected 
#local peack point with the condition of ratio_of_jzboundary_to_jzmax=0.5

def check_adjacent_points_for_minimum_current_density \
            (current_density, flags_to_avoid_rechecking_of_indexes, i, j, \
             minimum_current_density_in_cs, indexes_of_points_of_a_cs):
    # This function checks the point (i,j), if not checked already,  for the condition 
    # that current density "current_density" is larger than a minimum current density
    # "minimum_current_density_in_cs" which is usually taken as a fraction of the local
    # maximum value. If the condition is satisfied, the function calls itself to check 
    # recursively the condition at the adjacent points (i-1,j), (i+1,j), (i,j-1) and (i,j+1).
    # On the first call to this function from the function "detect_sheet_regions", the
    # condition is satisfied automatically as the point (i,j) correspond to the local
    # maxima in current density. On the recursive calls from itself, the condition is
    # checked on other points where it may or may not be satisfied. It adds the indexes of
    # points, where the condition is satisfied, to a list of indexes of points belonging to
    # the current sheet and continues checking the condition in the neighbourhood of the newly
    # found points by recursive calls to itself. The process continues until no point satisfying
    # the condition is found.

    if i>current_density.shape[0]-1 or i<0 or j>current_density.shape[1]-1 or j<0:
        return

    # Check the condition only if the point (i,j) has not been checked before
    if flags_to_avoid_rechecking_of_indexes[i, j] == 0:
        # Assign the value 1 to the flag to mark this point as 'checked' to prevent
        # incorrect repetition
        flags_to_avoid_rechecking_of_indexes[i, j] = 1 

        # check the condition
        if current_density[i,j] > minimum_current_density_in_cs:
            # add to the list as a current sheet point
            indexes_of_points_of_a_cs.append((i, j))
            
            # continue checking the condition at four adjacent points by recursive calls
            check_adjacent_points_for_minimum_current_density \
                (current_density, flags_to_avoid_rechecking_of_indexes, i-1, j, \
                 minimum_current_density_in_cs, indexes_of_points_of_a_cs)
            check_adjacent_points_for_minimum_current_density \
                (current_density, flags_to_avoid_rechecking_of_indexes, i, j-1, \
                 minimum_current_density_in_cs, indexes_of_points_of_a_cs) 
            check_adjacent_points_for_minimum_current_density \
                (current_density, flags_to_avoid_rechecking_of_indexes, i, j+1, \
                 minimum_current_density_in_cs, indexes_of_points_of_a_cs) 
            check_adjacent_points_for_minimum_current_density \
                (current_density, flags_to_avoid_rechecking_of_indexes, i+1, j, \
                 minimum_current_density_in_cs, indexes_of_points_of_a_cs)

# End of function "check_adjacent_points_for_minimum_current_density"s

##################-------------------------------------------------------------------

def detect_regions_around_peaks(current_density, indexes_of_local_maxima,ratio_of_jzboundary_to_jzmax):
    # This function finds points belonging to current sheets in the "current_density" data.
    # For each index of the local maxima provided in "indexes_of_local_maxima", it stores
    # in a list "indexes_of_points_of_a_cs" the indexes of all the points belonging to the
    # current sheet corresponding to the local maxima. And then each such set of the current
    # sheet points is appended to another list "indexes_of_points_of_all_cs" which is returned
    # to the calling program.

    # flag has zero value for unchecked points and will be assigned 1 for each checked point
    flags_to_avoid_rechecking_of_indexes = np.zeros_like(current_density)

    # A list to store indexes of the points belonging to current sheets found in the
    # "current_density" data. Each item of this list is another list containing indexes
    # of the points of an individual current sheet. 
    indexes_of_points_of_all_cs = []
    indexes_of_valid_local_maxima = []
    
    for index in indexes_of_local_maxima:
        # A list to store indexes of the points of an individual current sheet 
        indexes_of_points_of_a_cs = []

        i = index[0]
        j = index[1]

        # minimum current density to define boundaries of the current sheets
        minimum_current_density_in_cs = ratio_of_jzboundary_to_jzmax*current_density[i, j]

        # Function call to check if the points adjacent to the local maxima with index (i,j)
        # satisfy the condition that  current density be larger than the minimum current density.
        # On return, "indexes_of_points_of_a_cs" contains indexes of points of an individual
        # current sheet and "flags_to_avoid_rechecking_of_indexes" has the values 0 and 1 for the
        # unchecked and checked points, respectively. 
        check_adjacent_points_for_minimum_current_density \
            (current_density, flags_to_avoid_rechecking_of_indexes, i, j, \
             minimum_current_density_in_cs,indexes_of_points_of_a_cs)

        # Add the list of indexes of points of an individual current sheet as an item to
        # the list "indexes_of_points_of_all_cs" only if the list for individual current
        # sheet is non-empty. 
        #print(index,len(indexes_of_points_of_a_cs))
        if len(indexes_of_points_of_a_cs) > 0: 
            indexes_of_points_of_all_cs.append(indexes_of_points_of_a_cs)
            indexes_of_valid_local_maxima.append(index)

    return indexes_of_points_of_all_cs, indexes_of_valid_local_maxima
