#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

In this file we present the functions for characterization of the regions 
detected by image segmentation (e.g. detection current sheets) 
This file contains measurment of thickness and length of each region.

"""

import numpy as np 
import math
#import pickle
from scipy import interpolate
# from scipy.spatial.distance import pdist, squareform
# from numpy import random, nanmax, argmax, unravel_index


#####################################################################################################
#From here we implement the algorithm of hessian and eigenvector calculation 
def eigenvec(data,peak_index,x,y):
    # data is a 2-d array with x-variation along rows and y-variation along columns
    # This function returns x and y- components of the eigen vector corresponding 
    # to the largest eigen value of the Hessian matrix at the maximum of the data.
    
    #second derivative along row-direction       
    h11 = np.gradient(np.gradient(data, x, y,edge_order=2)[0], x, y,edge_order=2)[0]
    #second derivative along column direction 
    h22 = np.gradient(np.gradient(data, x, y,edge_order=2)[1], x, y,edge_order=2)[1]
    #second mix derivatives
    h12 = np.gradient(np.gradient(data, x, y,edge_order=2)[1], x, y,edge_order=2)[0] 
    h21 = np.gradient(np.gradient(data, x, y,edge_order=2)[0], x, y,edge_order=2)[1] 
    
    #Hessian matrix at the peak value of the data
    H = np.array([ [h11[peak_index],h12[peak_index]], [h21[peak_index], h22[peak_index]] ])
    
    eigen_val,eigen_vec = np.linalg.eig(H) 
    
    ind_small_eigen_val = np.argmin(np.abs(eigen_val))
    ind_large_eigen_val = np.argmax(np.abs(eigen_val))
        
    eigvec_of_large_eigval_comp_x=eigen_vec[:,ind_large_eigen_val][0]
    eigvec_of_large_eigval_comp_y=eigen_vec[:,ind_large_eigen_val][1]    
    
    return eigvec_of_large_eigval_comp_x,eigvec_of_large_eigval_comp_y
########################################################
#function of linefor calculation of thickness 

def line(y_peak_val,x_line,x_peak_val,M):
    
    y_line=M*(x_line-x_peak_val)+y_peak_val
    
    return y_line    

############################################################  
#Interpolation  
def interpolation_on_line(data,x,y,x_line,y_line):     
 interp_func_jz = interpolate.interp2d(y, x, data, kind='linear')
 i=0
 jz_on_line=np.zeros(len(x_line))
 for y1, x1 in zip(y_line,x_line):
     #jz_on_line=np.diagonal(interp_func_jz(y_line,x_line))
     jz_on_line[i]=interp_func_jz(y1,x1)
     i=i+1

 return jz_on_line
#######################################################
#Function for calculation of distance between two determined points
 
def find_thickness(jz_on_line,j_min,distance_along_line):
    thickness=np.nan
    j_index=np.nan
    j_value=np.nan
    if len(jz_on_line)>0:
        for j_index,j_value in enumerate(jz_on_line):
            if (j_value < j_min):
                thickness=distance_along_line[j_index]
                #print("find_thickness() stopped at j_index",j_index, ", with j_value =",j_value)
                break
    if np.isnan(thickness): #The loop has reached the end without coming lower than j_min         
        print("find_thickness() did not find thickness. Loop ended at j_index",j_index, ", with j_value =",j_value)
    return  thickness


def characterize_region(jz_magnitude, xx, yy, j_min):
    
    ###############################################################################
    # jz.argmax() gives flattened (row-major order) index of maximum value of jz
    # So we use np.unravel_index(...) to get 2-d indices of the maximum in row major
    # order which is the default order for np.unravel_index().   
    index_of_jzmax= np.unravel_index(jz_magnitude.argmax(),jz_magnitude.shape)
    x_coord_at_jzmax=xx[index_of_jzmax[0]]
    y_coord_at_jzmax=yy[index_of_jzmax[1]]

    # x- and y-components of the eigenvector corresponding to the largest eigen 
    # value of the Hessian
    Vx, Vy = eigenvec(jz_magnitude,index_of_jzmax,xx,yy)
    ###############################################################################
    # Calculation of the x and y coordinates on the line parallel to the eigen 
    # vector and passing through the point where current density of the CS under 
    # consideration is maximum. For this we set the slope of the line
    #"slope_of_line_parallel_to_eigvec=tan(alpha)=Vy/Vx", where alpha is the angle 
    # from the x-axis. Then we set the variable "n_points_on_line_on_each_side_of_jzmax" 
    # which is the number of points on the line on each side of the point of the 
    #maximum jz (excluding the maximum point). So the toal number of points on the 
    # line is "2*(n_points_on_line_on_each_side_of_jzmax)+1".
          
    slope_of_line_parallel_to_eigvec=Vy/Vx
    n_points_on_line_on_each_side_of_jzmax=1000
    
    ###############################################################################
    # We choose two sets of the x-coordinates on the line. One set "xPlus_on_line"
    # represents the positive side (value of x increasing) of the point of jzmax 
    # and has elements from the "x_coord_at_jzmax" (first element) to the
    # "x_max_on_line" (last element, the point where x-coordinate has the maximum 
    # value on the line). The other set "xMinus_on_line" represents the negative 
    # side (value of x decreasing) of the point of jzmax and has elements 
    # from the "x_coord_at_jzmax" (first element) to "x_min_on_line" (last 
    # element, the point where x-coordinate has the minimum value on the line).
    # Then the y-coordinates "yPlus_on_line" and "yMinus_on_line" on the two sides 
    # on the line are calculated using the equation of the line.
    
    x_min_on_line=np.min(xx)    
    x_max_on_line=np.max(xx)

    # Increasing values of x-coordinate from the x-coordinate of jzmax
    xPlus_on_line=np.linspace(x_coord_at_jzmax,x_max_on_line,n_points_on_line_on_each_side_of_jzmax+1)
    # decreasing values of x-coordinate from the x-coordinate of jzmax
    xMinus_on_line=np.flip(np.linspace(x_min_on_line,x_coord_at_jzmax,n_points_on_line_on_each_side_of_jzmax+1),0)
    
    yPlus_on_line = line(y_coord_at_jzmax,xPlus_on_line,x_coord_at_jzmax,slope_of_line_parallel_to_eigvec)    
    yMinus_on_line = line(y_coord_at_jzmax,xMinus_on_line,x_coord_at_jzmax,slope_of_line_parallel_to_eigvec)   
    
    ###############################################################################
    # Constraining the y-coordinate on the line within the limit of the 2-d domain
    if np.any(yPlus_on_line > np.max(yy)):
        xPlus_on_line=xPlus_on_line[yPlus_on_line < np.max(yy)]
        yPlus_on_line=yPlus_on_line[yPlus_on_line < np.max(yy)]
        #print('1')
            
    if np.any(yPlus_on_line < np.min(yy)):
        xPlus_on_line=xPlus_on_line[yPlus_on_line > np.min(yy)]
        yPlus_on_line=yPlus_on_line[yPlus_on_line > np.min(yy)]
        #print('2')
    
    if np.any(yMinus_on_line > np.max(yy)):
        xMinus_on_line=xMinus_on_line[yMinus_on_line < np.max(yy)]
        yMinus_on_line=yMinus_on_line[yMinus_on_line < np.max(yy)]
        #print('3')
    
    if np.any(yMinus_on_line < np.min(yy)):
        xMinus_on_line=xMinus_on_line[yMinus_on_line > np.min(yy)]
        yMinus_on_line=yMinus_on_line[yMinus_on_line > np.min(yy)]   
        #print('4')
    ###############################################################################
    # Calculating jz on the line by interpolation from 2-d grid 
    jz_on_line_plus_side = interpolation_on_line(jz_magnitude, xx, yy, xPlus_on_line, yPlus_on_line)  
    jz_on_line_minus_side = interpolation_on_line(jz_magnitude, xx, yy, xMinus_on_line, yMinus_on_line)
    # Distance along the line from the point of jzmax on its two sides 
    distance_on_plus_side=np.sqrt((xPlus_on_line-x_coord_at_jzmax)**2+(yPlus_on_line-y_coord_at_jzmax)**2)
    distance_on_minus_side=np.sqrt((xMinus_on_line-x_coord_at_jzmax)**2+(yMinus_on_line-y_coord_at_jzmax)**2) 
    ################################################################################
    # calculation of half thicknesse (distance from the point of jzmax to the point 
    # where jz=j_min) on the plus and minus sides. 
    half_thickness_plus_side = find_thickness(jz_on_line_plus_side, j_min, distance_on_plus_side)
    half_thickness_minus_side = find_thickness(jz_on_line_minus_side, j_min, distance_on_minus_side)
    
#    plt.pcolormesh(xx,yy,jz_magnitude.T,cmap='jet');plt.colorbar()
#    plt.plot(xPlus_on_line,yPlus_on_line,'w')
#    plt.plot(xMinus_on_line,yMinus_on_line,'g')
#    plt.axis('square')  
#    plt.show()
    return half_thickness_plus_side, half_thickness_minus_side
################################################################################
    
def build_region_frame(indexes_of_points_of_a_cs, x, y, jz_magnitude):    
    CS_indexes = indexes_of_points_of_a_cs
    x_indices=[index[0] for index in CS_indexes]
    y_indices=[index[1] for index in CS_indexes]
        
    # Build a 2D grid/frame around the region
    nx_cs_frame = np.max(x_indices) - np.min(x_indices) + 1
    ny_cs_frame = np.max(y_indices) - np.min(y_indices) + 1
    dx=x[1]-x[0]
    dy=y[1]-y[0]
    x_in_cs_frame=dx*range(nx_cs_frame)
    y_in_cs_frame=dy*range(ny_cs_frame)
    x_in_cs_frame_global_value=x[np.min(x_indices):np.max(x_indices)+1]
    y_in_cs_frame_global_value=y[np.min(y_indices):np.max(y_indices)+1]
    
    jz_in_cs_frame = np.zeros((nx_cs_frame, ny_cs_frame))
    # Transform global indices of CS points into the indices in the CS frame:
    x_index_in_cs_frame = x_indices - np.min(x_indices)
    y_index_in_cs_frame = y_indices - np.min(y_indices)

    jz_in_cs_frame[x_index_in_cs_frame, y_index_in_cs_frame] = jz_magnitude[x_indices,y_indices]
    
    return x_in_cs_frame_global_value, y_in_cs_frame_global_value, jz_in_cs_frame
##########################################################################
#functions of length 
    
def get_distance_of_points(a, b):
    delta_x = np.abs(b[0] - a[0])
    delta_y = np.abs(b[1] - a[1])        
    distance = math.hypot( delta_x, delta_y) # returns Euclidean distance
    return distance

def to_physical_coordinates(point, x, y):
    return(x[point[0]], y[point[1]])
   
def find_length_by_pariwise_distance(indexes_of_points_of_a_cs, x, y):
    max_distance = 0
    d_list=[]
    p1_list=[]
    p2_list=[]
    
    for point_index_from in indexes_of_points_of_a_cs:
        p1 = to_physical_coordinates(point_index_from, x, y)
        for point_index_to in indexes_of_points_of_a_cs:
            p2 = to_physical_coordinates(point_index_to, x, y)
            d = get_distance_of_points(p1, p2)
            d_list.append(d)
            p1_list.append(p1)
            p2_list.append(p2)
            
    max_distance = np.max(d_list)
    
    max_index = np.argmax(d_list)
    p1_max = p1_list[max_index]
    p2_max = p2_list[max_index]
    
    return max_distance, p1_max, p2_max
