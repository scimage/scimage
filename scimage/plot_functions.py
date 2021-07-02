#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The following functions can be used to test the results
by plotting the dectected local peak points and highlighting the points 
of each current-sheet, all painted over the image of the simulation results
"""

import numpy as np
import matplotlib.pyplot as plt


# Plotting detected local peak point
def plot_locations_of_local_maximas(x, y, jz, J_th, indexes_of_local_jzmax,
                                    markersize = 6):
    """
    Plotting detected local peak point

    Returns
    -------
    None.

    """
    x_indices=[index[0] for index in indexes_of_local_jzmax]
    y_indices=[index[1] for index in indexes_of_local_jzmax]

    plt.figure()
    plt.pcolor(x, y, jz, cmap='bwr', vmin=np.min(jz), vmax=np.max(jz), shading='auto')        
    plt.colorbar()
    plt.plot(y[y_indices],x[x_indices],'xk', markersize = markersize)
    
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.title('Identified peak points', fontsize = 16)
    plt.legend(['peak points'], prop={'size':'12', 'weight':'bold'}, markerscale=1.0)


# Plotting all the detected regions (e.g. current sheets in a plasma)
def plot_locations_of_region_points(x, y, jz, J_th,indexes_of_points_of_all_cs,
                                    markersize = 1, alpha = 0.6, color='black'):
    """
    Plotting all the detected regions (e.g. current sheets in a plasma)

    Returns
    -------
    None.
    """
    
    plt.figure()
    plt.pcolor(x,y, jz, cmap='bwr', vmin=np.min(jz), vmax=np.max(jz), shading='auto')
    #plt.contourf(x,y,jz,20,cmap='bwr',vmin=-0.5,vmax=0.5)
    plt.colorbar()
    for indexes_of_points_of_a_cs in indexes_of_points_of_all_cs:
        x_indices=[index[0] for index in indexes_of_points_of_a_cs]
        y_indices=[index[1] for index in indexes_of_points_of_a_cs]

        plt.plot(y[y_indices], x[x_indices], 'ok', markersize=markersize, alpha=alpha, color=color)

    plt.xlabel('$x$',fontsize=14)
    plt.ylabel('$y$',fontsize=14)
    plt.title('Detected regions', fontsize = 16)
    plt.legend(['region points'], prop={'size':'12', 'weight':'bold'}, markerscale=2.0)


# Plot one region of the whole image
def plot_region(coordinates_x_in_frame, coordinates_y_in_frame, values_of_frame,
               line_p1, line_p2, region_index = ''):
    fig, ax = plt.subplots()
    ax.pcolormesh(coordinates_y_in_frame, coordinates_x_in_frame, values_of_frame, 
                  alpha=0.8, zorder=1, shading='auto')
    ax.plot([line_p1[1], line_p2[1]], [line_p1[0], line_p2[0]], 
            linewidth=3, zorder=2 , color='black')
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel('$y$', fontsize=12)
    plt.title('Region ' + str(region_index))
    
