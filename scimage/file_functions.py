
import numpy as np
import os
    
# ----- file functions -------
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print("Created new folder:", folder_name)

def get_folder_name(real_data,average,cs_condition, 
                   lx, ly, nx, ny,J_th,J_rms, 
                   ratio_of_jzboundary_to_jzmax,ratio_of_uez_to_uiz,
                   number_of_points_upto_local_boundary):

    simulation_params = 'lx'+str(int(lx))+'_ly'+str(int(ly))+'_nx'+str(nx)+'_ny'+str(ny)

    if cs_condition:
        algorithm_parameters = 'ratio_of_uez_to_uiz'+str(int(ratio_of_uez_to_uiz))+'_ratio_of_jzboundary_to_jzmax_'+str(ratio_of_jzboundary_to_jzmax)+'_N'+str(number_of_points_upto_local_boundary)
    else:
        algorithm_parameters = 'J_th'+str(np.round(J_th,2))+'_J_rms'+str(np.round(J_rms,2))+'_ratio_of_jzboundary_to_jzmax_'+str(ratio_of_jzboundary_to_jzmax)+'_N'+str(number_of_points_upto_local_boundary)

        
    if real_data :
        if average:            
            folder_name = 'real_data_average_'+simulation_params+'_PARAMS_'+algorithm_parameters
        else:
            folder_name = 'real_data_'+simulation_params+'_PARAMS_'+algorithm_parameters

        if cs_condition:
            folder_name = 'cs_condition_'+'nx'+str(nx)+'_ny'+str(ny)+'_PARAMS_'+algorithm_parameters
    else:
        folder_name = 'generated_data_'+simulation_params+'_PARAMS_'+algorithm_parameters            

    return folder_name

# ---- functions for image files ----
def load_simulation_image(filename):
    data1=np.load(filename)
    
    values=data1['jz']
    
    # nx=data1['nx']
    # ny=data1['ny']
    nx = values.shape[0]
    ny = values.shape[1]
    
    lx=data1['lx']
    ly=data1['ly']
    
    x=np.linspace(-lx/2,lx/2,nx)
    y=np.linspace(-ly/2,ly/2,ny)    
    
    return values, nx, ny, lx, ly, x, y
