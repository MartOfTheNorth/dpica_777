#######################################################
# This section is for managing report of pICA.
# Researcher: Mart Panichvatana


########################################################
import os
import nibabel as nib
from nibabel.testing import data_path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats
import math


# import mne
# from mne import io
# from mne.datasets import sample
# from mne.stats import bonferroni_correction, fdr_correction

def ica_fuse_corr( x, y):  # x (48,8) y (48,8)

    # Linear or rank correlation

    if not ('y' in locals() ):
        y = x
        [ny1, ny2] = y.shape     
    else :
        ny2 = y.shape[1] 
    
    # Check dimensions of x and y
    if (x.shape[0] != y.shape[0] ) :
        print('X and Y must have the same number of rows.')
    #end

    [nx1, nx2] = x.shape    

    # Initialise pairwise correlation
    c = np.zeros((nx2, ny2))    

    for ii in range(0,nx2):     
        for jj in range(0,ny2):     
            x_ii = x[:, ii]
            y_jj = y[:, jj]
            c[ii, jj] = ica_fuse_corr2(x[:, ii], y[:, jj]) # Find corr each colume
    # End loop over rows

    return c

def ica_fuse_corr2(x, y): 
    # computes correlation coefficient
    meanX = np.mean(x)      
    meanY = np.mean(y)      

    # Remove mean
    x = x - meanX       
    y = y - meanY       

    corr_coeff = np.sum(np.sum(x*y, axis=0)) / math.sqrt(np.sum(np.sum(x*x)) * np.sum(np.sum(y*y)))

    return corr_coeff   

def find_maxpair_v1( coef_1_2, axis_input, num_pair=1): 
    NCOMP = coef_1_2.shape[0]
    Corr_matrix = abs(coef_1_2)
    coef_max_index_array = np.zeros(NCOMP).astype(int)
    coef_max_pair_array = []
    for i in range(num_pair) :        
            amax  = np.amax(Corr_matrix)
            amax_index = np.where(Corr_matrix == amax)
            amax_row = amax_index[0]
            amax_column = amax_index[1]
            amax_row = int(amax_row[0])
            amax_column = int(amax_column[0])

            # print('[LOG][Multirun ICA] Finding argmax -amax_index_pair  =', amax_index_pair , \
            #         'amax_index[0] = ', amax_index[0], '  amax_index[1]', amax_index[1])
            
            if len(coef_max_pair_array) <= num_pair :
                coef_max_pair_array.append([amax_row,amax_column])
            if axis_input == 1 : 
            # coef_max_index_array[amax_index[0]] = int(amax_index[1])
                    coef_max_index_array[amax_row] = int(amax_column)
                    Corr_matrix[amax_row,:] = 0
                    Corr_matrix[:,amax_column] = 0                                
            elif axis_input == 0 :                     
                    # coef_max_index_array[amax_index[1]] = int(amax_index[0])    
                    coef_max_index_array[amax_column] = int(amax_row)    
                    Corr_matrix[amax_row,:] = 0
                    Corr_matrix[:,amax_column] = 0
    return coef_max_index_array , coef_max_pair_array  

def find_maxpair_v2( coef_1_2, axis_input, num_pair=1): 
    # This function is like v1, but coef_1_2 has one input as vector.

    NCOMP = coef_1_2.shape[0]
    Corr_matrix = abs(coef_1_2)
    coef_max_index_array = np.zeros(NCOMP).astype(int)
    coef_max_pair_array = []
    for i in range(num_pair) :        
            amax  = np.amax(Corr_matrix)
            amax_index = np.where(Corr_matrix == amax)
            amax_row = amax_index[0]
            amax_column = amax_index[1]
            amax_row = int(amax_row[0])
            amax_column = int(amax_column[0])

            # print('[LOG][Multirun ICA] Finding argmax -amax_index_pair  =', amax_index_pair , \
            #         'amax_index[0] = ', amax_index[0], '  amax_index[1]', amax_index[1])
            
            if len(coef_max_pair_array) <= num_pair :
                coef_max_pair_array.append([amax_row,amax_column])
            if axis_input == 1 : 
            # coef_max_index_array[amax_index[0]] = int(amax_index[1])
                    coef_max_index_array[amax_row] = int(amax_column)
                    Corr_matrix[amax_row,:] = 0
                #     Corr_matrix[:,amax_column] = 0                                
            elif axis_input == 0 :                     
                    # coef_max_index_array[amax_index[1]] = int(amax_index[0])    
                    coef_max_index_array[amax_column] = int(amax_row)    
                #     Corr_matrix[amax_row,:] = 0
                    Corr_matrix[:,amax_column] = 0
    return coef_max_index_array , coef_max_pair_array  

def pica_MathLab_A_import(data_path_input, \
    Modality_X_filename_input, \
    Modality_Y_filename_input):

    ########################################################
    # print('Modality_X_filename_input = ',Modality_X_filename_input)
    # print('Modality_Y_filename_input = ',Modality_Y_filename_input)

    Modality_X_mathlab_1d_all = []
    Modality_Y_mathlab_1d_all = []

    if Modality_X_filename_input :
        # print('Modality_X_filename_input = ',Modality_X_filename_input)
        fname = os.path.join(data_path_input, Modality_X_filename_input)

        Modality_X_mathlab_1d_all = np.genfromtxt(fname, dtype=float )
        # print("Modality_X_mathlab_2d_all_len = ", len(Modality_X_mathlab_2d_all))
        Modality_X_mathlab_2d_all = np.asarray(Modality_X_mathlab_1d_all)
        # print("Modality_X_mathlab_2d_all.shape = " , Modality_X_mathlab_2d_all.shape )
        # print("Modality_X_mathlab_2d_all = " , Modality_X_mathlab_2d_all )


    if Modality_Y_filename_input :
        # print('Modality_Y_filename_input = ',Modality_Y_filename_input)
        fname = os.path.join(data_path_input, Modality_Y_filename_input)

        Modality_Y_mathlab_1d_all = np.genfromtxt(fname, dtype=float )
        # print("Modality_Y_mathlab_2d_all_len = ", len(Modality_Y_mathlab_2d_all))
        Modality_Y_mathlab_2d_all = np.asarray(Modality_Y_mathlab_1d_all)
        # print("Modality_Y_mathlab_2d_all.shape = " , Modality_Y_mathlab_2d_all.shape )
        # print("Modality_Y_mathlab_2d_all = " , Modality_Y_mathlab_2d_all )

       

    ########################################################
    return (Modality_X_mathlab_2d_all, Modality_Y_mathlab_2d_all)   

def pica_MathLab_S_import(data_path_input, \
    Modality_X_filename_input, \
    Modality_Y_filename_input):

    ########################################################
    # print('Modality_X_filename_input = ',Modality_X_filename_input)
    # print('Modality_Y_filename_input = ',Modality_Y_filename_input)

    Modality_X_mathlab_1d_all = []
    Modality_Y_mathlab_1d_all = []

    if Modality_X_filename_input :
        # print('Modality_X_filename_input = ',Modality_X_filename_input)
        fname = os.path.join(data_path_input, Modality_X_filename_input)

        Modality_X_mathlab_1d_all = np.genfromtxt(fname, dtype=float )
        # print("Modality_X_mathlab_2d_all_len = ", len(Modality_X_mathlab_2d_all))
        Modality_X_mathlab_2d_all = np.asarray(Modality_X_mathlab_1d_all)
        # print("Modality_X_mathlab_2d_all.shape = " , Modality_X_mathlab_2d_all.shape )
        # print("Modality_X_mathlab_2d_all = " , Modality_X_mathlab_2d_all )


    if Modality_Y_filename_input :
        # print('Modality_Y_filename_input = ',Modality_Y_filename_input)
        fname = os.path.join(data_path_input, Modality_Y_filename_input)

        Modality_Y_mathlab_1d_all = np.genfromtxt(fname, dtype=float )
        # print("Modality_Y_mathlab_2d_all_len = ", len(Modality_Y_mathlab_2d_all))
        Modality_Y_mathlab_2d_all = np.asarray(Modality_Y_mathlab_1d_all)
        # print("Modality_Y_mathlab_2d_all.shape = " , Modality_Y_mathlab_2d_all.shape )
        # print("Modality_Y_mathlab_2d_all = " , Modality_Y_mathlab_2d_all )


    ########################################################
    return (Modality_X_mathlab_2d_all, Modality_Y_mathlab_2d_all)   

        # pica_report_v1.pica_MathLab_A_import(LocalA_Corr_X, LocalA_Corr_Y, \
        #             LocalA_Corr_MathLab_X, LocalA_Corr_MathLab_Y )

def pica_large_correlate(Corr_X, Corr_Y):
    # Import
    # data i/o
    import os

    # compute in parallel
    from multiprocessing import Pool

    # the usual
    import numpy as np
    import pandas as pd

    import deepgraph as dg

    data_path = "D:\data\FITv2.0e\FIT_Output\correlations\\"

    # parameters (change these to control RAM usage)
    step_size = 1e5
    n_processes = 100

    # load samples as memory-map
    X = np.load('samples.npy', mmap_mode='r')

    # create node table that stores references to the mem-mapped samples
    v = pd.DataFrame({'index': range(X.shape[0])})

    # connector function to compute pairwise pearson correlations
    def corr(index_s, index_t):
        features_s = X[index_s]
        features_t = X[index_t]
        corr = np.einsum('ij,ij->i', features_s, features_t) / n_samples
        return corr

    # index array for parallelization
    pos_array = np.array(np.linspace(0, n_features*(n_features-1)//2, n_processes), dtype=int)

    # parallel computation
    def create_ei(i):

        from_pos = pos_array[i]
        to_pos = pos_array[i+1]

        # initiate DeepGraph
        g = dg.DeepGraph(v)

        # create edges
        g.create_edges(connectors=corr, step_size=step_size,
                    from_pos=from_pos, to_pos=to_pos)

        # store edge table
        g.e.to_pickle(data_path + '/{}.pickle'.format(str(i).zfill(3)))

    # computation
    # if __name__ == '__main__':
    os.makedirs(data_path, exist_ok=True)
    indices = np.arange(0, n_processes - 1)
    p = Pool()
    for _ in p.imap_unordered(create_ei, indices):
        pass


    # store correlation values
    files = os.listdir(data_path)
    files.sort()
    store = pd.HDFStore('e.h5', mode='w')
    for f in files:
        et = pd.read_pickle(data_path + '/{}'.format(f))
        store.append('e', et, format='t', data_columns=True, index=False)
    store.close()

    # load correlation table
    e = pd.read_hdf('e.h5')
    print(e)

    return ()   

# End of def pica_large_correlate


def pica_2d_correlate(Corr_X, Corr_Y, pltshow=False):


    print('Corr_X  (X x X)',Corr_X.shape )    
    print('Corr_Y  (Y x Y)',Corr_Y.shape )   

    # plt.matshow(Corr_X)
    # plt.colorbar()
    # plt.show()

    # plt.matshow(Corr_Y)
    # plt.colorbar()
    # plt.show()

    Coef_X_Y = np.corrcoef ( Corr_X, Corr_Y  )

    # most_correlate = pica_most_correlate4(Coef_X_Y)   # num_list =  [ 1 13] [12 13] [ 0 11] [3 8] [ 1 12] [1 5] [0 1] [ 9 13]

    # Reshap to one quarter.      
    ic_num = int((Coef_X_Y.shape[0])/2)
    Coef_X_Y = Coef_X_Y[:ic_num,-ic_num:]
    print('Coef_X_Y  rehape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
    # print('Coef_X_Y   = ', Coef_X_Y  )    

    # Round an array to the given number of decimals.
    Coef_X_Y = np.round(Coef_X_Y,8)

    # flatten
    coef_X_Y_flat = Coef_X_Y.flatten()

    # Sort in ascending order      
    rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

    # Reverse 
    coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

    print('coef_X_Y_flat   = ')     
    # print('coef_X_Y_flat   = ', coef_X_Y_flat  )     
    for x in range(5): 
        print(coef_X_Y_flat[x], sep = ", ") 

    print('rev_coef_X_Y_flat   = '  )   
    # print('rev_coef_X_Y_flat   = ', rev_coef_X_Y_flat  )   
    for x in range(5): 
        print(rev_coef_X_Y_flat[x], sep = ", ") 

    if pltshow  :
        plt.matshow(Coef_X_Y)
        plt.colorbar()
        plt.show()

    return (Coef_X_Y)   

def pica_2d_correlate3(Corr_X, Corr_Y):


    print('Corr_X  (X x X)',Corr_X.shape )    
    print('Corr_Y  (Y x Y)',Corr_Y.shape )   

    # plt.matshow(Corr_X)
    # plt.colorbar()
    # plt.show()

    # plt.matshow(Corr_Y)
    # plt.colorbar()
    # plt.show()

    Coef_X_Y = np.corrcoef ( Corr_X, Corr_Y , rowvar=False )

    # most_correlate = pica_most_correlate4(Coef_X_Y)   # num_list =  [ 1 13] [12 13] [ 0 11] [3 8] [ 1 12] [1 5] [0 1] [ 9 13]

    # Reshap to one quarter.      
    ic_num = int((Coef_X_Y.shape[0])/2)
    Coef_X_Y = Coef_X_Y[:ic_num,-ic_num:]
    print('Coef_X_Y  rehape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
    # print('Coef_X_Y   = ', Coef_X_Y  )    

    # Round an array to the given number of decimals.
    Coef_X_Y = np.round(Coef_X_Y,8)

    # flatten
    coef_X_Y_flat = Coef_X_Y.flatten()

    # Sort in ascending order      
    rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

    # Reverse 
    coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

    print('coef_X_Y_flat   = ')     
    # print('coef_X_Y_flat   = ', coef_X_Y_flat  )     
    for x in range(5): 
        print(coef_X_Y_flat[x], sep = ", ") 

    print('rev_coef_X_Y_flat   = '  )   
    # print('rev_coef_X_Y_flat   = ', rev_coef_X_Y_flat  )   
    for x in range(5): 
        print(rev_coef_X_Y_flat[x], sep = ", ") 


    # plt.matshow(Coef_X_Y)
    # plt.colorbar()
    # plt.show()

    return (Coef_X_Y)   

def pica_2d_correlate2(Corr_X, Corr_Y):


    print('Corr_X  (X x X)',Corr_X.shape )    
    print('Corr_Y  (Y x Y)',Corr_Y.shape )   

    Coef_X_Y = np.corrcoef ( Corr_X, Corr_Y,  rowvar=False  )

    # most_correlate = pica_most_correlate4(Coef_X_Y)   # num_list =  [ 1 13] [12 13] [ 0 11] [3 8] [ 1 12] [1 5] [0 1] [ 9 13]

    # Reshap to one quarter.      
    ic_num = int((Coef_X_Y.shape[0])/2)
    Coef_X_Y = Coef_X_Y[:ic_num,-ic_num:]
    print('Coef_X_Y  rehape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
    # print('Coef_X_Y   = ', Coef_X_Y  )    

    # Round an array to the given number of decimals.
    Coef_X_Y = np.round(Coef_X_Y,8)

    # flatten
    coef_X_Y_flat = Coef_X_Y.flatten()

    # Sort in ascending order      
    rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

    # Reverse 
    coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

    print('coef_X_Y_flat   = ')     
    # print('coef_X_Y_flat   = ', coef_X_Y_flat  )     
    # for x in range(len(coef_X_Y_flat)): 
    for x in range(5): 
        print(coef_X_Y_flat[x], sep = ", ") 

    # print('rev_coef_X_Y_flat   = '  )   
    # # print('rev_coef_X_Y_flat   = ', rev_coef_X_Y_flat  )   
    # for x in range(len(rev_coef_X_Y_flat)): 
    #     print(rev_coef_X_Y_flat[x], sep = ", ") 


    # plt.matshow(Coef_X_Y)
    # plt.colorbar()
    # plt.show()

    return (Coef_X_Y)   

def pica_2d_correlate5(Corr_X, Corr_Y,  data_path_save, data_file_save, pltshow=False, pltsave=False):


    # print('Corr_X  (X x X)',Corr_X.shape )    
    # print('Corr_Y  (Y x Y)',Corr_Y.shape )   

    # plt.matshow(Corr_X)
    # plt.colorbar()
    # plt.show()

    # plt.matshow(Corr_Y)
    # plt.colorbar()
    # plt.show()

    Coef_X_Y = np.corrcoef ( Corr_X, Corr_Y  )

    # most_correlate = pica_most_correlate4(Coef_X_Y)   # num_list =  [ 1 13] [12 13] [ 0 11] [3 8] [ 1 12] [1 5] [0 1] [ 9 13]

    # Reshap to one quarter.      
    ic_num = int((Coef_X_Y.shape[0])/2)
    Coef_X_Y = Coef_X_Y[:ic_num,-ic_num:]
    # print('Coef_X_Y  rehape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
    # print('Coef_X_Y   = ', Coef_X_Y  )    

    # Round an array to the given number of decimals.
    Coef_X_Y = np.round(Coef_X_Y,8)

    # flatten
    coef_X_Y_flat = Coef_X_Y.flatten()

    # Sort in ascending order      
    rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

    # Reverse 
    coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

    print('coef_X_Y_flat   = ')     
    # print('coef_X_Y_flat   = ', coef_X_Y_flat  )     
    for x in range(5): 
        print(coef_X_Y_flat[x], sep = ", ") 

    print('rev_coef_X_Y_flat   = '  )   
    # print('rev_coef_X_Y_flat   = ', rev_coef_X_Y_flat  )   
    for x in range(5): 
        print(rev_coef_X_Y_flat[x], sep = ", ") 

    if pltshow  :
        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.75)
        plt.clim(-0.3, 0.3)
        # changing the rc parameters and plotting a line plot
        plt.rcParams['figure.figsize'] = [2, 2]   
        plt.savefig(data_path_save  + data_file_save)
        plt.show()

    return (Coef_X_Y)   

def pica_2d_correlate6(Corr_X, Corr_Y,  data_path_save, data_file_save, screenshow=False, pltshow=False, pltsave=False):


    # print('Corr_X  (X x X)',Corr_X.shape )    
    # print('Corr_Y  (Y x Y)',Corr_Y.shape )   

    # plt.matshow(Corr_X)
    # plt.colorbar()
    # plt.show()

    # plt.matshow(Corr_Y)
    # plt.colorbar()
    # plt.show()

    Coef_X_Y = np.corrcoef ( Corr_X, Corr_Y  )

    # most_correlate = pica_most_correlate4(Coef_X_Y)   # num_list =  [ 1 13] [12 13] [ 0 11] [3 8] [ 1 12] [1 5] [0 1] [ 9 13]

    # Reshap to one quarter.      
    ic_num = int((Coef_X_Y.shape[0])/2)
    Coef_X_Y = Coef_X_Y[:ic_num,-ic_num:]
    # print('Coef_X_Y  rehape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
    # print('Coef_X_Y   = ', Coef_X_Y  )    

    # Round an array to the given number of decimals.
    Coef_X_Y = np.round(Coef_X_Y,8)

    if screenshow :
        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        print('coef_X_Y_flat   = ')     
        # print('coef_X_Y_flat   = ', coef_X_Y_flat  )     
        for x in range(5): 
            print(coef_X_Y_flat[x], sep = ", ") 

        print('rev_coef_X_Y_flat   = '  )   
        # print('rev_coef_X_Y_flat   = ', rev_coef_X_Y_flat  )   
        for x in range(5): 
            print(rev_coef_X_Y_flat[x], sep = ", ") 

    if pltshow  :
        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.75)
        plt.clim(-0.4, 0.4)
        # changing the rc parameters and plotting a line plot
        plt.rcParams['figure.figsize'] = [2, 2]           
        plt.show()

    if pltsave :
        plt.savefig(data_path_save  + data_file_save)

    return (Coef_X_Y)   

def pica_2d_correlate7(Corr_X, Corr_Y,  data_path_save, data_file_save, screenshow=False, pltshow=False, pltsave=False):

    Coef_X_Y = np.corrcoef ( Corr_X, Corr_Y  )

    # Reshap to one quarter.      
    ic_num = int((Coef_X_Y.shape[0])/2)
    Coef_X_Y = Coef_X_Y[:ic_num,-ic_num:]
    print('Coef_X_Y  rehape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
    # print('Coef_X_Y   = ', Coef_X_Y  )    

    # Round an array to the given number of decimals.
    Coef_X_Y = np.round(Coef_X_Y,10)

    if screenshow :
        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        print('coef_X_Y_flat   = ')     
        # print('coef_X_Y_flat   = ', coef_X_Y_flat  )     
        for x in range(5): 
            print(coef_X_Y_flat[x], sep = ", ") 

        print('rev_coef_X_Y_flat   = '  )   
        # print('rev_coef_X_Y_flat   = ', rev_coef_X_Y_flat  )   
        for x in range(5): 
            print(rev_coef_X_Y_flat[x], sep = ", ") 

    if pltshow  :
        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.75)
        # plt.clim(-0.4, 0.4)
        # changing the rc parameters and plotting a line plot
        plt.rcParams['figure.figsize'] = [2, 2]           
        plt.show()

    if pltsave :
        plt.savefig(data_path_save  + data_file_save)

    return (Coef_X_Y)  

# def pica_2d_correlate8(Corr_X, Corr_Y,  data_path_save, data_file_save, \
#         picture_scale = 0, screenshow=False, pltshow=False, pltsave=False):


def pica_2d_correlate8(Corr_X, Corr_Y,  data_path_save, data_file_save, title, xlabel, ylabel, \
        picture_scale = 0, screenshow=False, pltshow=False, pltsave=False):

    Coef_X_Y = np.corrcoef ( Corr_X, Corr_Y  )

    # Reshap to one quarter.      
    ic_num = int((Coef_X_Y.shape[0])/2)
    Coef_X_Y = Coef_X_Y[:ic_num,-ic_num:]
    # print('Coef_X_Y  rehape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
    # print('Coef_X_Y   = ', Coef_X_Y  )    

    # Round an array to the given number of decimals.
    Coef_X_Y = np.round(Coef_X_Y,10)

    if pltshow  :
        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.75)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)        
        if picture_scale != 0 :
            plt.clim(-(picture_scale), picture_scale)
        # changing the rc parameters and plotting a line plot
        # plt.rcParams['figure.figsize'] = [2, 2]           
        plt.show()

    if pltsave :
        plt.savefig(data_path_save  + data_file_save)
        
    if screenshow :
        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        print('coef_X_Y_flat   = ')     
        print('coef_X_Y_flat   = ', coef_X_Y_flat  )     
        for x in range(5): 
            print(coef_X_Y_flat[x], sep = ", ") 

        print('rev_coef_X_Y_flat   = '  )   
        print('rev_coef_X_Y_flat   = ', rev_coef_X_Y_flat  )   
        for x in range(5): 
            print(rev_coef_X_Y_flat[x], sep = ", ") 



    return (Coef_X_Y)  


def pica_2d_correlate9(Corr_X, Corr_Y,  data_path_save, data_file_save, title, xlabel, ylabel, \
        picture_scale = 0, screenshow=False, screensave=False, pltshow=False, pltsave=False):

    Coef_X_Y = np.corrcoef ( Corr_X, Corr_Y  )

    # Reshap to one quarter.      
    ic_num = int((Coef_X_Y.shape[0])/2)
    Coef_X_Y = Coef_X_Y[:ic_num,-ic_num:]
    # print('Coef_X_Y  rehape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
    # print('Coef_X_Y   = ', Coef_X_Y  )    

    # Round an array to the given number of decimals.
    Coef_X_Y = np.round(Coef_X_Y,10)

    if pltshow  :
        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.75)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)        
        if picture_scale != 0 :
            plt.clim(-(picture_scale), picture_scale)
        # changing the rc parameters and plotting a line plot
        # plt.rcParams['figure.figsize'] = [2, 2]           
        plt.show()



    if pltsave :
        plt.savefig(data_path_save  + data_file_save)

    if screenshow :
        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        print('Positive correlation coefficients = ')     
        # print('coef_X_Y_flat   = ', coef_X_Y_flat  )     
        for x in range(ic_num): 
            print(coef_X_Y_flat[x], sep = ", ") 

        print('Negative correlation coefficients = '  )   
        # print('rev_coef_X_Y_flat   = ', rev_coef_X_Y_flat  )   
        for x in range(ic_num): 
            print(rev_coef_X_Y_flat[x], sep = ", ") 

    if screensave : 
        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        with open(data_path_save  + data_file_save + ".txt", "w") as text_file:
            print("Positive correlation coefficients = ", file=text_file)     
            for x in range(ic_num): 
                print(coef_X_Y_flat[x], sep = ", ", file=text_file)

            print("Negative correlation coefficients = ", file=text_file)     
            for x in range(ic_num): 
                print(rev_coef_X_Y_flat[x], sep = ", ", file=text_file)



    return (Coef_X_Y)  


def pica_2d_correlate10(Corr_X, Corr_Y,  data_path_save, data_file_save, title, xlabel, ylabel, \
        picture_scale = 0, screenshow=False, screensave=False, pltshow=False, pltsave=False):

    Coef_X_Y = np.corrcoef ( Corr_X, Corr_Y  )

    # Reshap to one quarter.      
    ic_num = int((Coef_X_Y.shape[0])/2)
    Coef_X_Y = Coef_X_Y[:ic_num,-ic_num:]
    # print('===========Correlation Start=============== ')    
    # print('Coef_X_Y  rehape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
    # print('Coef_X_Y   = ', Coef_X_Y  )    

    # Round an array to the given number of decimals.
    Coef_X_Y = np.round(Coef_X_Y,10)


    if pltsave :
        # print('===========Correlation Start== Plot save============= ')    
        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.80)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)   
        
        # fig1 = plt.gcf()
        # fig1.savefig(data_path_save  + data_file_save)
        plt.savefig(data_path_save  + data_file_save)


    if pltshow  :
        # print('===========Correlation Start== Plot show============= ')    
        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.80)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)        

        if picture_scale != 0 :
            plt.clim(0, picture_scale)
        # changing the rc parameters and plotting a line plot
        plt.rcParams['figure.figsize'] = [2, 2]           
        plt.show()


    if screenshow :
        # print('===========Correlation Start== Screen============= ')    

        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        # print('Pearson correlation coefficients - Positive =  ')     
        # print('coef_X_Y_flat   = ', coef_X_Y_flat  )     
        for x in range(ic_num): 
            print(coef_X_Y_flat[x], sep = ", ") 

        print('')   

    if screensave : 
        # print('===========Correlation Start== Screen save============= ')    

        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        with open(data_path_save  + data_file_save + ".txt", "w") as text_file:
            print("Pearson correlation coefficients - Positive = ", file=text_file)     
            for x in range(ic_num): 
                print(coef_X_Y_flat[x], sep = ", ", file=text_file)

            print("Pearson correlation coefficients - Negative = ", file=text_file)     
            for x in range(ic_num): 
                print(rev_coef_X_Y_flat[x], sep = ", ", file=text_file)



    # print('===========Correlation Finish=============== ')    
    return (Coef_X_Y)  

def pica_2d_correlate11(Corr_X, Corr_Y,  data_path_save, data_file_save, title, xlabel, ylabel, \
        picture_scale = 0, screenshow=False, screensave=False, pltshow=False, pltsave=False):

    # Corr_matrix = np.abs(ica_fuse_corr(Corr_X.T, Corr_Y.T))  # 8 x 8

    Coef_X_Y_full = np.corrcoef ( Corr_X, Corr_Y  )

    # Reshap to one quarter.      
    ic_num = int((Coef_X_Y_full.shape[0])/2)
    # Coef_X_Y = Coef_X_Y[:ic_num,-ic_num:]    
    Coef_X_Y = Coef_X_Y_full[:ic_num:,ic_num:]
    print('===========Correlation Start=============== ')    
    print('Coef_X_Y  rehape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
    print('Coef_X_Y   = ', Coef_X_Y  )    

    # Round an array to the given number of decimals.
    Coef_X_Y = np.round(Coef_X_Y,10)

    if pltshow  :
        # print('===========Correlation Start== Plot show============= ')    
        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.80)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)        
        # CS = plt.contourf(X, Y, Z, levels=6, vmin=0.0, vmax=1.0, cmap=cm.coolwarm)

        # colorbar = clippedcolorbar(CS)

        if picture_scale != 0 :
            plt.clim(0, picture_scale)
        # changing the rc parameters and plotting a line plot
        plt.rcParams['figure.figsize'] = [2, 2]           
        plt.show()

    if pltsave :
        # print('===========Correlation Start== Plot save============= ')    
        plt.savefig(data_path_save  + data_file_save)

    if screenshow :
        # print('===========Correlation Start== Screen============= ')    

        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        print('Positive correlation coefficients = ')     
        # print('coef_X_Y_flat   = ', coef_X_Y_flat  )     
        for x in range(ic_num): 
            print(coef_X_Y_flat[x], sep = ", ") 

        # Experiment 
        print('Negative correlation coefficients = '  )   
        # print('rev_coef_X_Y_flat   = ', rev_coef_X_Y_flat  )   
        for x in range(ic_num): 
            print(rev_coef_X_Y_flat[x], sep = ", ") 

    if screensave : 
        # print('===========Correlation Start== Screen save============= ')    

        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        with open(data_path_save  + data_file_save + ".txt", "w") as text_file:
            print("Positive correlation coefficients = ", file=text_file)     
            for x in range(ic_num): 
                print(coef_X_Y_flat[x], sep = ", ", file=text_file)

            print("Negative correlation coefficients = ", file=text_file)     
            for x in range(ic_num): 
                print(rev_coef_X_Y_flat[x], sep = ", ", file=text_file)



    # print('===========Correlation Finish=============== ')    
    return (Coef_X_Y)  

def pica_2d_correlate12(Corr_X, Corr_Y,  data_path_save, data_file_save, title, xlabel, ylabel, \
        picture_scale = 0, screenshow=False, screensave=False, pltshow=False, pltsave=False):


    # References: 
    #  https://mne.tools/stable/auto_examples/stats/fdr_stats_evoked.html


    rows, cols = Corr_X.shape[0], Corr_Y.shape[0]
    corr = []
    pval = []
    for i in range(rows):
        for j in range(cols):
            corr_, pval_ = scipy.stats.pearsonr( Corr_X[i,:], Corr_Y[j,:] )
            corr.append(corr_)
            pval.append(pval_)
    corr = np.array(corr)
    pval = np.array(pval)

    print("") 
    print("Pearson correlation coefficients = ")     
    print('Data_X.shape   = ', Corr_X.shape  )     
    print('Data_Y.shape   = ', Corr_Y.shape  )     
    print('Corr.shape   = ', corr.shape  )   
    print('corr   = ', corr  )   
    print('p-val.shape   = ', pval.shape  )  
    print('p-val   = ', pval  )  
    print("") 



    #Method 1
    # def fdr(p_vals):

    #     from scipy.stats import rankdata
    #     ranked_p_values = rankdata(p_vals)
    #     fdr = p_vals * len(p_vals) / ranked_p_values
    #     fdr[fdr > 1] = 1

    #     return fdr    
    # fdr_output =  fdr(pval)

    #Method 2
    # from statsmodels.stats.multitest import fdrcorrection
    # rejected = fdrcorrection(pval,  alpha=alpha, method='indep')
    # rejected, q-value = fdrcorrection(pval,  alpha=alpha, method='indep')
    # alpha = 0.05
    
    #Method 3
    alpha = 0.05   
    # T, pval = stats.ttest_1samp(Corr_X, 0)  # pval(106)
    reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='indep') 
    # threshold_fdr = np.min(np.abs(corr)[reject_fdr])
    print("")    
    print("False Discovery Rate (FDR) correction = ")     
    print('alpha   = ', alpha  )     
    print('reject_fdr   = ', reject_fdr  )   
    print('p-val_fdr   = ', pval_fdr  )   
    # print('threshold_fdr   = ', threshold_fdr  )   
    print("") 

    # # Original code
    # Coef_X_Y = np.corrcoef ( Corr_X, Corr_Y  )  # X(5, 63) Y(7, 63)  # XY(12, 12)

    # # Reshap to one quarter.      
    # ic_num = int((Coef_X_Y.shape[0])/2)


    if pltshow  :
        # print('===========Correlation Start== Plot show============= ')    
        plt.plot(times, T, 'k', label='T-stat')
        xmin, xmax = plt.xlim()
        # plt.hlines(threshold_uncorrected, xmin, xmax, linestyle='--', colors='k',
        #         label='p=0.05 (uncorrected)', linewidth=2)
        # plt.hlines(threshold_bonferroni, xmin, xmax, linestyle='--', colors='r',
        #         label='p=0.05 (Bonferroni)', linewidth=2)
        plt.hlines(threshold_fdr, xmin, xmax, linestyle='--', colors='b',
                label='p=0.05 (FDR)', linewidth=2)
        plt.legend()
        plt.xlabel("Time (ms)")
        plt.ylabel("T-stat")
        plt.show()

        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.80)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)        
        # CS = plt.contourf(X, Y, Z, levels=6, vmin=0.0, vmax=1.0, cmap=cm.coolwarm)

        # colorbar = clippedcolorbar(CS)

        if picture_scale != 0 :
            plt.clim(0, picture_scale)
        # changing the rc parameters and plotting a line plot
        plt.rcParams['figure.figsize'] = [2, 2]           
        plt.show()

    if pltsave :
        # print('===========Correlation Start== Plot save============= ')    
        plt.savefig(data_path_save  + data_file_save)

    if screenshow :
        print("")         
        # print('===========Correlation Start== Screen============= ')    

        # # flatten
        # coef_X_Y_flat = Coef_X_Y.flatten()

        # # Sort in ascending order      
        # rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # # Reverse 
        # coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        # print('Positive correlation coefficients = ')     
        # # print('coef_X_Y_flat   = ', coef_X_Y_flat  )     
        # for x in range(ic_num): 
        #     print(coef_X_Y_flat[x], sep = ", ") 

        # # Experiment 
        # print('Negative correlation coefficients = '  )   
        # # print('rev_coef_X_Y_flat   = ', rev_coef_X_Y_flat  )   
        # for x in range(ic_num): 
        #     print(rev_coef_X_Y_flat[x], sep = ", ") 

    if screensave : 
        print("")        
        # print('===========Correlation Start== Screen save============= ')    

        # # flatten
        # coef_X_Y_flat = Coef_X_Y.flatten()

        # # Sort in ascending order      
        # rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # # Reverse 
        # coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        # with open(data_path_save  + data_file_save + ".txt", "w") as text_file:
        #     print("Positive correlation coefficients = ", file=text_file)     
        #     for x in range(ic_num): 
        #         print(coef_X_Y_flat[x], sep = ", ", file=text_file)

        #     print("Negative correlation coefficients = ", file=text_file)     
        #     for x in range(ic_num): 
        #         print(rev_coef_X_Y_flat[x], sep = ", ", file=text_file)



    # print('===========Correlation Finish=============== ')    
    return (corr)  

def pica_2d_correlate13(Corr_X, Corr_Y,  data_path_save, data_file_save, title, xlabel, ylabel, \
        picture_scale = 0, screenshow=False, screensave=False, pltshow=False, pltsave=False):

    Coef_X_Y = np.abs(ica_fuse_corr(Corr_X.T, Corr_Y.T))  # 8 x 8

    ic_num, col_num = ((Coef_X_Y.shape))
    pair_num = min(ic_num, col_num)

    # Identify the max-corr pair of Coef_X_Y
    Coef_2d_L1_output, Coef_max_pair_output = (find_maxpair_v1(Coef_X_Y, 1, pair_num))     

    if pltshow  :
        # print('===========Correlation Start== Plot show============= ')    
        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.80)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)        
        # CS = plt.contourf(X, Y, Z, levels=6, vmin=0.0, vmax=1.0, cmap=cm.coolwarm)

        # colorbar = clippedcolorbar(CS)

        if picture_scale != 0 :
            plt.clim(0, picture_scale)
        # changing the rc parameters and plotting a line plot
        plt.rcParams['figure.figsize'] = [2, 2]           
        plt.show()

    if pltsave :
        # print('===========Correlation Start== Plot save============= ')    
        plt.savefig(data_path_save  + data_file_save)

    if screenshow :
        # print('===========Correlation Start== Screen============= ')    
        print(' ')     
        # print('===========Correlation Start=============== ')    
        print('Coef_X_Y  shape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
        print('Coef_X_Y   = ', Coef_X_Y  )    

        print('Maximum-Correlation pair = ')      
        for x in range(pair_num): 
            print(Coef_max_pair_output[x]  , sep = ", ") 

        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]


        print('Maximum-Correlation coefficients = ')      
        for x in range(ic_num): 
            print(coef_X_Y_flat[x], sep = ", ") 

        print('Maximum-Correlation coefficients and pair = ')      
        for x in range(pair_num): 
            # print(Coef_max_pair_output[x], coef_X_Y_flat[x], sep = ", ") 
            print(' ', (Coef_max_pair_output[x]), ',  %4.4f' %( coef_X_Y_flat[x]))
             

        # print('===========Correlation Finish==============')
        print(' ')

    if screensave : 
        # print('===========Correlation Start== Screen save============= ')    

        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        with open(data_path_save  + data_file_save + ".txt", "w") as text_file:
            print("Correlation coefficients = ", file=text_file)     
            print("===========Correlation Start=============== ", file=text_file)    
            print("Coef_X_Y  shape (N) x (N))", Coef_X_Y.shape , file=text_file)
            print("Coef_X_Y   = ", Coef_X_Y  , file=text_file)   

            print('Maximum-Correlation pair = ', file=text_file) 
            for x in range(pair_num): 
                print(Coef_max_pair_output[x]  , sep = ", ", file=text_file) 
        
            print('Maximum-Correlation coefficients = ', file=text_file)      
            for x in range(ic_num): 
                print(coef_X_Y_flat[x], sep = ", ", file=text_file)

            print('Maximum-Correlation coefficients and pair = ', file=text_file)     
            for x in range(pair_num): 
                print(Coef_max_pair_output[x], coef_X_Y_flat[x], sep = ", ", file=text_file) 

            print("===========Correlation Finish==============", file=text_file) 

    # print('===========Correlation Finish=============== ')    
    return (Coef_X_Y)  


def pica_2d_correlate14(Corr_X, Corr_Y,  data_path_save, data_file_save, title, xlabel, ylabel, \
        picture_scale = 0, screenshow=False, screensave=False, pltshow=False, pltsave=False):

    Coef_X_Y = np.abs(ica_fuse_corr(Corr_X.T, Corr_Y.T))  # 8 x 8

    ic_num, col_num = ((Coef_X_Y.shape))
    pair_num = min(ic_num, col_num)

    # Identify the max-corr pair of Coef_X_Y
    Coef_2d_L1_output, Coef_max_pair_output = (find_maxpair_v1(Coef_X_Y, 1, pair_num))     

    if pltshow  :
        # print('===========Correlation Start== Plot show============= ')    
        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.80)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)        
        # CS = plt.contourf(X, Y, Z, levels=6, vmin=0.0, vmax=1.0, cmap=cm.coolwarm)

        # colorbar = clippedcolorbar(CS)

        if picture_scale != 0 :
            plt.clim(0, picture_scale)
        # changing the rc parameters and plotting a line plot
        plt.rcParams['figure.figsize'] = [2, 2]           
        plt.show()

    if pltsave :
        # print('===========Correlation Start== Plot save============= ')    
        plt.savefig(data_path_save  + data_file_save)

    if screenshow :
        # print('===========Correlation Start== Screen============= ')    
        print(' ')     
        # print('===========Correlation Start=============== ')    
        print('Coef_X_Y  shape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
        # print('Coef_X_Y   = ', Coef_X_Y  )    

        print('The 1st Maximum-Correlation pair = ')      
        print(Coef_max_pair_output[0]  , sep = ", ") 

        # print('All Maximum-Correlation pair = ')      
        # for x in range(pair_num): 
        #     print(Coef_max_pair_output[x]  , sep = ", ") 

        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]


        # print('Maximum-Correlation coefficients = ')      
        # for x in range(ic_num): 
        #     print(coef_X_Y_flat[x], sep = ", ") 

        # print('Maximum-Correlation coefficients and pair = ')      
        # for x in range(pair_num): 
        #     # print(Coef_max_pair_output[x], coef_X_Y_flat[x], sep = ", ") 
        #     print(' ', (Coef_max_pair_output[x]), ',  %4.4f' %( coef_X_Y_flat[x]))
             

        # print('===========Correlation Finish==============')
        print(' ')

    if screensave : 
        # print('===========Correlation Start== Screen save============= ')    

        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        with open(data_path_save  + data_file_save + ".txt", "w") as text_file:
            print("Correlation coefficients = ", file=text_file)     
            print("===========Correlation Start=============== ", file=text_file)    
            # print("Coef_X_Y  shape (N) x (N))", Coef_X_Y.shape , file=text_file)
            # print("Coef_X_Y   = ", Coef_X_Y  , file=text_file)   

            print('Maximum-Correlation pair = ', file=text_file) 
            for x in range(pair_num): 
                print(Coef_max_pair_output[x]  , sep = ", ", file=text_file) 
        
            print('Maximum-Correlation coefficients = ', file=text_file)      
            for x in range(ic_num): 
                print(coef_X_Y_flat[x], sep = ", ", file=text_file)

            print('Maximum-Correlation coefficients and pair = ', file=text_file)     
            for x in range(pair_num): 
                print(Coef_max_pair_output[x], coef_X_Y_flat[x], sep = ", ", file=text_file) 

            print("===========Correlation Finish==============", file=text_file) 

    # print('===========Correlation Finish=============== ')    
    return (Coef_X_Y)  


def pica_2d_correlate15(Corr_X, Corr_Y,  data_path_save, data_file_save, title, xlabel, ylabel, \
        picture_scale = 0, screenshow=False, screensave=False, pltshow=False, pltsave=False):

    Coef_X_Y = np.abs(ica_fuse_corr(Corr_X.T, Corr_Y.T))  # 8 x 8

    ic_num, col_num = ((Coef_X_Y.shape))
    pair_num = min(ic_num, col_num)

    # Identify the max-corr pair of Coef_X_Y
    Coef_2d_L1_output, Coef_max_pair_output = (find_maxpair_v1(Coef_X_Y, 1, pair_num))     

    if pltshow  :
        # print('===========Correlation Start== Plot show============= ')    
        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.80)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)        
        # CS = plt.contourf(X, Y, Z, levels=6, vmin=0.0, vmax=1.0, cmap=cm.coolwarm)

        # colorbar = clippedcolorbar(CS)

        if picture_scale != 0 :
            plt.clim(0, picture_scale)
        # changing the rc parameters and plotting a line plot
        plt.rcParams['figure.figsize'] = [2, 2]           
        plt.show()

    if pltsave :
        # print('===========Correlation Start== Plot save============= ')    
        plt.savefig(data_path_save  + data_file_save)

    if screenshow :
        # print('===========Correlation Start== Screen============= ')    
        print(' ')     
        # print('===========Correlation Start=============== ')    
        print('Coef_X_Y  shape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
        # print('Coef_X_Y   = ', Coef_X_Y  )    

        print('The 1st Maximum-Correlation pair = ')      
        print(Coef_max_pair_output[0]  , sep = ", ") 

        # print('All Maximum-Correlation pair = ')      
        # for x in range(pair_num): 
        #     print(Coef_max_pair_output[x]  , sep = ", ") 

        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]


        # print('Maximum-Correlation coefficients = ')      
        # for x in range(ic_num): 
        #     print(coef_X_Y_flat[x], sep = ", ") 

        print('Maximum-Correlation coefficients of each pairs = ')      
        for x in range(pair_num): 
            # print(Coef_max_pair_output[x], coef_X_Y_flat[x], sep = ", ") 
            print(' ', (Coef_max_pair_output[x]), ',  %4.16f' %( coef_X_Y_flat[x]))
             

        # print('===========Correlation Finish==============')
        print(' ')

    if screensave : 
        # print('===========Correlation Start== Screen save============= ')    

        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        with open(data_path_save  + data_file_save + ".txt", "w") as text_file:
            print("Correlation coefficients = ", file=text_file)     
            print("===========Correlation Start=============== ", file=text_file)    
            # print("Coef_X_Y  shape (N) x (N))", Coef_X_Y.shape , file=text_file)
            # print("Coef_X_Y   = ", Coef_X_Y  , file=text_file)   

            print('Maximum-Correlation pair = ', file=text_file) 
            for x in range(pair_num): 
                print(Coef_max_pair_output[x]  , sep = ", ", file=text_file) 
        
            print('Maximum-Correlation coefficients = ', file=text_file)      
            for x in range(ic_num): 
                print(coef_X_Y_flat[x], sep = ", ", file=text_file)

            print('Maximum-Correlation coefficients and pair = ', file=text_file)     
            for x in range(pair_num): 
                print(Coef_max_pair_output[x], coef_X_Y_flat[x], sep = ", ", file=text_file) 

            print("===========Correlation Finish==============", file=text_file) 

    # print('===========Correlation Finish=============== ')    
    return (Coef_max_pair_output[0] ) 

def pica_2d_correlate16(Corr_X, Corr_Y,  data_path_save, data_file_save, title, xlabel, ylabel, \
        picture_scale = 0, screenshow=False, screensave=False, pltshow=False, pltsave=False):

    Coef_X_Y = np.abs(ica_fuse_corr(Corr_X.T, Corr_Y.T))  # 8 x 8

    # print("pica_2d_correlate16   Coef_X_Y.shape = ", Coef_X_Y.shape) 
    np.savetxt( data_path_save +  data_file_save + ".csv", Coef_X_Y, delimiter=",")

    ic_num, col_num = ((Coef_X_Y.shape))
    pair_num = max(ic_num, col_num)
    # pair_num = 3

    # Identify the max-corr pair of Coef_X_Y
    Coef_2d_L1_output, Coef_max_pair_output = (find_maxpair_v2(Coef_X_Y, 1, pair_num))     

    if pltsave :
        # print('===========Correlation Start== Plot save============= ')    
        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.80)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)       

        plt.savefig(data_path_save  + data_file_save)

    if pltshow  :
        # print('===========Correlation Start== Plot show============= ')    
        plt.matshow(Coef_X_Y)
        plt.colorbar( shrink=0.80)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)        

        if picture_scale != 0 :
            plt.clim(0, picture_scale)
        # changing the rc parameters and plotting a line plot
        plt.rcParams['figure.figsize'] = [2, 2]           
        plt.show()



    if screenshow :
        # print('===========Correlation Start== Screen============= ')    
        print(' ')     
        # print('===========Correlation Start=============== ')    
        print('Coef_X_Y  shape (N) x (N))', Coef_X_Y.shape )    #  (8, 8)
        # print('Coef_X_Y   = ', Coef_X_Y  )    

        print('The 1st Maximum-Correlation pair = ')      
        print(Coef_max_pair_output[0]  , sep = ", ") 

        # print('All Maximum-Correlation pair = ')      
        # for x in range(pair_num): 
        #     print(Coef_max_pair_output[x]  , sep = ", ") 

        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]


        # print('Maximum-Correlation coefficients = ')      
        # for x in range(ic_num): 
        #     print(coef_X_Y_flat[x], sep = ", ") 

        print('Maximum-Correlation coefficients of each pairs = ')      
        for x in range(pair_num): 
            # print(Coef_max_pair_output[x], coef_X_Y_flat[x], sep = ", ") 
            print(' ', (Coef_max_pair_output[x]), ',  %4.16f' %( coef_X_Y_flat[x]))
             

        # print('===========Correlation Finish==============')
        print(' ')

    if screensave : 
        # print('===========Correlation Start== Screen save============= ')    

        # flatten
        coef_X_Y_flat = Coef_X_Y.flatten()

        # Sort in ascending order      
        rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    

        # Reverse 
        coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

        with open(data_path_save  + data_file_save + ".txt", "w") as text_file:
            print("Correlation coefficients = ", file=text_file)     
            print("===========Correlation Start=============== ", file=text_file)    
            # print("Coef_X_Y  shape (N) x (N))", Coef_X_Y.shape , file=text_file)
            # print("Coef_X_Y   = ", Coef_X_Y  , file=text_file)   

            print('Maximum-Correlation pair = ', file=text_file) 
            for x in range(pair_num): 
                print(Coef_max_pair_output[x]  , sep = ", ", file=text_file) 
        
            print('Maximum-Correlation coefficients = ', file=text_file)      
            for x in range(ic_num): 
                print(coef_X_Y_flat[x], sep = ", ", file=text_file)

            print('Maximum-Correlation coefficients and pair = ', file=text_file)     
            for x in range(pair_num): 
                print(Coef_max_pair_output[x], coef_X_Y_flat[x], sep = ", ", file=text_file) 

            print("===========Correlation Finish==============", file=text_file) 

    # print('===========Correlation Finish=============== ')    
    return (Coef_max_pair_output[0] ) 

def pica_LocalA_correlate(LocalA_Corr_X, LocalA_Corr_Y, \
    LocalA_Corr_MathLab_X, LocalA_Corr_MathLab_Y):



    def compute_covariance ( x, y ):    
        mean_x = np.mean( x )
        mean_y = np.mean( y )
        std_x  = np.std ( x )
        std_y  = np.std ( y )
        n      = len    ( x )

        # Return matrix as same size of one input
        return ((sum (x - mean_x ) * (y - mean_y ) ) * 1/(n-1))


    def compute_correlation_coefficients( x, y ):    
        mean_x = np.mean( x )
        mean_y = np.mean( y )
        std_x  = np.std ( x )
        std_y  = np.std ( y )
        n      = len    ( x )

        # Return Single
        return (( x - mean_x ) * ( y - mean_y )).sum() / n / ( std_x * std_y )

    def compute_correlation_coefficients_Pearson( x, y ):    
        mean_x = np.mean( x )
        mean_y = np.mean( y )
        std_x  = np.std ( x )
        std_y  = np.std ( y )
        n      = len    ( x )

        # Return (Pearson’s correlation coefficient , Two-tailed p-value) 
        return (compute_covariance(x, y) / (std_x * std_y))

    def compute_correlation_coefficients_Spearman( x, y ):    
        mean_x = np.mean( x )
        mean_y = np.mean( y )
        std_x  = np.std ( x )
        std_y  = np.std ( y )
        n      = len    ( x )

        # Return ( Spearman correlation matrix or correlation coefficient , Two-tailed p-value) 
        return (compute_covariance(np.rank(x), np.rank(y)) / (np.std(np.rank(x)) * np.std(np.rank(y))) )


    print('Modality_X === LocalA_mixer_X (sN_X x r_X)',LocalA_Corr_X.shape )    # (43, 8)
    print('Modality_Y === LocalA_mixer_Y (sN_Y x r_Y)',LocalA_Corr_Y.shape )    # (43, 8)

    print('Modality_X === LocalA_mixer_MathLab_X (sN_X x r_X)',LocalA_Corr_MathLab_X.shape )    # (43, 8)
    print('Modality_Y === LocalA_mixer_MathLab_Y (sN_Y x r_Y)',LocalA_Corr_MathLab_Y.shape  )   # (43, 8)


    # Correlation_Coefficients
    print('===== Correlation_Coefficients via np.corrcoef ')

    # print('===== Correlation_Coefficients via np.corrcoef === LocalA_Corr_MathLab_X, LocalA_Corr_MathLab_Y,')
    # LocalA_coef_X_Y = np.corrcoef ( LocalA_Corr_MathLab_X, LocalA_Corr_MathLab_Y,  rowvar=False  )
    # num_indices_row =   [1, 0, 3, 2, 7, 5, 6, 4]
    # num_indices_col =   [5, 3, 0, 7, 2, 4, 1, 6]

    # LocalA_coef_X_Y = np.corrcoef ( LocalA_Corr_X, LocalA_Corr_Y,  rowvar=False  )
    # num_indices_row =   [3, 6, 1, 4, 0, 5, 7, 2]
    # num_indices_col =   [3, 2, 1, 5, 0, 7, 4, 6]    
    # num_indices_row =   [1, 7, 4, 6, 5, 0, 2, 3]
    # num_indices_col =   [6, 3, 2, 7, 0, 5, 4, 1]
    # num_indices_row =   [7, 2, 6, 4, 5, 0, 1, 3]
    # num_indices_col =   [5, 2, 7, 1, 3, 0, 6, 4]

    # LocalA_coef_X_Y = np.corrcoef ( LocalA_Corr_X, LocalA_Corr_X,  rowvar=False  )
    #     
    # LocalA_coef_X_Y = np.corrcoef ( LocalA_Corr_MathLab_X, LocalA_Corr_MathLab_X,  rowvar=False  )
    # num_indices_row =   [0, 1, 2, 3, 4, 5, 6, 7]
    # num_indices_col =   [0, 1, 2, 3, 4, 5, 6, 7]

    LocalA_coef_X_Y = np.corrcoef ( LocalA_Corr_X, LocalA_Corr_MathLab_X,  rowvar=False  )
    # num_indices_row =   [0, 5, 2, 3, 4, 6, 1, 7]
    # num_indices_col =   [7, 0, 6, 5, 3, 1, 2, 4]

    # LocalA_coef_X_Y = np.corrcoef ( LocalA_Corr_Y, LocalA_Corr_MathLab_Y,  rowvar=False  )
    # num_indices_row =   [2, 3, 5, 1, 0, 7, 4, 6]
    # num_indices_col =   [4, 1, 6, 2, 3, 0, 7, 5]

    # print('LocalA_coef_X_Y (N_X+N_Y) x (N_X+N_Y))',LocalA_coef_X_Y.shape )    #  (16, 16)
    # print('LocalA_coef_X_Y (N_X+N_Y) x (N_X+N_Y))',LocalA_coef_X_Y )  



    # most_correlate = pica_most_correlate2(LocalA_coef_X_Y)   # num_list =  [ 1 13] [12 13] [ 0 11] [3 8] [ 1 12] [1 5] [0 1] [ 9 13]
###    
    most_correlate = pica_most_correlate4(LocalA_coef_X_Y)   # num_list =  [ 1 13] [12 13] [ 0 11] [3 8] [ 1 12] [1 5] [0 1] [ 9 13]
###

    # plt.matshow(LocalA_coef_X_Y)
    # plt.xticks(range(LocalA_coef_X_Y.shape[0]), LocalA_coef_X_Y.shape[0])
    # plt.yticks(range(LocalA_coef_X_Y.shape[0]), LocalA_coef_X_Y.shape[0])
    # plt.colorbar()
    # plt.show()

    # print('===== Correlation_Coefficients via np.corrcoef === LocalA_Corr_X, LocalA_Corr_Y')
    # LocalA_coef_X_Y = np.corrcoef ( LocalA_Corr_X, LocalA_Corr_Y,  rowvar=False  )
    # most_correlate = pica_most_correlate2(LocalA_coef_X_Y)   # num_list =  [10 11] [ 9 10] [ 9 11] [ 8 11] [1 9] [ 8 10] [ 9 12] [ 8 15]

    plt.matshow(LocalA_coef_X_Y)
    # plt.xticks(range(LocalA_coef_X_Y.shape[0]), LocalA_coef_X_Y.shape[0])
    # plt.yticks(range(LocalA_coef_X_Y.shape[0]), LocalA_coef_X_Y.shape[0])
    plt.colorbar()
    plt.show()



    return ()   

        # pica_report_v1.pica_Source_correlate(S_source_X, S_source_Y, \
        #             self.S_source_MathLab_X, self.S_source_MathLab_Y)

def pica_Source_correlate(S_sources_X, S_sources_Y, \
    S_sources_MathLab_X, S_sources_MathLab_Y):

    print('Modality_X === S_sources_X (r_X x d_X) ',S_sources_X.shape )     # (8, 153594)
    print('Modality_Y === S_sources_Y (r_Y x d_Y) ',S_sources_Y.shape )     # (8, 367)

    print('Modality_X === S_sources_MathLab_X (r_X x d_X) ',S_sources_MathLab_X.shape )     # (8, 58179)
    print('Modality_Y === S_sources_MathLab_Y (r_Y x d_Y) ',S_sources_MathLab_Y.shape )     # (8, 367)


    # Correlation_Coefficients
    print('===== Correlation_Coefficients via np.corrcoef ')

    # S_sources_coef_X_Y = np.corrcoef ( S_sources_MathLab_Y, S_sources_MathLab_Y,  rowvar=False  )
    S_sources_coef_X_Y = np.corrcoef ( S_sources_Y, S_sources_MathLab_Y,  rowvar=False  )

    print('S_sources_coef_X_Y (d_X+d_Y) x (d_X+d_Y))',S_sources_coef_X_Y.shape )    
    # print('S_sources_coef_X_Y (d_X+d_Y) x (d_X+d_Y))',S_sources_coef_X_Y )  

    # most_correlate = pica_most_correlate4(S_sources_coef_X_Y)   

    # plt.matshow(S_sources_coef_X_Y)
    # plt.colorbar()
    # plt.show()


    return ()  


def pica_most_correlate4(coef_X_Y_input):

    # Reshap to one quarter.        #  (16, 16)
    ic_num = int((coef_X_Y_input.shape[0])/2)
    coef_X_Y_input = coef_X_Y_input[:ic_num,-ic_num:]
    print('coef_X_Y_input  rehape (N) x (N))', coef_X_Y_input.shape )    #  (8, 8)
    print('coef_X_Y_input   = ', coef_X_Y_input  )    

    # Round an array to the given number of decimals.
    coef_X_Y_input = np.round(coef_X_Y_input,8)

    coef_indice = 0
    X_i = 0
    Y_i = 0
    num_indices_row = []
    num_indices_col = []
   


    # flatten
    coef_X_Y_flat = coef_X_Y_input.flatten()
    # print('coef_X_Y_flat =  ', coef_X_Y_flat )   

    # Sort in ascending order
    # coef_X_Y_flat = np.unique(coef_X_Y_flat)
    # rev_coef_X_Y_flat = np.unique(coef_X_Y_flat)        
    rev_coef_X_Y_flat = np.sort(coef_X_Y_flat)    
    # print(' coef_X_Y_flat = np.sort(coef_X_Y_flat)    = ', coef_X_Y_flat )   

    # Reverse 
    # rev_coef_X_Y_flat = coef_X_Y_flat[::-1]
    coef_X_Y_flat = rev_coef_X_Y_flat[::-1]

    print('coef_X_Y_flat   = ', coef_X_Y_flat  )     
    print('rev_coef_X_Y_flat   = ', rev_coef_X_Y_flat  )     

    while coef_indice <= ic_num :
        # print('coef_indice   = ', coef_indice  )   

        # print('coef_X_Y_flat[X_i]   = ', coef_X_Y_flat[X_i]  )     
        # print('rev_coef_X_Y_flat[Y_i]   = ', rev_coef_X_Y_flat[Y_i]  ) 
        if abs(coef_X_Y_flat[X_i]) == 1 or abs(rev_coef_X_Y_flat[Y_i]) == 1:
            # print('num_indices_row =  ', X_i )              
            # print('num_indices_col =  ', Y_i )     
            num_indices_row.append(X_i )
            num_indices_col.append(Y_i )
            X_i = X_i + 1                
            Y_i = Y_i + 1
            # coef_indice = len(num_indices_row) + 1
        elif coef_X_Y_flat[X_i] > abs(rev_coef_X_Y_flat[Y_i]) :
            # Possitive > Negative
            # print('Possitive > Negative' )  

            index = np.where(coef_X_Y_input  == coef_X_Y_flat[X_i])

            # print('index =  ', *index )   
            # print('num_indices_row =  ', *num_indices_row )              
            # print('num_indices_col =  ', *num_indices_col )              
            if not (len(index[0])==0) :     # index is not empty

                # print('len_index =  ', len(index[0]))   
                index_lengh = len(index[0])
                index_run = 0
                while index_run < index_lengh :
                    index_row = (index[0])[index_run]
                    index_col = (index[1])[index_run]

                    if not num_indices_row :        # num_indices is empty                   
                        # num_indices.append(np.array(index))      # Append Negative   
                        # num_indices.append(np.array(index))      # Append Negative   
                        num_indices_row.append(index_row)
                        num_indices_col.append(index_col)
                        X_i = X_i + 1       
                        index_run = index_run + 1          
                    else :

                        # Search index in num_indices                   
                        index_in_indice1 = np.where(num_indices_row  == index_row )                    
                        # print('index_in_indice1 =  ', *index_in_indice1 )                       

                        index_in_indice2 = np.where(num_indices_col  == index_col)  
                        # print('index_in_indice2 =  ', *index_in_indice2 )      

                        if (np.array(index_in_indice1).size==0) and (np.array(index_in_indice2).size==0)  :     # If search = found ==> size > 0 (aka Not empty), not append.
            
                            num_indices_row.append(index_row)
                            num_indices_col.append(index_col)
                            # print('num_indices_row =  ', len(num_indices_row) )  
                        X_i = X_i + 1
                        index_run = index_run + 1          


        else :            
            # Possitive < Negative
            # print('Possitive < Negative' )  

            rev_index = np.where(coef_X_Y_input  == rev_coef_X_Y_flat[Y_i])

            # print('rev_index =  ', *rev_index )   
            # print('num_indices_row =  ', *num_indices_row )              
            # print('num_indices_col =  ', *num_indices_col )              
            if not (len(rev_index[0])==0) :     # rev_index is not empty

                # print('len_rev_index =  ', len(rev_index[0]))   
                index_lengh = len(rev_index[0])
                index_run = 0
                while index_run < index_lengh :
                    rev_index_row = (rev_index[0])[index_run]
                    rev_index_col = (rev_index[1])[index_run]

                    if not num_indices_row :        # num_indices is empty                   
                        # num_indices.append(np.array(rev_index))      # Append Negative   
                        # num_indices.append(np.array(rev_index))      # Append Negative   
                        num_indices_row.append(rev_index_row)
                        num_indices_col.append(rev_index_col)
                        Y_i = Y_i + 1       
                        index_run = index_run + 1          
                    else :

                        # Search rev_index in num_indices                   
                        index_in_indice1 = np.where(num_indices_row  == rev_index_row )                    
                        # print('index_in_indice1 =  ', *index_in_indice1 )                       

                        index_in_indice2 = np.where(num_indices_col  == rev_index_col)  
                        # print('index_in_indice2 =  ', *index_in_indice2 )      

                        if (np.array(index_in_indice1).size==0) and (np.array(index_in_indice2).size==0)  :     # If search = found ==> size > 0 (aka Not empty), not append.
            
                            num_indices_row.append(rev_index_row)
                            num_indices_col.append(rev_index_col)
                            # print('num_indices_col =  ', len(num_indices_col) )  

                        Y_i = Y_i + 1
                        index_run = index_run + 1          

        coef_indice = len(num_indices_row) + 1
    # End of While loop

    # print('coef_indice = ', coef_indice, ' X_i =' , X_i, " Y_i = " , Y_i , ' num_indices_row =  ', num_indices_row, ' num_indices_col =  ', num_indices_col )   
    print('coef_indice = ', coef_indice, ' X_i =' , X_i, " Y_i = " , Y_i , '  len_num_indices_row =  ', len(num_indices_row) , '  len_num_indices_col =  ', len(num_indices_col))   

    print('num_indices_row =  ', num_indices_row)
    # print('len_num_indices_row =  ', len(num_indices_row))

    print('num_indices_col =  ', num_indices_col)   
    # print('len_num_indices_col =  ', len(num_indices_col))

    return (num_indices_row, num_indices_col )      


def pica_LocalA_correlate(LocalA_Corr_X, LocalA_Corr_Y, \
    LocalA_Corr_MathLab_X, LocalA_Corr_MathLab_Y):



    def compute_covariance ( x, y ):    
        mean_x = np.mean( x )
        mean_y = np.mean( y )
        std_x  = np.std ( x )
        std_y  = np.std ( y )
        n      = len    ( x )

        # Return matrix as same size of one input
        return ((sum (x - mean_x ) * (y - mean_y ) ) * 1/(n-1))


    def compute_correlation_coefficients( x, y ):    
        mean_x = np.mean( x )
        mean_y = np.mean( y )
        std_x  = np.std ( x )
        std_y  = np.std ( y )
        n      = len    ( x )

        # Return Single
        return (( x - mean_x ) * ( y - mean_y )).sum() / n / ( std_x * std_y )

    def compute_correlation_coefficients_Pearson( x, y ):    
        mean_x = np.mean( x )
        mean_y = np.mean( y )
        std_x  = np.std ( x )
        std_y  = np.std ( y )
        n      = len    ( x )

        # Return (Pearson’s correlation coefficient , Two-tailed p-value) 
        return (compute_covariance(x, y) / (std_x * std_y))

    def compute_correlation_coefficients_Spearman( x, y ):    
        mean_x = np.mean( x )
        mean_y = np.mean( y )
        std_x  = np.std ( x )
        std_y  = np.std ( y )
        n      = len    ( x )

        # Return ( Spearman correlation matrix or correlation coefficient , Two-tailed p-value) 
        return (compute_covariance(np.rank(x), np.rank(y)) / (np.std(np.rank(x)) * np.std(np.rank(y))) )


    print('Modality_X === LocalA_mixer_X (sN_X x r_X)',LocalA_Corr_X.shape )    # (43, 8)
    print('Modality_Y === LocalA_mixer_Y (sN_Y x r_Y)',LocalA_Corr_Y.shape )    # (43, 8)

    print('Modality_X === LocalA_mixer_MathLab_X (sN_X x r_X)',LocalA_Corr_MathLab_X.shape )    # (43, 8)
    print('Modality_Y === LocalA_mixer_MathLab_Y (sN_Y x r_Y)',LocalA_Corr_MathLab_Y.shape  )   # (43, 8)


    # Correlation_Coefficients
    print('===== Correlation_Coefficients via np.corrcoef ')

    # print('===== Correlation_Coefficients via np.corrcoef === LocalA_Corr_MathLab_X, LocalA_Corr_MathLab_Y,')

    LocalA_coef_X_Y = np.corrcoef ( LocalA_Corr_X, LocalA_Corr_MathLab_X,  rowvar=False  )



   
    most_correlate = pica_most_correlate4(LocalA_coef_X_Y)   # num_list =  [ 1 13] [12 13] [ 0 11] [3 8] [ 1 12] [1 5] [0 1] [ 9 13]
###

    # plt.matshow(LocalA_coef_X_Y)
    # plt.xticks(range(LocalA_coef_X_Y.shape[0]), LocalA_coef_X_Y.shape[0])
    # plt.yticks(range(LocalA_coef_X_Y.shape[0]), LocalA_coef_X_Y.shape[0])
    # plt.colorbar()
    # plt.show()

    # print('===== Correlation_Coefficients via np.corrcoef === LocalA_Corr_X, LocalA_Corr_Y')
    # LocalA_coef_X_Y = np.corrcoef ( LocalA_Corr_X, LocalA_Corr_Y,  rowvar=False  )
    # most_correlate = pica_most_correlate2(LocalA_coef_X_Y)   # num_list =  [10 11] [ 9 10] [ 9 11] [ 8 11] [1 9] [ 8 10] [ 9 12] [ 8 15]

    plt.matshow(LocalA_coef_X_Y)
    # plt.xticks(range(LocalA_coef_X_Y.shape[0]), LocalA_coef_X_Y.shape[0])
    # plt.yticks(range(LocalA_coef_X_Y.shape[0]), LocalA_coef_X_Y.shape[0])
    plt.colorbar()
    plt.show()



    return ()   

        # pica_report_v1.pica_Source_correlate(S_source_X, S_source_Y, \
        #             self.S_source_MathLab_X, self.S_source_MathLab_Y)
