#######################################################
# This function is to create mark from nero-images.
# Ouput file is in NIFTI (nii) format.
# Researcher: Mart Panichvatana


########################################################
# This section is to call nibabel
# https://nipy.org/nibabel/gettingstarted.html
# Use nii image file from FITv2.0e in TReNDs.
import os
from os.path import dirname, join as pjoin
import nibabel as nib
from nibabel.testing import data_path
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
import scipy.io as sio
import re
from io import StringIO
from datetime import datetime, timedelta
from itertools import accumulate

# MASK_FILE_Name = 'mask_fmri_pica.nii.gz'
# MASK_FILE_Name = 'mask_fmri_pica_v3.nii.gz'
MASK_FILE_Name = "ABCD_mask_fmri_dpica_v1.nii.gz"


def pica_mask_creation(data_path_input, mask_path_input, b_plot_mask=False, b_plot_nii=False):

    # Define input and output directory path
    # data_path = "D:\\data\\Fusion_Data\\fmri_gene_full\\Healthy\\"
    # mask_path = "D:\\data\\Fusion_Data\\fmri_gene_full\\Mask\\"
    data_path = data_path_input
    mask_path = mask_path_input

    # Define brain-image dimention using one example image file.
    x_size = 0
    y_size = 0
    z_size = 0
    folder1 = os.listdir(data_path)
    # fmri_file = "con_0008.img"
    # fmri_path_file =  data_path + fmri_file
    fmri_all = []
    fmri_1d_all = []
    fmri_subject_all = []

    for subject_folders in folder1:
            fmri_subject_all.append(subject_folders)             
            folder2 = os.listdir(data_path + "//" + subject_folders)
            for fmri_folders in folder2:
                folder3 = os.listdir(data_path + "//" + subject_folders + \
                    "//" + fmri_folders)
                for img_file in folder3:
                    if img_file[-3:] == "img":
                        example_nii = os.path.join(data_path , subject_folders, \
                            fmri_folders,  img_file)
                        n1_img = nib.load(example_nii)
                        fmri_all = n1_img.get_fdata()
                        first_vol = fmri_all[:, :, :, 0] 
                        # print ("first_vol.shape =", first_vol.shape)
                        x_size, y_size, z_size = first_vol.shape
                        # first_vol.shape = (53, 63, 46)``
                        break    
                break
            break

    ########################################################
    # Loop all folders to collect ".img" file into fmri_all
    # to collect all nii file names.


    fmri_all = []
    fmri_1d_all = []
    fmri_subject_all = []
    mask_ind = []
    first_img_file = True


    for subject_folders in folder1:
        # if subject_folders[-3:] == "172" :  ## To filter for 1 file
        # if subject_folders[-1:] == "0" :  ## To filter for 4 file   
            fmri_subject_all.append(subject_folders)             
            folder2 = os.listdir(data_path + "//" + subject_folders)
            for fmri_folders in folder2:
                folder3 = os.listdir(data_path + "//" + subject_folders + \
                    "//" + fmri_folders)
                for img_file in folder3:
                    if img_file[-3:] == "img":
                        # print("img_file = ", data_path + "//" + subject_folders + \
                        #     "//" + fmri_folders + "//" + img_file + \
                        #     img_file + ". Shape = " + data.shape  )
                        data = nib.load(data_path + "//" + subject_folders + \
                            "//" + fmri_folders + "//" + img_file).get_fdata() 
                        data_3d_reshape = data.reshape(x_size, y_size, z_size)  
                        data_1d_reshape = data.reshape(x_size*y_size*z_size) 
                        fmri_all.append(data_3d_reshape)
                        fmri_1d_all.append(data_1d_reshape)
                        # print("img_file = " , img_file ,".shape = ", data.shape )
                        temp = np.isnan(fmri_1d_all)
                        # print ("temp = np.isnan(fmri_1d_all) = ", temp)  
                        # temp = np.where(temp != 0, )
                        temp = np.where(temp == 1, 0, 1 )
                        # print ("temp = np.where(temp == 0, 0, 1 ) = ", temp)  
                        if first_img_file :
                            mask_ind = temp
                            first_img_file = False
                            print ("mask_ind = ", mask_ind)                                   
                        else   :
                            # mask_ind = list(set(mask_ind) & set(data_1d_reshape))  
                            # mask_ind = np.where(mask_ind == data_1d_reshape, mask_ind, data_1d_reshape)
                            mask_ind = np.where(data_1d_reshape > mask_ind , 1, 0)

    # Check and define number of files
    # print("fmri_all = ", len(fmri_all) )
    # print("fmri_1d_all = ", len(fmri_1d_all) )
    fmri_1d_all_len = len(fmri_1d_all)
    print("fmri_1d_all_len = ", fmri_1d_all_len )
    fmri_1d_all = np.asarray(fmri_1d_all)
    print("fmri_1d_all.shape = ", fmri_1d_all.shape)

    mask_ind = np.asarray(mask_ind)
    print("mask_ind.shape = ", mask_ind.shape)      #  (1, 153594)
    print("mask_ind nonzero count =" , np.count_nonzero(mask_ind))    #  76123 ==> 64086  ==> 58179
    ########################################################
    # Compute Mask file
    ########################################################

    # Count Nan on each column
    fmri_1d_mask = np.count_nonzero(np.isnan(fmri_1d_all), axis=0)
    # fmri_1d_mask = np.count_nonzero(fmri_1d_all, axis=0)
    print("fmri_1d_mask.shape = ", fmri_1d_mask.shape)                      # (153594,)
    print (fmri_1d_mask)                                                    # [43 43 43 ... 43 43 43]

    # Creating mask=1 based on xx% of Nan condition
    # Higher percent = less mask (less gray)
    print("Masking at  1 of Nan  ")
    fmri_1d_mask = np.where(fmri_1d_mask == fmri_1d_all_len, 0, 1)
    print("fmri_1d_mask.shape = ", fmri_1d_mask.shape)                      # (153594,)
    print (fmri_1d_mask)                                                    # [0 0 0 ... 0 0 0]

    # Convert mask from 1D to 3D
    fmri_3d_mask = fmri_1d_mask.reshape(x_size, y_size, z_size)
    print("fmri_3d_mask.shape = ", fmri_3d_mask.shape)                      # (53, 63, 46)
    print("fmri_3d_mask isnan count =" , np.count_nonzero(fmri_3d_mask))    #  76123  ==> 58179
    # print (fmri_3d_mask)                                                    # [[[0 0 0 ... 0 0 0]

    # Save 3D mask file to local path
    # temp = np.ones((x_size, y_size, z_size, 1), dtype=np.int16)
    # fmri_3d_mask_saved = nib.Nifti1Image(temp, np.eye(4))
    fmri_3d_mask_saved = nib.Nifti1Image(fmri_3d_mask, np.eye(4))
    mask_fmri_pica_filename = MASK_FILE_Name
    mask_fmri_pica_filename_path = os.path.join(mask_path, mask_fmri_pica_filename)
    if os.path.exists(mask_fmri_pica_filename_path):
            os.remove(mask_fmri_pica_filename_path)
    nib.save(fmri_3d_mask_saved, mask_fmri_pica_filename_path)  
    print("Saving mask file. ")
    ########################################################
    # Plot Mask file in black/white 

    b_plot_mask = True
    if b_plot_mask :
        # Define Plot subgraph
        slice_display_number = 4

        # Convert mask array to an unsigned byte
        mask_uint8 = fmri_3d_mask.astype(np.uint8)  
        mask_uint8*=255
        # print("mask_uint8.shape = ", mask_uint8.shape)


        fig, ax = plt.subplots(1, slice_display_number, figsize=[15, 2])

        n = 0
        slice = 0
        for _ in range(slice_display_number):
            ax[n].imshow(mask_uint8[:, :, slice], 'gray')
            ax[n].set_title('Slice number: {}'.format(slice), color='black')
            if n == 0:
                ax[n].set_ylabel('Mask image (fmri_3d_mask)', fontsize=25, color='black')
            n += 1
            slice += 10

        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    if b_plot_mask :
        # Define Plot subgraph
        slice_display_number = 4

     
        mask_ind = mask_ind.reshape(x_size, y_size, z_size)
        # Convert mask array to an unsigned byte
        mask_uint8 = mask_ind.astype(np.uint8)  
        mask_uint8*=255
        # print("mask_uint8.shape = ", mask_uint8.shape)


        fig, ax = plt.subplots(1, slice_display_number, figsize=[15, 2])

        n = 0
        slice = 0
        for _ in range(slice_display_number):
            ax[n].imshow(mask_uint8[:, :, slice], 'gray')
            ax[n].set_title('Slice number: {}'.format(slice), color='black')
            if n == 0:
                ax[n].set_ylabel('Mask image (mask_ind)', fontsize=25, color='black')
            n += 1
            slice += 10

        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    ########################################################
    # Plot brain images on screen 

    if b_plot_nii :
        for img, subject in zip(fmri_all, fmri_subject_all) : 
            fig, ax = plt.subplots(1, slice_display_number, figsize=[15, 2])

            n = 0
            slice = 0
            for _ in range(slice_display_number):
                ax[n].imshow(img[:, :, slice], 'gray')
                ax[n].set_title('Slice number: {}'.format(slice), color='r')
                if n == 0:
                    ax[n].set_ylabel(subject, fontsize=25, color='b')
                n += 1
                slice += 10
                

            fig.subplots_adjust(wspace=0, hspace=0)
            # plt.show()


    ########################################################
    # Return
    output_mask_file_location = mask_path + MASK_FILE_Name
    return (output_mask_file_location, x_size, y_size, z_size)


def pica_mask_creation2(data_path_input, mask_path_input, b_plot_mask=False, b_plot_nii=False):

    # Define input and output directory path
    # data_path = "D:\\data\\Fusion_Data\\fmri_gene_full\\Healthy\\"
    # mask_path = "D:\\data\\Fusion_Data\\fmri_gene_full\\Mask\\"
    data_path = data_path_input
    mask_path = mask_path_input

    # Define brain-image dimention using one example image file.
    x_size = 0
    y_size = 0
    z_size = 0
    folder1 = os.listdir(data_path)
    # fmri_file = "con_0008.img"
    # fmri_path_file =  data_path + fmri_file
    fmri_all = []
    fmri_1d_all = []
    fmri_subject_all = []

    for subject_folders in folder1:
            fmri_subject_all.append(subject_folders)             
            folder2 = os.listdir(data_path + "//" + subject_folders)
            for fmri_folders in folder2:
                folder3 = os.listdir(data_path + "//" + subject_folders + \
                    "//" + fmri_folders)
                for img_file in folder3:
                    if img_file[-3:] == "img":
                        example_nii = os.path.join(data_path , subject_folders, \
                            fmri_folders,  img_file)
                        n1_img = nib.load(example_nii)
                        fmri_all = n1_img.get_fdata()
                        first_vol = fmri_all[:, :, :, 0] 
                        # print ("first_vol.shape =", first_vol.shape)
                        x_size, y_size, z_size = first_vol.shape
                        # first_vol.shape = (53, 63, 46)``
                        break    
                break
            break

    ########################################################
    # Loop all folders to collect ".img" file into fmri_all
    # to collect all nii file names.


    fmri_all = []
    fmri_1d_all = []
    fmri_subject_all = []
    mask_ind = []
    first_img_file = True


    for subject_folders in folder1:
        # if subject_folders[-3:] == "172" :  ## To filter for 1 file
        # if subject_folders[-1:] == "0" :  ## To filter for 4 file   
            fmri_subject_all.append(subject_folders)             
            folder2 = os.listdir(data_path + "//" + subject_folders)
            for fmri_folders in folder2:
                folder3 = os.listdir(data_path + "//" + subject_folders + \
                    "//" + fmri_folders)
                for img_file in folder3:
                    if img_file[-3:] == "img":

                        data = nib.load(data_path + "//" + subject_folders + \
                            "//" + fmri_folders + "//" + img_file).get_fdata() 
                        data_3d_reshape = data.reshape(x_size, y_size, z_size)  
                        data_1d_reshape = data.reshape(x_size*y_size*z_size) 
                        fmri_all.append(data_3d_reshape)
                        fmri_1d_all.append(data_1d_reshape)
                        # print("img_file = " , img_file ,".shape = ", data.shape )
                        # temp = np.isnan(fmri_1d_all)        # Find the non-NAN
                        temp = np.isnan(data_1d_reshape)        # Find the non-NAN
                        # print ("temp = np.isnan(data_1d_reshape)= ", temp)  
                        # temp = np.where(temp != 0, )
                        temp = np.where(temp != 0, 0, 1 )   # Find No-zero  Debug mask
                        # temp = np.where(temp == 1, 0, 1 )   # Find No-zero
                        # temp = np.where(temp == 1, 0, 1 )
                        # print ("temp = np.where(temp == 1, 0, 1 ) = ", temp)  
                        if first_img_file :
                            mask_ind = temp
                            first_img_file = False
                            # print ("mask_ind = ", mask_ind)                                   
                        else   :
                            # mask_ind = list(set(mask_ind) & set(data_1d_reshape))  
                            # mask_ind = np.where(mask_ind == data_1d_reshape, mask_ind, data_1d_reshape)
                            # mask_ind = np.where(data_1d_reshape == mask_ind , 1, 0)
                            mask_ind = np.logical_and(mask_ind,temp)
                            print ("mask_ind = ", mask_ind)     

    # Check and define number of files
    # print("fmri_all = ", len(fmri_all) )
    # print("fmri_1d_all = ", len(fmri_1d_all) )
    fmri_1d_all_len = len(fmri_1d_all)
    # print("fmri_1d_all_len = ", fmri_1d_all_len )
    fmri_1d_all = np.asarray(fmri_1d_all)
    print("fmri_1d_all.shape = ", fmri_1d_all.shape)

    mask_ind = np.asarray(mask_ind)
    print("mask_ind.shape = ", mask_ind.shape)      #  (1, 153594)
    print("mask_ind nonzero count =" , np.count_nonzero(mask_ind))    #  76123 ==> 64086  ==> 58179


    # Creating mask=1 based on xx% of Nan condition
    # Higher percent = less mask (less gray)
    print("Masking at  0 of Nan  ")
    mask_ind = np.where(mask_ind == 0, 0, 1)


    # Convert mask from 1D to 3D
    fmri_3d_mask = mask_ind.reshape(x_size, y_size, z_size)
    print("fmri_3d_mask.shape = ", fmri_3d_mask.shape)                      # (53, 63, 46)
    print("fmri_3d_mask isnan count =" , np.count_nonzero(fmri_3d_mask))    #  76123  ==> 58179
    # print (fmri_3d_mask)                                                    # [[[0 0 0 ... 0 0 0]

    # Save 3D mask file to local path
    fmri_3d_mask_saved = nib.Nifti1Image(fmri_3d_mask, np.eye(4))
    mask_fmri_pica_filename = MASK_FILE_Name
    mask_fmri_pica_filename_path = os.path.join(mask_path, mask_fmri_pica_filename)
    if os.path.exists(mask_fmri_pica_filename_path):
            os.remove(mask_fmri_pica_filename_path)
    nib.save(fmri_3d_mask_saved, mask_fmri_pica_filename_path)  
    print("Saving 58179 mask file. ")


    # Return
    output_mask_file_location = mask_path + MASK_FILE_Name
    return (output_mask_file_location, x_size, y_size, z_size)


def pica_mask_creation3(data_path_input, mask_path_input, b_plot_mask=False, b_plot_nii=False):

    # Define input and output directory path
    # data_path = "D:\\data\\Fusion_Data\\fmri_gene_full\\Healthy\\"
    # mask_path = "D:\\data\\Fusion_Data\\fmri_gene_full\\Mask\\"
    data_path = data_path_input
    mask_path = mask_path_input

    # Define brain-image dimention using one example image file.
    x_size = 0
    y_size = 0
    z_size = 0
    folder1 = os.listdir(data_path)
    # fmri_file = "con_0008.img"
    # fmri_path_file =  data_path + fmri_file
    fmri_all = []
    fmri_1d_all = []
    fmri_subject_all = []

    for subject_folders in folder1:
            fmri_subject_all.append(subject_folders)             
            baseline_folders = os.listdir(data_path_input + "//" + subject_folders  + "//Baseline")
            # for fmri_folders in baseline_folders:
            anat_found = False
            for anat_folders in baseline_folders:
                if anat_folders[:6] == "anat_2" and not(anat_found) :
                    nii_folders = os.listdir(data_path_input + "//" + subject_folders + "//Baseline" \
                        "//" + anat_folders)
                    anat_found = True
            if not anat_found :
                print("Please check ANAT folders!!!!!!!!!!!")
            else :
                for img_file in nii_folders:
                    if img_file == "Sm6mwc1pT1.nii":
                        # print("img_file = ", data_path_input + "//" + subject_folders + \
                        #     "//" + fmri_folders + "//" + img_file )
                        # n1_img = nib.load(data_path_input + "//" + subject_folders + "//Baseline" \
                        #     "//" + anat_folders + "//"  + img_file).get_fdata()   
                        # fmri_all = n1_img.get_fdata()
                        fmri_all = nib.load(data_path_input + "//" + subject_folders + "//Baseline" \
                            "//" + anat_folders + "//"  + img_file).get_fdata()   
                        # fmri_all = n1_img.get_fdata()
                        # first_vol = fmri_all[:, :, :, 0] 
                        first_vol = fmri_all[:, :, :] 
                        # print ("first_vol.shape =", first_vol.shape)
                        x_size, y_size, z_size = first_vol.shape
                        # first_vol.shape = (53, 63, 46)``
                        break    
                break
            break


    fmri_all = []
    fmri_1d_all = []
    fmri_subject_all = []
    mask_ind = []
    first_img_file = True

    for subject_folders in folder1:
            fmri_subject_all.append(subject_folders)             
            baseline_folders = os.listdir(data_path_input + "//" + subject_folders  + "//Baseline")
            # for fmri_folders in baseline_folders:
            anat_found = False
            for anat_folders in baseline_folders:
                # anat_found = False
                if anat_folders[:6] == "anat_2" and not(anat_found) :
                    nii_folders = os.listdir(data_path_input + "//" + subject_folders + "//Baseline" \
                        "//" + anat_folders)
                    anat_found = True
                    break
            if not anat_found :
                print("Please check ANAT folders!!!!!!!!!!!")
            else :
                for img_file in nii_folders:
                    if img_file == "Sm6mwc1pT1.nii":
                        # print("img_file = ", data_path_input + "//" + subject_folders + \
                        #     "//" + fmri_folders + "//" + img_file )
                        data = nib.load(data_path_input + "//" + subject_folders + "//Baseline" \
                            "//" + anat_folders + "//"  + img_file).get_fdata()   

                        data_3d_reshape = data.reshape(x_size, y_size, z_size)  
                        data_1d_reshape = data.reshape(x_size*y_size*z_size) 

                        print (" np.count_nonzero(data_1d_reshape)  = ", np.count_nonzero(data_1d_reshape)   )  #  np.count_nonzero(data_1d_reshape)  =  1242302
                        print (" np.sum(np.isnan(data_1d_reshape) )  = ",  np.sum(np.isnan(data_1d_reshape)) ) #  np.sum(np.isnan(data_1d_reshape) )  =  0

                        fmri_all.append(data_3d_reshape)    
                        fmri_1d_all.append(data_1d_reshape)
                        # print("img_file = " , img_file ,".shape = ", data.shape )
                        # temp = np.isnan(fmri_1d_all)        # Find the non-NAN

                        # print (" np.count_nonzero(data_1d_reshape)  = ", np.count_nonzero(data_1d_reshape)   )  #  np.count_nonzero(data_1d_reshape)  =  1242302
                        # print (" np.sum(np.isnan(data_1d_reshape) )  = ",  np.sum(np.isnan(data_1d_reshape)) ) #  np.sum(np.isnan(data_1d_reshape) )  =  0


                        # temp = np.isnan(data_1d_reshape)        # Find the non-NAN
                        temp = data_1d_reshape
                        # print ("temp = np.isnan(data_1d_reshape)= ", temp)  
                        # temp = np.where(temp != 0, )
                        # temp = np.where(temp == 1, 0, 1 )   # Find No-zero
                        # temp = np.where((temp==0)|(temp==1), temp^1, temp)
                        # temp = np.where(temp == 1, 1, 0 )   # Find No-zero
                        # temp = np.where(temp == 1, 1, 0 )   # Find No-zero
                        # temp = np.where(temp != 0, 0, 1 )   # Find No-zero  # Debug mask
                        print (" np.count_nonzero(temp)  = ", np.count_nonzero(temp)   )  #   np.count_nonzero(temp)  =  865848

                        # temp = np.where(temp == 1, 0, 1 )
                        # print ("temp = np.where(temp == 1, 0, 1 ) = ", temp)  
                        if first_img_file :
                            mask_ind = temp
                            first_img_file = False
                            # print ("mask_ind = ", mask_ind)                                   
                        else   :
                            # mask_ind = list(set(mask_ind) & set(data_1d_reshape))  
                            # mask_ind = np.where(mask_ind == data_1d_reshape, mask_ind, data_1d_reshape)
                            # mask_ind = np.where(data_1d_reshape == mask_ind , 1, 0)
                            mask_ind = np.logical_and(mask_ind,temp)
                            # print ("mask_ind = ", mask_ind)     

    # Check and define number of files
    # print("fmri_all = ", len(fmri_all) )
    # print("fmri_1d_all = ", len(fmri_1d_all) )
    fmri_1d_all_len = len(fmri_1d_all)
    # print("fmri_1d_all_len = ", fmri_1d_all_len )
    fmri_1d_all = np.asarray(fmri_1d_all)
    print("fmri_1d_all.shape = ", fmri_1d_all.shape)  # fmri_1d_all.shape =  (31, 2122945)

    mask_ind = np.asarray(mask_ind)
    print("mask_ind.shape = ", mask_ind.shape)      #  mask_ind.shape =  (2122945,)
    print("mask_ind nonzero count =" , np.count_nonzero(mask_ind))    # mask_ind nonzero count = 705142


    # Creating mask=1 based on xx% of Nan condition
    # Higher percent = less mask (less gray)
    print("Masking at  0 of Nan  ")
    mask_ind = np.where(mask_ind == 0, 0, 1)


    # Convert mask from 1D to 3D
    fmri_3d_mask = mask_ind.reshape(x_size, y_size, z_size)
    print("fmri_3d_mask.shape = ", fmri_3d_mask.shape)                      # fmri_3d_mask.shape =  (121, 145, 121)
    print("fmri_3d_mask isnan count =" , np.count_nonzero(fmri_3d_mask))    #  fmri_3d_mask isnan count = 705142
    # print (fmri_3d_mask)                                                    # [[[0 0 0 ... 0 0 0]

    # Save 3D mask file to local path
    fmri_3d_mask_saved = nib.Nifti1Image(fmri_3d_mask, np.eye(4))
    mask_fmri_pica_filename = MASK_FILE_Name
    mask_fmri_pica_filename_path = os.path.join(mask_path, mask_fmri_pica_filename)
    if os.path.exists(mask_fmri_pica_filename_path):
            os.remove(mask_fmri_pica_filename_path)
    nib.save(fmri_3d_mask_saved, mask_fmri_pica_filename_path)  
    print("Saving 2122945==>705142 mask file. ")


    ########################################################
    # Plot Mask file in black/white 

    b_plot_mask = True
    if b_plot_mask :
        # Define Plot subgraph
        slice_display_number = 4

        # Convert mask array to an unsigned byte
        mask_uint8 = fmri_3d_mask.astype(np.uint8)  
        mask_uint8*=255
        # print("mask_uint8.shape = ", mask_uint8.shape)


        fig, ax = plt.subplots(1, slice_display_number, figsize=[15, 2])

        n = 0
        slice = 0
        for _ in range(slice_display_number):
            ax[n].imshow(mask_uint8[:, :, slice], 'gray')
            ax[n].set_title('Slice number: {}'.format(slice), color='black')
            if n == 0:
                ax[n].set_ylabel('Mask image (fmri_3d_mask)', fontsize=25, color='black')
            n += 1
            slice += 10

        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()


    ########################################################
    # Plot brain images on screen 

    # b_plot_nii = True
    if b_plot_nii :
        for img, subject in zip(fmri_all, fmri_subject_all) : 
            fig, ax = plt.subplots(1, slice_display_number, figsize=[15, 2])

            n = 0
            slice = 0
            for _ in range(slice_display_number):
                ax[n].imshow(img[:, :, slice], 'gray')
                ax[n].set_title('Slice number: {}'.format(slice), color='r')
                if n == 0:
                    ax[n].set_ylabel(subject, fontsize=25, color='b')
                n += 1
                slice += 10
                

            fig.subplots_adjust(wspace=0, hspace=0)
            plt.show()

    # Return
    output_mask_file_location = mask_path + MASK_FILE_Name
    return (output_mask_file_location, x_size, y_size, z_size)

def pica_mask_creation5(data_path_input, mask_path_input, mask_path_file_name, b_plot_mask=False, b_plot_nii=False):

    # Define input and output directory path
    # data_path = "D:\\data\\Fusion_Data\\fmri_gene_full\\Healthy\\"
    # mask_path = "D:\\data\\Fusion_Data\\fmri_gene_full\\Mask\\"
    data_path = data_path_input
    mask_path = mask_path_input
    folderchar = "/"

    # Define brain-image dimention using one example image file.
    x_size = 0
    y_size = 0
    z_size = 0
    folder1 = os.listdir(data_path)
    # fmri_file = "con_0008.img"
    # fmri_path_file =  data_path + fmri_file
    fmri_all = []
    fmri_1d_all = []
    fmri_subject_all = []

    for subject_folders in folder1:
        baseline_path_name = os.path.join(data_path_input + subject_folders  + folderchar + "Baseline")
        if os.path.exists(baseline_path_name):
            fmri_subject_all.append(subject_folders)      
            baseline_folders = os.listdir(data_path_input + subject_folders  + folderchar + "Baseline")       
            # for fmri_folders in baseline_folders:
            anat_found = False
            for anat_folders in baseline_folders:
                if anat_folders[:6] == "anat_2" and not(anat_found) :
                    nii_folders = os.listdir(data_path_input +  subject_folders + folderchar + "Baseline" + \
                        folderchar + anat_folders)
                    anat_found = True
                    break
            if not anat_found :
                print("Please check ANAT folders!!!!!!!!!!!")
            else :
                for img_file in nii_folders:
                    if img_file == "Sm6mwc1pT1.nii":
                        # print("img_file = ", data_path_input + "//" + subject_folders + \
                        #     "//" + fmri_folders + "//" + img_file )
                        # n1_img = nib.load(data_path_input + "//" + subject_folders + "//Baseline" \
                        #     "//" + anat_folders + "//"  + img_file).get_fdata()   
                        # fmri_all = n1_img.get_fdata()
                        fmri_all = nib.load(data_path_input + subject_folders + folderchar + "Baseline" + \
                            folderchar + anat_folders + folderchar + img_file).get_fdata()   
                        # fmri_all = n1_img.get_fdata()
                        # first_vol = fmri_all[:, :, :, 0] 
                        first_vol = fmri_all[:, :, :] 
                        # print ("first_vol.shape =", first_vol.shape)
                        x_size, y_size, z_size = first_vol.shape
                        # first_vol.shape = (53, 63, 46)``
                        break    
                break
            break


    fmri_all = []
    fmri_1d_all = []
    fmri_subject_all = []
    mask_ind = []
    first_img_file = True
    number_nii = 0
    for subject_folders in folder1:
        baseline_path_name = os.path.join(data_path_input + subject_folders  + folderchar + "Baseline")
        if os.path.exists(baseline_path_name):
            fmri_subject_all.append(subject_folders)      
            baseline_folders = os.listdir(data_path_input + subject_folders  + folderchar + "Baseline")                
            # baseline_folders = os.listdir(data_path_input + subject_folders  + folderchar + "Baseline")
            # for fmri_folders in baseline_folders:
            anat_found = False
            for anat_folders in baseline_folders:
                # anat_found = False
                if anat_folders[:6] == "anat_2" and not(anat_found) :
                    nii_folders = os.listdir(data_path_input + subject_folders + folderchar + "Baseline" + \
                        folderchar + anat_folders)
                    anat_found = True
                    number_nii = number_nii + 1
                    break
            if not anat_found :
                print("Please check ANAT folders!!!!!!!!!!! ==>", str(subject_folders) )
            else :
                for img_file in nii_folders:
                    if img_file == "Sm6mwc1pT1.nii":
                        # print("img_file = ", data_path_input + "//" + subject_folders + \
                        #     "//" + fmri_folders + "//" + img_file )
                        data = nib.load(data_path_input + subject_folders + folderchar + "Baseline" + \
                            folderchar + anat_folders + folderchar  + img_file).get_fdata()   

                        data_3d_reshape = data.reshape(x_size, y_size, z_size)  
                        # data_1d_reshape = data.reshape(x_size*y_size*z_size) 

                        # print (" np.count_nonzero(data_1d_reshape)  = ", np.count_nonzero(data_1d_reshape)   )  #  np.count_nonzero(data_1d_reshape)  =  1242302
                        # print (" np.sum(np.isnan(data_1d_reshape) )  = ",  np.sum(np.isnan(data_1d_reshape)) ) #  np.sum(np.isnan(data_1d_reshape) )  =  0

                        fmri_3d_all = np.array(np.isnan(data_3d_reshape))

                        # fmri_all.append(data_3d_reshape)    
                        # fmri_1d_all.append(data_1d_reshape)
                        # print("img_file = " , img_file ,".shape = ", data.shape )
                        # temp = np.isnan(fmri_1d_all)        # Find the non-NAN

                        # print (" np.count_nonzero(data_1d_reshape)  = ", np.count_nonzero(data_1d_reshape)   )  #  np.count_nonzero(data_1d_reshape)  =  1242302
                        # print (" np.sum(np.isnan(data_1d_reshape) )  = ",  np.sum(np.isnan(data_1d_reshape)) ) #  np.sum(np.isnan(data_1d_reshape) )  =  0


                        # temp = np.isnan(data_1d_reshape)        # Find the non-NAN
                        # temp = data_1d_reshape
                        temp3d = np.where(fmri_3d_all != 0, 0, 1 )   # Find No-zero  Debug mask
                        # print("Folder=", subject_folders, "temp3d nonzero count =" , np.count_nonzero(temp3d))  


                        # print ("temp = np.isnan(data_1d_reshape)= ", temp)  
                        # temp = np.where(temp != 0, )
                        # temp = np.where(temp == 1, 0, 1 )   # Find No-zero
                        # temp = np.where((temp==0)|(temp==1), temp^1, temp)
                        # temp = np.where(temp == 1, 1, 0 )   # Find No-zero
                        # temp = np.where(temp == 1, 1, 0 )   # Find No-zero
                        # temp = np.where(temp != 0, 0, 1 )   # Find No-zero  # Debug mask
                        # print (" np.count_nonzero(temp)  = ", np.count_nonzero(temp)   )  #   np.count_nonzero(temp)  =  865848

                        # temp = np.where(temp == 1, 0, 1 )
                        # print ("temp = np.where(temp == 1, 0, 1 ) = ", temp)  
                        if first_img_file :
                            # mask_ind = temp
                            # first_img_file = False
                            # print ("mask_ind = ", mask_ind)       

                            # mask1d_ind = temp1d
                            mask3d_ind = temp3d
                            first_img_file = False
                            # print("mask3d_ind nonzero count =" , np.count_nonzero(mask3d_ind))  
                            print("No=", str(number_nii), "Folder=", subject_folders, "temp3d nonzero count =" , np.count_nonzero(temp3d) , \
                                    "mask3d_ind nonzero count =" , np.count_nonzero(mask3d_ind))                               
                            # print ("mask_ind = ", mask_ind)  

                        else   :
                            # mask_ind = list(set(mask_ind) & set(data_1d_reshape))  
                            # mask_ind = np.where(mask_ind == data_1d_reshape, mask_ind, data_1d_reshape)
                            # mask_ind = np.where(data_1d_reshape == mask_ind , 1, 0)
                            # mask_ind = np.logical_and(mask_ind,temp)
                            mask3d_ind = np.logical_and(mask3d_ind,temp3d)
                            # print("mask3d_ind nonzero count =" , np.count_nonzero(mask3d_ind))  
                            print("No=", str(number_nii), "Folder=", subject_folders, "temp3d nonzero count =" , np.count_nonzero(temp3d) , \
                                    "mask3d_ind nonzero count =" , np.count_nonzero(mask3d_ind))   

                            # print ("mask_ind = ", mask_ind)     
        else :
            print("Please check Baseline folders!!!!!!!!!!! ==>", str(subject_folders) )

    # Check and define number of files
    # print("fmri_all = ", len(fmri_all) )
    # print("fmri_1d_all = ", len(fmri_1d_all) )
    # fmri_1d_all_len = len(fmri_1d_all)
    # print("fmri_1d_all_len = ", fmri_1d_all_len )
    # fmri_1d_all = np.asarray(fmri_1d_all)
    # print("fmri_1d_all.shape = ", fmri_1d_all.shape)  # fmri_1d_all.shape =  (31, 2122945)

    # mask_ind = np.asarray(mask_ind)
    # print("mask_ind.shape = ", mask_ind.shape)      #  mask_ind.shape =  (2122945,)
    # print("mask_ind nonzero count =" , np.count_nonzero(mask_ind))    # mask_ind nonzero count = 705142

    mask3d_ind = (np.asarray(mask3d_ind)).astype(int)
    print("mask3d_ind.shape = ", mask3d_ind.shape)      #  (1, 153594)
    print("mask3d_ind nonzero count =" , np.count_nonzero(mask3d_ind))    #  76123 ==> 64086  ==> 58179



    # Creating mask=1 based on xx% of Nan condition
    # Higher percent = less mask (less gray)
    # print("Masking at  0 of Nan  ")
    # mask_ind = np.where(mask_ind == 0, 0, 1)


    # Convert mask from 1D to 3D
    # fmri_3d_mask = mask_ind.reshape(x_size, y_size, z_size)
    # print("fmri_3d_mask.shape = ", fmri_3d_mask.shape)                      # fmri_3d_mask.shape =  (121, 145, 121)
    # print("fmri_3d_mask isnan count =" , np.count_nonzero(fmri_3d_mask))    #  fmri_3d_mask isnan count = 705142
    # print (fmri_3d_mask)                                                    # [[[0 0 0 ... 0 0 0]

    # Save 3D mask file to local path
    fmri_3d_mask_saved = nib.Nifti1Image(mask3d_ind, np.eye(4))
    # fmri_3d_mask_saved = nib.Nifti1Image(mask3d_ind)
    # mask_fmri_pica_filename = MASK_FILE_Name
    # mask_fmri_pica_filename_path = os.path.join(mask_path, mask_fmri_pica_filename)
    mask_fmri_pica_filename_path = mask_path_file_name
    if os.path.exists(mask_fmri_pica_filename_path):
            os.remove(mask_fmri_pica_filename_path)
    nib.save(fmri_3d_mask_saved, mask_fmri_pica_filename_path)  
    print("====Saved mask file.====")


    ########################################################
    # Plot Mask file in black/white 

    # b_plot_mask = True
    if b_plot_mask :
        # Define Plot subgraph
        slice_display_number = 4

        # Convert mask array to an unsigned byte
        mask_uint8 = fmri_3d_mask_saved.astype(np.uint8)  
        mask_uint8*=255
        # print("mask_uint8.shape = ", mask_uint8.shape)


        fig, ax = plt.subplots(1, slice_display_number, figsize=[15, 2])

        n = 0
        slice = 0
        for _ in range(slice_display_number):
            ax[n].imshow(mask_uint8[:, :, slice], 'gray')
            ax[n].set_title('Slice number: {}'.format(slice), color='black')
            if n == 0:
                ax[n].set_ylabel('Mask image (fmri_3d_mask)', fontsize=25, color='black')
            n += 1
            slice += 10

        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()


    ########################################################
    # Plot brain images on screen 

    # b_plot_nii = True
    if b_plot_nii :
        for img, subject in zip(fmri_all, fmri_subject_all) : 
            fig, ax = plt.subplots(1, slice_display_number, figsize=[15, 2])

            n = 0
            slice = 0
            for _ in range(slice_display_number):
                ax[n].imshow(img[:, :, slice], 'gray')
                ax[n].set_title('Slice number: {}'.format(slice), color='r')
                if n == 0:
                    ax[n].set_ylabel(subject, fontsize=25, color='b')
                n += 1
                slice += 10
                

            fig.subplots_adjust(wspace=0, hspace=0)
            plt.show()

    # Return
    # output_mask_file_location = mask_path + MASK_FILE_Name
    return (mask_path_file_name, x_size, y_size, z_size)


def pica_masked_Modality_X_creation(NCOM_X_input, data_path_input, mask_file_location_input, \
    x_size, y_size, z_size):

    # Setup
    

    # Define input and output directory path

    mask_data = nib.load(mask_file_location_input).get_fdata()  #  53x63x46 
    # mask_data = np.loadtxt(mask_file_location_input, comments="#", delimiter=",", unpack=False)
    # mask_data = np.loadtxt(mask_file_location_input,  unpack=False)

    # mask_data = np.squeeze(mask_data, 3)
    # print("mask_data.shape = ", mask_data.shape )
    # print("mask_data = ", mask_data)
    # print("mask_data isnan count =" , np.count_nonzero(mask_data))    #   58179

    # plt.matshow( np.reshape(mask_data,(x_size*y_size,z_size)))
    # plt.colorbar()
    # plt.show()

    ########################################################
    # Loop all folders to collect ".img" file into fmri_all
    # to collect all nii file names.

    folder1 = os.listdir(data_path_input)
    fmri_all = []
    fmri_1d_all = []
    fmri_1d_masked_all = []
    # fmri_1d_mat_masked_all = []
    fmri_subject_all = []


    for subject_folders in folder1:
        # if subject_folders[-3:] == "172" :  ## To filter for 1 subject
        # if subject_folders[-1:] == "0" :  ## To filter for 4 subject   
            fmri_subject_all.append(subject_folders)             
            folder2 = os.listdir(data_path_input + "//" + subject_folders)
            for fmri_folders in folder2:
                folder3 = os.listdir(data_path_input + "//" + subject_folders + \
                    "//" + fmri_folders)
                for img_file in folder3:
                    if img_file[-3:] == "img":
                        # print("img_file = ", data_path_input + "//" + subject_folders + \
                        #     "//" + fmri_folders + "//" + img_file )
                        data = nib.load(data_path_input + "//" + subject_folders + \
                            "//" + fmri_folders + "//" + img_file).get_fdata()   
                        # Resize 4D to 3D
                        data = np.squeeze(data, 3)       # (53, 63, 46, 1)   ==> (53, 63, 46)
                        # Masking data
                        data_masked = data[mask_data > 0]  # != 0   153594  ==>(58179,)  mask =array([[0., 0., 0., ... 0., 0.
                        # data = np.reshape(data,(x_size*y_size*z_size))
                        data = data.T       # (53, 63, 46) ==> (46, 63, 53)
                        data = data[mask_data.T > 0] # (58179,)

                        # print("Masked data shape = ", data.shape )
                        # Resizing to Gold Standard
                        # fmri_data = data.reshape(x_size, y_size, z_size)
                        # fmri_1d_all.append(np.reshape(fmri_data,(x_size*y_size*z_size)))
                        # Addpending 
                        # fmri_data = np.squeeze(data, 1)
                        # print("Squeezed masked data shape = ", fmri_data.shape )
                        # fmri_1d_all.append(fmri_data)
                        fmri_1d_masked_all.append(data_masked)
                        # fmri_1d_mat_masked_all.append(data)

                        # print("img_file = " , img_file ,".shape = ", data.shape )

    # Check and define number of files
    # print("fmri_all = ", len(fmri_all) )
    # print("fmri_1d_mat_masked_all = ", len(fmri_1d_mat_masked_all) )
    # fmri_1d_mat_masked_all_len = len(fmri_1d_mat_masked_all)
    # fmri_1d_mat_masked_all = np.asarray(fmri_1d_mat_masked_all)
    # print("fmri_1d_mat_masked_all.shape = ", fmri_1d_mat_masked_all.shape)
    fmri_1d_masked_all = np.asarray(fmri_1d_masked_all)
    # print("fmri_1d_masked_all.shape = ", fmri_1d_masked_all.shape)


    ########################################################
    # Component Estimation
    # Akaike information criterion
    # https://en.wikipedia.org/wiki/Akaike_information_criterion 
    # Minimum description length
    # https://en.wikipedia.org/wiki/Minimum_description_length
    
    NCOM_X_ouput = NCOM_X_input      # Default
    
    ########################################################
    # Return  (clean_data_X, NSUB_X, NVOX_X, NCOM_X)
    return (fmri_1d_masked_all, fmri_1d_masked_all.shape[0], fmri_1d_masked_all.shape[1], NCOM_X_ouput)                


            # self.clean_data_Y1, self.NSUB_Y1, self.NVOX_Y1 , self.NCOM_Y1 = \
            #     pica_mask_v1.pica_Modality_Y_creation(data_path, \
            #         mask_file_location, x_size, y_size)

def pica_masked_Modality_X_creation2(NCOM_X_input, data_path_input, mask_file_location_input, \
    x_size, y_size, z_size):

    # Setup
    

    # Define input and output directory path
    # data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
    # mat_fname = pjoin(mask_file_location_input, 'testdouble_7.4_GLNX86.mat')
    mask_loadmat= sio.loadmat(mask_file_location_input)
    mask_data = mask_loadmat['mask_ind_3d_double']

    print(" ")
    # mask_data = nib.load(mask_file_location_input).get_fdata()
    # mask_data = np.loadtxt(mask_file_location_input, comments="#", delimiter=",", unpack=False)
    # mask_data = np.loadtxt(mask_file_location_input,  unpack=False)

    # mask_data = np.squeeze(mask_data, 3)
    # print("mask_data.shape = ", mask_data.shape )
    # print("mask_data = ", mask_data)
    # print("mask_data isnan count =" , np.count_nonzero(mask_data))    #   58179

    # plt.matshow( np.reshape(mask_data,(x_size*y_size,z_size)))
    # plt.colorbar()
    # plt.show()

    ########################################################
    # Loop all folders to collect ".img" file into fmri_all
    # to collect all nii file names.

    folder1 = os.listdir(data_path_input)
    fmri_all = []
    fmri_1d_all = []
    fmri_subject_all = []


    for subject_folders in folder1:
        # if subject_folders[-3:] == "172" :  ## To filter for 1 subject
        # if subject_folders[-1:] == "0" :  ## To filter for 4 subject   
            fmri_subject_all.append(subject_folders)             
            folder2 = os.listdir(data_path_input + "//" + subject_folders)
            for fmri_folders in folder2:
                folder3 = os.listdir(data_path_input + "//" + subject_folders + \
                    "//" + fmri_folders)
                for img_file in folder3:
                    if img_file[-3:] == "img":
                        # print("img_file = ", data_path_input + "//" + subject_folders + \
                        #     "//" + fmri_folders + "//" + img_file )
                        # data = nib.load(data_path_input + "//" + subject_folders + \
                        #     "//" + fmri_folders + "//" + img_file).get_fdata()   
                        data = nib.load(data_path_input + "//" + subject_folders + \
                            "//" + fmri_folders + "//" + img_file).get_fdata()   
                        # Masking data
                        # data = data[mask_data != 0]     # Marking wtih converting 3d to 1d automatically
                        data = np.squeeze(data, 3)
                        data = data[mask_data > 0]     # Marking wtih converting 3d to 1d automatically
                        # print("Masked data shape = ", data.shape )
                        # Resizing to Gold Standard
                        # fmri_data = data.reshape(x_size, y_size, z_size)
                        # fmri_1d_all.append(np.reshape(fmri_data,(x_size*y_size*z_size)))
                        # Addpending 
                        # fmri_data = np.squeeze(data, 1)
                        fmri_data = data
                        # print("Squeezed masked data shape = ", fmri_data.shape )
                        fmri_1d_all.append(fmri_data)
                        # print("img_file = " , img_file ,".shape = ", data.shape )

    # Check and define number of files
    # print("fmri_all = ", len(fmri_all) )
    # print("fmri_1d_all = ", len(fmri_1d_all) )
    # fmri_1d_all_len = len(fmri_1d_all)
    fmri_1d_all = np.asarray(fmri_1d_all)
    print("fmri_1d_all.shape = ", fmri_1d_all.shape)

    ########################################################
    # Component Estimation
    # Akaike information criterion
    # https://en.wikipedia.org/wiki/Akaike_information_criterion 
    # Minimum description length
    # https://en.wikipedia.org/wiki/Minimum_description_length
    
    NCOM_X_ouput = NCOM_X_input      # Default
    
    ########################################################
    # Return  (clean_data_X, NSUB_X, NVOX_X, NCOM_X)
    return (fmri_1d_all, fmri_1d_all.shape[0], fmri_1d_all.shape[1], NCOM_X_ouput)                


            # self.clean_data_Y1, self.NSUB_Y1, self.NVOX_Y1 , self.NCOM_Y1 = \
            #     pica_mask_v1.pica_Modality_Y_creation(data_path, \
            #         mask_file_location, x_size, y_size)


def pica_masked_Modality_X_creation3(NCOM_X_input, data_path_input, mask_file_location_input, \
    x_size, y_size, z_size):

    # Setup
    #       - Import fMRI file to numpy array. Automatically masking the file.
    # Input 
    #       - NCOM : Number of component
    #       - data_path_input : Path of fMRI data
    #       - mask_file_location_input : Path of Masked file.
    # Output : 
    #       - fmri_1d_masked_all, fmri_1d_mat_masked_all, fmri_1d_masked_all.shape[0], fmri_1d_masked_all.shape[1], NCOM_X_ouput)


    # Loading masked data
    mask_data = nib.load(mask_file_location_input).get_fdata()  #  53x63x46 
    # mask_data = np.loadtxt(mask_file_location_input, comments="#", delimiter=",", unpack=False)
    # mask_data = np.loadtxt(mask_file_location_input,  unpack=False)

    # mask_data = np.squeeze(mask_data, 3)
    print("mask_data.shape = ", mask_data.shape )   #  (121, 145, 121)
    # print("mask_data = ", mask_data)
    print("mask_data count_nonzero =" , np.count_nonzero(mask_data))    #   1158864
    print("mask_data")

    # plt.matshow( np.reshape(mask_data,(x_size*y_size,z_size)))
    # plt.colorbar()
    # plt.show()

    ########################################################
    # Loop all folders to collect ".img" file into fmri_all
    # to collect all nii file names.

    folder1 = os.listdir(data_path_input)
    fmri_all = []
    fmri_1d_all = []
    fmri_1d_masked_all = []
    # fmri_1d_mat_masked_all = []
    fmri_folder_file_name = []
    fmri_subject_all = []
    

    for subject_folders in folder1:
        # if subject_folders[-3:] == "172" :  ## To filter for 1 subject
        # if subject_folders[-1:] == "0" :  ## To filter for 4 subject   
            fmri_subject_all.append(subject_folders)             
            baseline_folders = os.listdir(data_path_input + "//" + subject_folders  + "//Baseline")
            # for fmri_folders in baseline_folders:
            anat_found = False
            for anat_folders in baseline_folders:
                if anat_folders[:6] == "anat_2" and not(anat_found) :
                    nii_folders = os.listdir(data_path_input + "//" + subject_folders + "//Baseline" \
                        "//" + anat_folders)
                    anat_found = True
                    break

            if not anat_found :
                print("Please check ANAT folders!!!!!!!!!!!")
            else :
                for img_file in nii_folders:
                    if img_file == "Sm6mwc1pT1.nii":
                        # print("img_file = ", data_path_input + "//" + subject_folders + \
                        #     "//" + fmri_folders + "//" + img_file )
                        current_folder_file_name = data_path_input + "//" + subject_folders + "//Baseline" \
                            "//" + anat_folders + "//"  + img_file
                        data = nib.load(current_folder_file_name).get_fdata()     # (121, 145, 121)
                        # print("data count_nonzero =" , np.count_nonzero(data))    #                          
                        # data = nib.load(data_path_input + "//" + subject_folders + "//Baseline" \
                        #     "//" + anat_folders + "//"  + img_file).get_fdata()   
                        # data.shape (121, 145, 121)  = 2,122,945
                        # Resize 4D to 3D
                        # data = np.squeeze(data, 3)       # (53, 63, 46, 1)   ==> (53, 63, 46)

                        # Masking data
                        data_masked = data[mask_data > 0]  # != 0   # (1158864,)
                        # data = np.reshape(data,(x_size*y_size*z_size))
                        # data = data.T       # (53, 63, 46) ==> (46, 63, 53)
                        data = data[mask_data.T > 0] # (58179,)

                        # print("Masked data shape = ", data.shape )
                        # Resizing to Gold Standard
                        # fmri_data = data.reshape(x_size, y_size, z_size)
                        # fmri_1d_all.append(np.reshape(fmri_data,(x_size*y_size*z_size)))
                        # Addpending 
                        # fmri_data = np.squeeze(data, 1)
                        # print("Squeezed masked data shape = ", fmri_data.shape )
                        # fmri_1d_all.append(fmri_data)
                        fmri_1d_masked_all.append(data_masked)
                        # fmri_1d_mat_masked_all.append(data)
                        fmri_folder_file_name.append(subject_folders)

                        # print("img_file = " , img_file ,".shape = ", data.shape )



    # Check and define number of files
    # print("fmri_all = ", len(fmri_all) )
    # print("fmri_1d_mat_masked_all = ", len(fmri_1d_mat_masked_all) )
    # fmri_1d_mat_masked_all_len = len(fmri_1d_mat_masked_all)
    # fmri_1d_mat_masked_all = np.asarray(fmri_1d_mat_masked_all)
    # print("fmri_1d_mat_masked_all.shape = ", fmri_1d_mat_masked_all.shape)
    fmri_1d_masked_all = np.asarray(fmri_1d_masked_all)
    fmri_folder_file_name = np.asarray(fmri_folder_file_name)
    print("fmri_1d_masked_all.shape = ", fmri_1d_masked_all.shape)
    print("fmri_folder_file_name.shape = ", fmri_folder_file_name.shape)


    ########################################################
    # Component Estimation
    # Akaike information criterion
    # https://en.wikipedia.org/wiki/Akaike_information_criterion 
    # Minimum description length
    # https://en.wikipedia.org/wiki/Minimum_description_length
    
    NCOM_X_ouput = NCOM_X_input      # Default
    
    ########################################################
    # Return  (clean_data_X, NSUB_X, NVOX_X, NCOM_X)
    return (fmri_1d_masked_all, fmri_folder_file_name, fmri_1d_masked_all.shape[0], fmri_1d_masked_all.shape[1], NCOM_X_ouput)                


            # self.clean_data_Y1, self.NSUB_Y1, self.NVOX_Y1 , self.NCOM_Y1 = \
            #     pica_mask_v1.pica_Modality_Y_creation(data_path, \
            #         mask_file_location, x_size, y_size)

                # pica_mask_v2.pica_masked_Modality_X_creation4(self.NCOM_X, data_path, data_site_path, data_sites, \
                #     mask_file_location, x_size, y_size, z_size)

def pica_masked_Modality_X_creation4(NCOM_X_input, data_path_input, data_site_path, data_sites, mask_file_location_input, \
    x_size, y_size, z_size):

    # Setup
    #       - Import fMRI file to numpy array. Automatically masking the file.
    # Input 
    #       - NCOM : Number of component
    #       - data_path_input : Path of fMRI data
    #       - mask_file_location_input : Path of Masked file.
    # Output : 
    #       - fmri_1d_masked_all, fmri_1d_mat_masked_all, fmri_1d_masked_all.shape[0], fmri_1d_masked_all.shape[1], NCOM_X_ouput)


    # Loading masked data
    mask_data = nib.load(mask_file_location_input).get_fdata()  #  53x63x46 
    # mask_data = np.loadtxt(mask_file_location_input, comments="#", delimiter=",", unpack=False)
    # mask_data = np.loadtxt(mask_file_location_input,  unpack=False)

    # mask_data = np.squeeze(mask_data, 3)
    print("mask_data.shape = ", mask_data.shape )   #  (121, 145, 121)
    # print("mask_data = ", mask_data)
    print("mask_data count_nonzero =" , np.count_nonzero(mask_data))    #   1158864
    print("mask_data")

    # plt.matshow( np.reshape(mask_data,(x_size*y_size,z_size)))
    # plt.colorbar()
    # plt.show()


    ########################################################
    # Pull site list from file
    # Data_sites_X = pica_import_subject_to_array( data_path_input, data_sites,) 
    Data_sites_X = pica_import_subject_to_array( data_site_path, data_sites,) 

    ########################################################
    # Loop all folders to collect ".img" file into fmri_all
    # to collect all nii file names.

    folder1 = os.listdir(data_path_input)
    fmri_all = []
    fmri_1d_all = []
    fmri_1d_masked_all = []
    # fmri_1d_mat_masked_all = []
    fmri_folder_file_name = []
    fmri_subject_all = []
    
    NUM_SUB = Data_sites_X.shape
    print ('Data_sites_X.shape = ', NUM_SUB)
    # for subject_folders in folder1:
    i = 0
    for subject_folders in Data_sites_X:
            i = i + 1
        # if subject_folders[-3:] == "172" :  ## To filter for 1 subject
        # if subject_folders[-1:] == "0" :  ## To filter for 4 subject   
            fmri_subject_all.append(subject_folders)             
            baseline_folders = os.listdir(data_path_input + "//" + subject_folders  + "//Baseline")
            # for fmri_folders in baseline_folders:
            anat_found = False



            for anat_folders in baseline_folders:
                if anat_folders[:6] == "anat_2" and not(anat_found) :
                    nii_folders = os.listdir(data_path_input + "//" + subject_folders + "//Baseline" \
                        "//" + anat_folders)
                    anat_found = True
                    break

            if not anat_found :
                print("Please check ANAT folders!!!!!!!!!!!")
            else :
                for img_file in nii_folders:
                    if img_file == "Sm6mwc1pT1.nii":
                        current_folder_file_name = data_path_input + "//" + subject_folders + "//Baseline" \
                            "//" + anat_folders + "//"  + img_file
                        # print("No ", i, " of " , NUM_SUB, " : img_file =", current_folder_file_name )
                        data = nib.load(current_folder_file_name).get_fdata()     # (121, 145, 121)
                        # print("data count_nonzero =" , np.count_nonzero(data))    #                          
                        # data = nib.load(data_path_input + "//" + subject_folders + "//Baseline" \
                        #     "//" + anat_folders + "//"  + img_file).get_fdata()   
                        # data.shape (121, 145, 121)  = 2,122,945
                        # Resize 4D to 3D
                        # data = np.squeeze(data, 3)       # (53, 63, 46, 1)   ==> (53, 63, 46)

                        # Masking data
                        data_masked = data[mask_data > 0]  # != 0   # (1158864,)
                        # data = np.reshape(data,(x_size*y_size*z_size))
                        # data = data.T       # (53, 63, 46) ==> (46, 63, 53)
                        data = data[mask_data.T > 0] # (58179,)

                        # print("Masked data shape = ", data.shape )
                        # Resizing to Gold Standard
                        # fmri_data = data.reshape(x_size, y_size, z_size)
                        # fmri_1d_all.append(np.reshape(fmri_data,(x_size*y_size*z_size)))
                        # Addpending 
                        # fmri_data = np.squeeze(data, 1)
                        # print("Squeezed masked data shape = ", fmri_data.shape )
                        # fmri_1d_all.append(fmri_data)
                        fmri_1d_masked_all.append(data_masked)
                        # fmri_1d_mat_masked_all.append(data)
                        fmri_folder_file_name.append(subject_folders)

                        # print("img_file = " , img_file ,".shape = ", data.shape )



    # Check and define number of files
    # print("fmri_all = ", len(fmri_all) )
    # print("fmri_1d_mat_masked_all = ", len(fmri_1d_mat_masked_all) )
    # fmri_1d_mat_masked_all_len = len(fmri_1d_mat_masked_all)
    # fmri_1d_mat_masked_all = np.asarray(fmri_1d_mat_masked_all)
    # print("fmri_1d_mat_masked_all.shape = ", fmri_1d_mat_masked_all.shape)
    fmri_1d_masked_all = np.asarray(fmri_1d_masked_all)
    fmri_folder_file_name = np.asarray(fmri_folder_file_name)
    print("fmri_1d_masked_all.shape = ", fmri_1d_masked_all.shape)
    print("fmri_folder_file_name.shape = ", fmri_folder_file_name.shape)


    ########################################################
    # Component Estimation
    # Akaike information criterion
    # https://en.wikipedia.org/wiki/Akaike_information_criterion 
    # Minimum description length
    # https://en.wikipedia.org/wiki/Minimum_description_length
    
    NCOM_X_ouput = NCOM_X_input      # Default
    
    ########################################################
    # Return  (clean_data_X, NSUB_X, NVOX_X, NCOM_X)
    return (fmri_1d_masked_all, fmri_folder_file_name, fmri_1d_masked_all.shape[0], fmri_1d_masked_all.shape[1], NCOM_X_ouput)                


            # self.clean_data_Y1, self.NSUB_Y1, self.NVOX_Y1 , self.NCOM_Y1 = \
            #     pica_mask_v1.pica_Modality_Y_creation(data_path, \
            #         mask_file_location, x_size, y_size)

    if os.path.exists(mask_fmri_pica_filename_path):
            os.remove(mask_fmri_pica_filename_path)

def pica_masked_Modality_X_creation4_sai(NCOM_X_input, data_path_input, data_site_path, data_sites, mask_file_location_input, \
    x_size, y_size, z_size):

    # Setup
    #       - Import fMRI file to numpy array. Automatically masking the file.
    # Input 
    #       - NCOM : Number of component
    #       - data_path_input : Path of fMRI data
    #       - mask_file_location_input : Path of Masked file.
    # Output : 
    #       - fmri_1d_masked_all, fmri_1d_mat_masked_all, fmri_1d_masked_all.shape[0], fmri_1d_masked_all.shape[1], NCOM_X_ouput)


    # Loading masked data
    # mask_data = nib.load(mask_file_location_input).get_fdata()  #  53x63x46 
    # # mask_data = np.loadtxt(mask_file_location_input, comments="#", delimiter=",", unpack=False)
    # # mask_data = np.loadtxt(mask_file_location_input,  unpack=False)

    # # mask_data = np.squeeze(mask_data, 3)
    # print("mask_data.shape = ", mask_data.shape )   #  (121, 145, 121)
    # # print("mask_data = ", mask_data)
    # print("mask_data count_nonzero =" , np.count_nonzero(mask_data))    #   1158864
    # print("mask_data")

    # plt.matshow( np.reshape(mask_data,(x_size*y_size,z_size)))
    # plt.colorbar()
    # plt.show()


    ########################################################
    # Pull site list from file
    # Data_sites_X = pica_import_subject_to_array( data_path_input, data_sites,) 
    # Data_sites_X = pica_import_subject_to_array( data_site_path, data_sites,) 

    ########################################################
    # Loop all folders to collect ".img" file into fmri_all
    # to collect all nii file names.
    # data_path_input = "/data/qneuromark/Data/ABCD/Data_BIDS/Raw_Data/NDARINV003RTV85/Baseline/anat_20181001100822"
    data_path_input = "/data/qneuromark/Data/ABCD/Data_BIDS/Raw_Data/NDARINV003RTV85/Baseline/anat_20181001100822/"
    folder1 = os.listdir(data_path_input)
    fmri_all = []
    fmri_1d_all = []
    fmri_1d_masked_all = []
    # fmri_1d_mat_masked_all = []
    fmri_folder_file_name = []
    fmri_subject_all = []
    
    # NUM_SUB = Data_sites_X.shape
    # print ('Data_sites_X.shape = ', NUM_SUB)
    # for subject_folders in folder1:
    i = 0
    for subject_folders in data_path_input:
            i = i + 1
        # if subject_folders[-3:] == "172" :  ## To filter for 1 subject
        # if subject_folders[-1:] == "0" :  ## To filter for 4 subject   
            # fmri_subject_all.append(subject_folders)             
            # baseline_folders = os.listdir(data_path_input + "//" + subject_folders  + "//Baseline")
            # # # for fmri_folders in baseline_folders:
            # # anat_found = False

            # nii_folders = data_path_input
            nii_folders = os.listdir(data_path_input)
            # for anat_folders in baseline_folders:
            #     if anat_folders[:6] == "anat_2" and not(anat_found) :
            #         nii_folders = os.listdir(data_path_input + "//" + subject_folders + "//Baseline" \
            #             "//" + anat_folders)
            #         anat_found = True
            #         break
            anat_found = True
            if not anat_found :
                print("Please check ANAT folders!!!!!!!!!!!")
            else :
                for img_file in nii_folders:
                    if img_file == "Sm6mwc1pT1.nii":
                        # current_folder_file_name = data_path_input + "//" + subject_folders + "//Baseline" \
                        #     "//" + anat_folders + "//"  + img_file
                        current_folder_file_name = data_path_input  + "//"  + img_file
                        # print("No ", i, " of " , NUM_SUB, " : img_file =", current_folder_file_name )
                        data = nib.load(current_folder_file_name).get_fdata()     # (121, 145, 121)
                        # print("data count_nonzero =" , np.count_nonzero(data))    #                          
                        # data = nib.load(data_path_input + "//" + subject_folders + "//Baseline" \
                        #     "//" + anat_folders + "//"  + img_file).get_fdata()   
                        # data.shape (121, 145, 121)  = 2,122,945
                        # Resize 4D to 3D
                        # data = np.squeeze(data, 3)       # (53, 63, 46, 1)   ==> (53, 63, 46)

                        # Masking data
                        data_masked = data[mask_data > 0]  # != 0   # (1158864,)
                        # data = np.reshape(data,(x_size*y_size*z_size))
                        # data = data.T       # (53, 63, 46) ==> (46, 63, 53)
                        data = data[mask_data.T > 0] # (58179,)

                        # print("Masked data shape = ", data.shape )
                        # Resizing to Gold Standard
                        # fmri_data = data.reshape(x_size, y_size, z_size)
                        # fmri_1d_all.append(np.reshape(fmri_data,(x_size*y_size*z_size)))
                        # Addpending 
                        # fmri_data = np.squeeze(data, 1)
                        # print("Squeezed masked data shape = ", fmri_data.shape )
                        # fmri_1d_all.append(fmri_data)
                        fmri_1d_masked_all.append(data_masked)
                        # fmri_1d_mat_masked_all.append(data)
                        fmri_folder_file_name.append(subject_folders)

                        # print("img_file = " , img_file ,".shape = ", data.shape )



    # Check and define number of files
    # print("fmri_all = ", len(fmri_all) )
    # print("fmri_1d_mat_masked_all = ", len(fmri_1d_mat_masked_all) )
    # fmri_1d_mat_masked_all_len = len(fmri_1d_mat_masked_all)
    # fmri_1d_mat_masked_all = np.asarray(fmri_1d_mat_masked_all)
    # print("fmri_1d_mat_masked_all.shape = ", fmri_1d_mat_masked_all.shape)
    fmri_1d_masked_all = np.asarray(fmri_1d_masked_all)
    fmri_folder_file_name = np.asarray(fmri_folder_file_name)
    print("fmri_1d_masked_all.shape = ", fmri_1d_masked_all.shape)
    print("fmri_folder_file_name.shape = ", fmri_folder_file_name.shape)


    ########################################################
    # Component Estimation
    # Akaike information criterion
    # https://en.wikipedia.org/wiki/Akaike_information_criterion 
    # Minimum description length
    # https://en.wikipedia.org/wiki/Minimum_description_length
    
    NCOM_X_ouput = NCOM_X_input      # Default
    
    ########################################################
    # Return  (clean_data_X, NSUB_X, NVOX_X, NCOM_X)
    return (fmri_1d_masked_all, fmri_folder_file_name, fmri_1d_masked_all.shape[0], fmri_1d_masked_all.shape[1], NCOM_X_ouput)                


            # self.clean_data_Y1, self.NSUB_Y1, self.NVOX_Y1 , self.NCOM_Y1 = \
            #     pica_mask_v1.pica_Modality_Y_creation(data_path, \
            #         mask_file_location, x_size, y_size)

    if os.path.exists(mask_fmri_pica_filename_path):
            os.remove(mask_fmri_pica_filename_path)

def pica_masked_Modality_X_creation5(NCOM_X_input, data_path_input, data_site_path, data_sites, mask_file_location_input, \
    x_size, y_size, z_size):

    # Setup
    #       - Import fMRI file to numpy array. Automatically masking the file.
    # Input 
    #       - NCOM : Number of component
    #       - data_path_input : Path of fMRI data
    #       - mask_file_location_input : Path of Masked file.
    # Output : 
    #       - fmri_1d_masked_all, fmri_1d_mat_masked_all, fmri_1d_masked_all.shape[0], fmri_1d_masked_all.shape[1], NCOM_X_ouput)


    # Loading masked data
    mask_data = nib.load(mask_file_location_input).get_fdata()  #  53x63x46 
    # mask_data = np.loadtxt(mask_file_location_input, comments="#", delimiter=",", unpack=False)
    # mask_data = np.loadtxt(mask_file_location_input,  unpack=False)

    # mask_data = np.squeeze(mask_data, 3)
    print("mask_data.shape = ", mask_data.shape )   #  (121, 145, 121)
    # print("mask_data = ", mask_data)
    print("mask_data count_nonzero =" , np.count_nonzero(mask_data))    #   1158864
    print("mask_data")

    # plt.matshow( np.reshape(mask_data,(x_size*y_size,z_size)))
    # plt.colorbar()
    # plt.show()


    ########################################################
    # Pull site list from file
    # Data_sites_X = pica_import_subject_to_array( data_path_input, data_sites,) 
    Data_sites_X = pica_import_subject_to_array( data_site_path, data_sites,) 

    ########################################################
    # Loop all folders to collect ".img" file into fmri_all
    # to collect all nii file names.

    folder1 = os.listdir(data_path_input)
    fmri_all = []
    fmri_1d_all = []
    fmri_1d_masked_all = []
    # fmri_1d_mat_masked_all = []
    fmri_folder_file_name = []
    fmri_subject_all = []
    
    NUM_SUB = Data_sites_X.shape
    print ('Data_sites_X.shape = ', NUM_SUB)
    # for subject_folders in folder1:
    i = 0
    for subject_folders in Data_sites_X:
        i = i + 1
        # if subject_folders[-3:] == "172" :  ## To filter for 1 subject
        # if subject_folders[-1:] == "0" :  ## To filter for 4 subject   
        
        subject_path = data_path_input + "//" + subject_folders  + "//Baseline"
        if os.path.exists(subject_path):
            fmri_subject_all.append(subject_folders)             
            baseline_folders = os.listdir(subject_path)
            # for fmri_folders in baseline_folders:
            anat_found = False


            for anat_folders in baseline_folders:
                if anat_folders[:6] == "anat_2" and not(anat_found) :
                    nii_folders = os.listdir(data_path_input + "//" + subject_folders + "//Baseline" \
                        "//" + anat_folders)
                    anat_found = True
                    break

            if not anat_found :
                print("Please check ANAT folders!!!!!!!!!!!")
            else :
                for img_file in nii_folders:
                    if img_file == "Sm6mwc1pT1.nii":
                        current_folder_file_name = data_path_input + "//" + subject_folders + "//Baseline" \
                            "//" + anat_folders + "//"  + img_file
                        print("No ", i, " of " , NUM_SUB, " : img_file =", current_folder_file_name )
                        data = nib.load(current_folder_file_name).get_fdata()     # (121, 145, 121)
                        # print("data count_nonzero =" , np.count_nonzero(data))    #                          
                        # data = nib.load(data_path_input + "//" + subject_folders + "//Baseline" \
                        #     "//" + anat_folders + "//"  + img_file).get_fdata()   
                        # data.shape (121, 145, 121)  = 2,122,945
                        # Resize 4D to 3D
                        # data = np.squeeze(data, 3)       # (53, 63, 46, 1)   ==> (53, 63, 46)

                        # Masking data
                        data_masked = data[mask_data > 0]  # != 0   # (1158864,)
                        # data = np.reshape(data,(x_size*y_size*z_size))
                        # data = data.T       # (53, 63, 46) ==> (46, 63, 53)
                        data = data[mask_data.T > 0] # (58179,)

                        # print("Masked data shape = ", data.shape )
                        # Resizing to Gold Standard
                        # fmri_data = data.reshape(x_size, y_size, z_size)
                        # fmri_1d_all.append(np.reshape(fmri_data,(x_size*y_size*z_size)))
                        # Addpending 
                        # fmri_data = np.squeeze(data, 1)
                        # print("Squeezed masked data shape = ", fmri_data.shape )
                        # fmri_1d_all.append(fmri_data)
                        fmri_1d_masked_all.append(data_masked)
                        # fmri_1d_mat_masked_all.append(data)
                        fmri_folder_file_name.append(subject_folders)

                        # print("img_file = " , img_file ,".shape = ", data.shape )
        else :
            print("Not found subject folder = " , str(subject_folders))


    # Check and define number of files
    # print("fmri_all = ", len(fmri_all) )
    # print("fmri_1d_mat_masked_all = ", len(fmri_1d_mat_masked_all) )
    # fmri_1d_mat_masked_all_len = len(fmri_1d_mat_masked_all)
    # fmri_1d_mat_masked_all = np.asarray(fmri_1d_mat_masked_all)
    # print("fmri_1d_mat_masked_all.shape = ", fmri_1d_mat_masked_all.shape)
    fmri_1d_masked_all = np.asarray(fmri_1d_masked_all)
    fmri_folder_file_name = np.asarray(fmri_folder_file_name)
    print("fmri_1d_masked_all.shape = ", fmri_1d_masked_all.shape)
    print("fmri_folder_file_name.shape = ", fmri_folder_file_name.shape)


    ########################################################
    # Component Estimation
    # Akaike information criterion
    # https://en.wikipedia.org/wiki/Akaike_information_criterion 
    # Minimum description length
    # https://en.wikipedia.org/wiki/Minimum_description_length
    
    NCOM_X_ouput = NCOM_X_input      # Default
    
    ########################################################
    # Return  (clean_data_X, NSUB_X, NVOX_X, NCOM_X)
    return (fmri_1d_masked_all, fmri_folder_file_name, fmri_1d_masked_all.shape[0], fmri_1d_masked_all.shape[1], NCOM_X_ouput)                


            # self.clean_data_Y1, self.NSUB_Y1, self.NVOX_Y1 , self.NCOM_Y1 = \
            #     pica_mask_v1.pica_Modality_Y_creation(data_path, \
            #         mask_file_location, x_size, y_size)


def pica_Modality_Y_creation(NCOM_Y_input, data_path_input):

    # Define input and output directory path

    # mask_data = nib.load(snf_file_location_input).get_fdata()
    # mask_data = np.squeeze(mask_data, 3)
    # print("mask_data.shape = ", mask_data.shape )

    ########################################################
    # Loop all folders to collect ".asc" file into snp_all
    # to collect all asc file names.

    folder1 = os.listdir(data_path_input)
    # snp_all = []
    snp_1d_all = []
    # snp_subject_all = []


    for subject_folders in folder1:
        # if subject_folders[-3:] == "172" :  ## To filter for 1 subject
        # if subject_folders[-1:] == "0" :  ## To filter for 4 subject   
            # snp_subject_all.append(subject_folders)             
            folder2 = os.listdir(data_path_input + "//" + subject_folders)
            for snp_folders in folder2:
                folder3 = os.listdir(data_path_input + "//" + subject_folders + \
                    "//" + snp_folders)
                for img_file in folder3:
                    if img_file[-3:] == "asc":
                        # print("img_file = ", data_path_input + "//" + subject_folders + \
                        #     "//" + snp_folders + "//" + img_file  )      
                        # fo = open((data_path_input + "//" + subject_folders + \
                        #     "//" + snp_folders + "//" + img_file), 'r')
                        # file_content = fo.read().strip()
                        # file_content = file_content.replace('\r\n', ';')
                        # file_content = file_content.replace('\n', ';')
                        # file_content = file_content.replace('\r', ';')
                        # numpy.matrix(file_content)
                        # fo.close()


                        # Masking data
                        # data = data[mask_data > 0]
                        # Resizing to Gold Standard
                        # snp_data = data.reshape(x_size, y_size, z_size)
                        # snp_1d_all.append(np.matrix(file_content))
                        # print("img_file = " , img_file ,".shape = ", data.shape )

                        fname = os.path.join(data_path_input, subject_folders, snp_folders, img_file)
                        # print("snp_fname = ", fname )

                        snp_1d_all.append(np.genfromtxt(fname, dtype=float ))



    # Check and define number of files
    # print("snp_1d_all = ", len(snp_1d_all) )
    # snp_1d_all_len = len(snp_1d_all)
    snp_1d_all = np.asarray(snp_1d_all)
    # print("snp_1d_all.shape = ", snp_1d_all.shape)
    # snp_1d_all = np.squeeze(snp_1d_all, 2)
    # print("snp_1d_all.shape = ", snp_1d_all.shape)
    # print("snp_1d_all = ", snp_1d_all)

    ########################################################
    # Component Estimation
    # Akaike information criterion
    # https://en.wikipedia.org/wiki/Akaike_information_criterion 
    # Minimum description length
    # https://en.wikipedia.org/wiki/Minimum_description_length

    NCOM_Y_output = NCOM_Y_input     # Default
    
    ########################################################
    # Return  (clean_data_y, NSUB_Y, NVOX_Y, NCOM_Y)
    return (snp_1d_all, snp_1d_all.shape[0], snp_1d_all.shape[1], NCOM_Y_output)   


def pica_Modality_Y_creation2(NCOM_Y_input, data_path_input, snp_file_name, \
    subject_data_X1, x_size, y_size):


    fname = os.path.join(data_path_input, snp_file_name)

    snp_1d_all = []
    i = 0

    for subject_name in subject_data_X1:
        # print (subject_name)

        IID_text = "(.*)" + subject_name[4:] + "(.*)"
        print (subject_name , "  ==> ", IID_text )
        i = i + 1
        j = 0 
        
        snp_read_file = open(fname, "r")
        for line in snp_read_file:
            j = j+1

            if re.match(IID_text, line):
                # print (line)
                # snp_1d_all.append(line)
                sting_temp  = StringIO(line)
                snp_temp = np.genfromtxt(sting_temp)        # (388901,)



# Plot graph

                # # setting the ranges and no. of intervals
                # range = (0, 6)
                # bins = 2  
                
                # # plotting a histogram
                # plt.hist(snp_temp, bins, range, color = 'green',
                #         histtype = 'bar', rwidth = 0.5)
                
                # # x-axis label
                # plt.xlabel('SNP')
                # # frequency label
                # plt.ylabel('No. gene')
                # # plot title
                # plt.title('Histogram')
                # # function to show the plot
                # plt.show()

                # print ("Plot")
# Remove Nan
                # Remove Nan to 0
                snp_temp_num = np.nan_to_num(snp_temp)


                # #np.count_nonzero # counts values that is not 0\false
                # snp_temp_nan = np.count_nonzero(~np.isnan(snp_temp))
                # print ("np.count_nonzero(~np.isnan(data))= ", snp_temp_nan)   #  377571  # 109
                # print ("np.count_nonzero(np.isnan(data))= ", np.count_nonzero(np.isnan(snp_temp)))   # 11330  # 7

                # snp_temp_num = np.nan_to_num(snp_temp)
                # print ("np.count_nonzero(~np.isnan(snp_temp_num))= ", np.count_nonzero(~np.isnan(snp_temp_num)))   #  
                # print ("np.count_nonzero(np.isnan(snp_temp_num))= ", np.count_nonzero(np.isnan(snp_temp_num)) )   # 

                # temp = np.isnan(snp_temp)
                # print ("temp = np.isnan(fmri_1d_all) = ", temp)   # [False  True False ... False False False]
                # # temp = np.where(temp != 0, )
                # temp = np.where(temp == 1, 0, 1 )
                # # print ("temp = np.where(temp == 0, 0, 1 ) = ", temp)  
                # # mask_ind = np.where(data_1d_reshape > mask_ind , 1, 0)

# Plot graph
                snp_temp_num_non_id = np.delete(snp_temp_num,[0,1,2,3,4,5])
                # defining labels
                # activities = ['eat', 'sleep', 'work', 'play']
                
                # # portion covered by each label
                # slices = [3, 7, 8, 6]
                
                # # color for each label
                # colors = ['r', 'y', 'g', 'b']
                
                # # plotting the pie chart
                # plt.pie(snp_temp_num_graph,  
                #         startangle=90, shadow = True, 
                #         radius = 1.2)
                
                # # plotting legend
                # plt.legend()
                
                # # showing the plot
                # plt.show()

                # setting the ranges and no. of intervals
                # # range = (0, 6)
                # # bins = 2  
                
                # # plotting a histogram
                # # plt.hist(snp_temp_num_graph, bins, range, color = 'green',
                # #         histtype = 'bar', rwidth = 0.5)
                # plt.hist(snp_temp_num_graph,  bins='auto', color = 'green',
                #         histtype = 'bar', rwidth = 0.5)                
                
                # # x-axis label
                # plt.xlabel('SNP')
                # # frequency label
                # plt.ylabel('No. gene')
                # # plot title
                # plt.title('Histogram')
                # # function to show the plot
                # plt.show()

                # print ("Plot")

# Append new 

                # Append new snp record to Modality Y.
                snp_1d_all.append(snp_temp_num_non_id)
                # for no, row in enumerate(line.split(' ')[1:]):
                #     # snp_1d_all.append([])
                #     for elem in row.split(' '):
                #         snp_1d_all[no].append([(elem)])
                # print (subject_name, " is found at line = ", j , "." )


                break
        snp_read_file.close()
    print(' Break ')




    # Check and define number of files
    # print("snp_1d_all = ", len(snp_1d_all) )
    # snp_1d_all_len = len(snp_1d_all)
    snp_1d_all = np.asarray(snp_1d_all)
    print("snp_1d_all.shape = ", snp_1d_all.shape)   # shape:(2, 388901)
    # snp_1d_all = np.squeeze(snp_1d_all, 2)

    # snp_1d_all = np.delete(snp_1d_all,[0,1],1)
    # print("snp_1d_all.shape = ", snp_1d_all.shape)
    # print("snp_1d_all = ", snp_1d_all)

    ########################################################
    # Component Estimation
    # Akaike information criterion
    # https://en.wikipedia.org/wiki/Akaike_information_criterion 
    # Minimum description length
    # https://en.wikipedia.org/wiki/Minimum_description_length

    NCOM_Y_output = NCOM_Y_input     # Default
    
    ########################################################
    # Return  (clean_data_y, NSUB_Y, NVOX_Y, NCOM_Y)
    # return (snp_1d_all, snp_1d_all.shape[0], snp_1d_all.shape[1], NCOM_Y_output)   
    return (snp_1d_all, snp_1d_all.shape[0], snp_1d_all.shape[1], NCOM_Y_output)   


def pica_Modality_Y_creation3(NCOM_Y_input, data_path_input, snp_file_name, \
    data_ID_path_input, snp_ID_file_name, \
    subject_data_X1, x_size, y_size):

    ini_time = datetime.now()

    print('==============Loading ABAC ID file ==============')

    data_path = data_ID_path_input
    snp_id_file_name    = snp_ID_file_name

    # Define file name
    snp_id_path_fname = os.path.join(data_path, snp_id_file_name)
    print("snp_id_path_fname = ", snp_id_path_fname )

    # Reading file to array                                                                                 
    array_1d_ID = []
    array_1d_ID.append(np.genfromtxt(snp_id_path_fname, dtype=str ))
    array_1d_ID = np.asarray(array_1d_ID)     # shape:(1, 3, 22)
    array_1d_ID = np.squeeze(array_1d_ID, 0)  # shape:(3, 22)
    snp_id = (array_1d_ID)

    print('snp_id.size = ', snp_id.size)
    print('snp_id.size.shape = ', snp_id.shape)


    print('==============Loading ABAC SNP file ==============')

    data_path = data_path_input
    snp_file_name    = snp_file_name

    # Define file name
    snp_path_fname = os.path.join(data_path, snp_file_name)
    print("snp_path_fname = ", snp_path_fname )

    # Reading file to array                                                                                 
    array_1d_snp = []
    array_1d_snp.append(np.genfromtxt(snp_path_fname, dtype=str ))
    array_1d_snp = np.asarray(array_1d_snp)     # shape:(1, 3, 22)
    array_1d_ID = np.squeeze(array_1d_snp, 0)  # shape:(3, 22)
    snp_input = (array_1d_snp)

    print('snp_input.size = ', snp_input.size)
    print('snp_input.size.shape = ', snp_input.shape)

    # fname = os.path.join(data_path_input, snp_file_name)
    i = 0


    print('==============Start appending ==============')
    # NUM_APPEND_ID = subject_data_X1.shape[0]
    snp_2d_all = []
    i = 0
    for subject_id in subject_data_X1 : 
    # for i in range(NUM_APPEND_ID):
        # subject_id = np.array(subject_data_X1)[i]
        # subject_id = subject_id[4:]
        result = (np.where(snp_id == subject_id))[0]

        if (result.size > 0 ) :
            print('==Append No ', i, ' Subject = ', subject_id, ' : Index = ', str(result) , '==')
            snp_2d_all.append(snp_input[i,:])
            print('==Time : Appended successfully =' , str(datetime.now() - ini_time) ,  ': snp_2d_all shape =', np.array(snp_2d_all).shape, '==')  
            i = i + 1
        else:
            print('===Error: Not found this subject ID=', subject_id ,'.===') 
        # end of if




    # for subject_name in subject_data_X1:
    #     # print (subject_name)

    #     IID_text = "(.*)" + subject_name[4:] + "(.*)"
    #     print (subject_name , "  ==> ", IID_text )
    #     i = i + 1
    #     j = 0 
        
    #     snp_read_file = open(fname, "r")
    #     for line in snp_read_file:
    #         j = j+1

    #         if re.match(IID_text, line):
    #             # print (line)
    #             # snp_1d_all.append(line)
    #             sting_temp  = StringIO(line)
    #             snp_temp = np.genfromtxt(sting_temp)        # (388901,)

    #             # Remove Nan to 0
    #             snp_temp_num = np.nan_to_num(snp_temp)

    #             snp_temp_num_non_id = np.delete(snp_temp_num,[0,1,2,3,4,5])

    #             # Append new snp record to Modality Y.
    #             snp_1d_all.append(snp_temp_num_non_id)



    #             break
    #     snp_read_file.close()
    # print(' Break ')




    # Check and define number of files
    # print("snp_1d_all = ", len(snp_1d_all) )
    # snp_1d_all_len = len(snp_1d_all)
    snp_2d_all = np.asarray(snp_2d_all)
    print("snp_2d_all.shape = ", snp_2d_all.shape)   # shape:(2, 388901)
    # snp_1d_all = np.squeeze(snp_1d_all, 2)

    # snp_1d_all = np.delete(snp_1d_all,[0,1],1)
    # print("snp_1d_all.shape = ", snp_1d_all.shape)
    # print("snp_1d_all = ", snp_1d_all)

    ########################################################
    # Component Estimation
    # Akaike information criterion
    # https://en.wikipedia.org/wiki/Akaike_information_criterion 
    # Minimum description length
    # https://en.wikipedia.org/wiki/Minimum_description_length

    NCOM_Y_output = NCOM_Y_input     # Default
    
    ########################################################
    # Return  (clean_data_y, NSUB_Y, NVOX_Y, NCOM_Y)
    # return (snp_1d_all, snp_1d_all.shape[0], snp_1d_all.shape[1], NCOM_Y_output)   
    return (snp_2d_all, snp_2d_all.shape[0], snp_2d_all.shape[1], NCOM_Y_output)   


def pica_Modality_clean_data_creation_from_file1(data_path_input, data_file_name, \
    SITE_NUM, NSUB_list):

    data_2d_output = []

    data_2d_all = pica_import_csv_to_array(data_path_input, data_file_name) 

    print("data_2d_all.shape = ", data_2d_all.shape)   # shape:(777, 429655)

    if SITE_NUM == 3 :
            NSUB_start = 0
            NSUB_end = NSUB_list[1]
            data_1 = data_2d_all[:NSUB_end,:]    
            print("data_1.shape = ", data_1.shape)    

            NSUB_start = NSUB_list[1]
            NSUB_end = NSUB_list[2]
            data_2 = data_2d_all[NSUB_start:NSUB_end,:]
            print("data_2.shape = ", data_2.shape)    

            NSUB_start = NSUB_list[2]
            NSUB_end = 0
            data_3 = data_2d_all[NSUB_start:,:]
            print("data_3.shape = ", data_3.shape)    
    if SITE_NUM == 2 :
            NSUB_start = 0
            NSUB_end = NSUB_list[1]
            data_1 = data_2d_all[:NSUB_end,:]    
            print("data_1.shape = ", data_1.shape)    

            NSUB_start = NSUB_list[1]
            NSUB_end = 0
            data_2 = data_2d_all[NSUB_start:,:]
            print("data_2.shape = ", data_2.shape)  

    return (data_1, data_2)   

def pica_Modality_clean_data_creation_from_file2(data_path_input, data_file_name, \
    SITE_NUM, NSUB_list):

    data_2d_output = []

    data_2d_all = pica_import_csv_to_array(data_path_input, data_file_name) 

    print("data_2d_all.shape = ", data_2d_all.shape)   # shape:(777, 429655)

    if SITE_NUM == 3 :
            NSUB_start = 0
            NSUB_end = NSUB_list[1]
            data_1 = data_2d_all[:NSUB_end,:]    
            print("data_1.shape = ", data_1.shape)    

            NSUB_start = NSUB_list[1]
            NSUB_end = NSUB_list[2]
            data_2 = data_2d_all[NSUB_start:NSUB_end,:]
            print("data_2.shape = ", data_2.shape)    

            NSUB_start = NSUB_list[2]
            NSUB_end = 0
            data_3 = data_2d_all[NSUB_start:,:]
            print("data_3.shape = ", data_3.shape)    
    if SITE_NUM == 2 :
            NSUB_start = 0
            NSUB_end = NSUB_list[1]
            data_1 = data_2d_all[:NSUB_end,:]    
            print("data_1.shape = ", data_1.shape)    

            NSUB_start = NSUB_list[1]
            NSUB_end = 0
            data_2 = data_2d_all[NSUB_start:,:]
            print("data_2.shape = ", data_2.shape)  

    return (data_1, data_2)   

def pica_Modality_clean_data_creation_from_file3(data_path_input, data_file_name_input, \
    data_path_output, data_file_name_output, SITE_NUM, NSUB_list):

    data_2d_output = []
    NSUB_len = len(NSUB_list)
    NSUB_Accumate_All = list(accumulate(NSUB_list))

    ini_time = datetime.now()
    print("[LOG][Flow_1_Setup]=====Creating Local Modality - Reading " + str(data_file_name_input) + "=====")         
    data_2d_all = pica_import_csv_to_array(data_path_input, data_file_name_input) 
    print(str(data_file_name_input) + " shape = " + str(data_2d_all.shape) ) 
    print("[LOG][Flow_1_Setup]=====Creating Local Modality - Reading time is " + str(datetime.now() - ini_time) + "=====")   


    for run in range (NSUB_len):  
        ini_time = datetime.now()
        print("[LOG][Flow_1_Setup]=====Creating Local Modality - Local" + str(run) + "=====")         

        if run == 0 :   
            NSUB_start = 0
            NSUB_end = NSUB_list[run]
            data_2d_local = data_2d_all[NSUB_start:NSUB_end , :] 
            data_2d_output.append(data_2d_local)
        else :
            NSUB_start =  int(NSUB_Accumate_All[run-1])
            NSUB_end = int(NSUB_Accumate_All[run])                    
            data_2d_local = data_2d_all[NSUB_start:NSUB_end , :] 
            data_2d_output.append(data_2d_local)

        data_path_output_local = data_path_output + "/local" + str(run) + "/simulatorRun/"
        if not os.path.exists(data_path_output_local):
            os.makedirs(data_path_output_local)                                                 
        np.savetxt( str(data_path_output_local) + str(data_file_name_output) , data_2d_local, delimiter=",")

        print("[LOG][Flow_1_Setup]=====Creating Local Modality - Local" + str(run) + " - " + str(data_file_name_output) + \
            "  " + str(data_2d_local.shape) + "=====")   
        print("[LOG][Flow_1_Setup]=====Creating Local Modality - Local" + str(run) + " - " + str(data_file_name_output) + \
            "  Building time is " + str(datetime.now() - ini_time) + "=====")   

    return (data_2d_output)   


def pica_Modality_X_creation_from_file1(NCOM_X_input, data_path_input, fmri_file_name, \
    x_size, y_size):

    fmri_1d_all = []

    fmri_1d_all = pica_import_csv_to_array(data_path_input, fmri_file_name) 

    # print("fmri_1d_all.shape = ", fmri_1d_all.shape)   # shape:(2, 388901)

    NCOM_X_output = NCOM_X_input     # Default

    return (fmri_1d_all, fmri_1d_all.shape[0], fmri_1d_all.shape[1], NCOM_X_output)   

def pica_Modality_X_creation_from_file2(NCOM_X_input, data_path_input, fmri_file_name):

    fmri_1d_all = []

    fmri_1d_all = pica_import_csv_to_array(data_path_input, fmri_file_name) 

    # print("fmri_1d_all.shape = ", fmri_1d_all.shape)   # shape:(2, 388901)

    NCOM_X_output = NCOM_X_input     # Default

    return (fmri_1d_all, fmri_1d_all.shape[0], fmri_1d_all.shape[1], NCOM_X_output)   




def pica_Modality_Y_creation_from_file1(NCOM_Y_input, data_path_input, snp_file_name, \
    x_size, y_size):

    snp_1d_all = []

    snp_1d_all = pica_import_csv_to_array(data_path_input, snp_file_name) 

    # print("snp_1d_all.shape = ", snp_1d_all.shape)   # shape:(2, 388901)

    NCOM_Y_output = NCOM_Y_input     # Default

    return (snp_1d_all, snp_1d_all.shape[0], snp_1d_all.shape[1], NCOM_Y_output)   


def pica_Modality_XY_creation_from_file(NCOM_X_input, data_path_input, fmri_file_name):

    data_1d_all = []

    data_1d_all = pica_import_csv_to_array(data_path_input, fmri_file_name) 

    # print("data_1d_all.shape = ", data_1d_all.shape)   

    NCOM_X_output = NCOM_X_input     # Default

    return (data_1d_all, data_1d_all.shape[0], data_1d_all.shape[1], NCOM_X_output)   



def pica_import_data_to_array(data_path_input, file_name_input) :

    file_name = os.path.join(data_path_input, file_name_input)

    array_1d_output = []
    array_1d_output.append(np.genfromtxt(file_name, dtype=float ))

    array_1d_output = np.asarray(array_1d_output)
    array_1d_output = np.squeeze(array_1d_output, 0)
    # print("array_1d_output.shape = ", array_1d_output.shape)
    # print("array_1d_output = ", array_1d_output)
    return (array_1d_output)   

def pica_import_data_to_array2(data_path_input, file_name_input) :

    file_name = os.path.join(data_path_input, file_name_input)

    array_1d_output = []
    array_1d_output.append(np.genfromtxt(file_name, dtype=int ))
    array_1d_output = array_1d_output[0]

    return (array_1d_output)   

def pica_import_data_to_array3(data_path_input, file_name_input) :

    file_name = os.path.join(data_path_input, file_name_input)

    array_1d_output = []
    array_1d_output.append(np.genfromtxt(file_name, dtype='unicode' ))
    array_1d_output = array_1d_output[0]
    
    return (array_1d_output)   

def pica_import_data_to_array4(data_path_input, file_name_input) :

    file_name = os.path.join(data_path_input, file_name_input)

    array_1d_output = []
    array_1d_output.append(np.genfromtxt(file_name, dtype=float ))

    array_1d_output = array_1d_output[0]

    return (array_1d_output)   

def pica_import_data(data_path_input, file_name_input) :

    ########################################################
    # folder_name = os.listdir(data_path_input)
    file_name = os.path.join(data_path_input, file_name_input)

    array_1d_output = []
    array_1d_output.append(np.genfromtxt(file_name))

    # array_1d_output = np.asarray(array_1d_output)
    # array_1d_output = np.squeeze(array_1d_output, 0)
    # print("array_1d_output.shape = ", array_1d_output.shape)
    # print("array_1d_output = ", array_1d_output)

    ########################################################
    # return (array_1d_output, array_1d_output.shape[0], array_1d_output.shape[1])   
    return (array_1d_output)   




# def pica_import_csv_to_array(data_path_input, file_name_input) :
#     import csv
#     ########################################################
#     # folder_name = os.listdir(data_path_input)
#     file_name = os.path.join(data_path_input, file_name_input)

#     array_1d_output = []
#     with open(file_name, newline='') as csvfile:
#         array_1d_output = list(csv.reader(csvfile))

#     # array_1d_output = []
#     # array_1d_output.append(np.genfromtxt(file_name, dtype=float ))

#     # array_1d_output = np.asarray(array_1d_output)
#     array_1d_output = np.asarray(array_1d_output, dtype=float)
#     # np.array(x, dtype=float)
#     # array_1d_output = np.squeeze(array_1d_output, 0)
#     # print("array_1d_output.shape = ", array_1d_output.shape)
#     # print("array_1d_output = ", array_1d_output)

#     ########################################################
#     # return (array_1d_output, array_1d_output.shape[0], array_1d_output.shape[1])   
#     return (array_1d_output)   

def pica_import_csv_to_array(data_path_input, file_name_input, encode_me='') :
    import csv
    ########################################################
    # folder_name = os.listdir(data_path_input)
    file_name = os.path.join(data_path_input, file_name_input)

    array_1d_output = []
    if encode_me == '' :
        with open(file_name, newline='') as csvfile:
            array_1d_output = list(csv.reader(csvfile))
    else :
        with open(file_name, newline='', encoding=encode_me) as csvfile:
            array_1d_output = list(csv.reader(csvfile))

    # array_1d_output = []
    # array_1d_output.append(np.genfromtxt(file_name, dtype=float ))

    # array_1d_output = np.asarray(array_1d_output)

    array_1d_output = np.asarray(array_1d_output, dtype=float)
    # np.array(x, dtype=float)
    # array_1d_output = np.squeeze(array_1d_output, 0)
    # print("array_1d_output.shape = ", array_1d_output.shape)
    # print("array_1d_output = ", array_1d_output)

    ########################################################
    # return (array_1d_output, array_1d_output.shape[0], array_1d_output.shape[1])   
    return (array_1d_output)   

def pica_import_csv_to_array2(data_path_input, file_name_input, encode_me='') :
    import csv
    ########################################################
    # folder_name = os.listdir(data_path_input)
    file_name = os.path.join(data_path_input, file_name_input)

    array_1d_output = []
    if encode_me == '' :
        with open(file_name, newline='') as csvfile:
            array_1d_output = list(csv.reader(csvfile))
    else :
        with open(file_name, newline='', encoding=encode_me) as csvfile:
            array_1d_output = list(csv.reader(csvfile))

    # array_1d_output = []
    # array_1d_output.append(np.genfromtxt(file_name, dtype=float ))

    # array_1d_output = np.asarray(array_1d_output)

    # array_1d_output = np.asarray(array_1d_output, dtype=float)
    array_1d_output = np.asarray(array_1d_output)
    # np.array(x, dtype=float)
    # array_1d_output = np.squeeze(array_1d_output, 0)
    # print("array_1d_output.shape = ", array_1d_output.shape)
    # print("array_1d_output = ", array_1d_output)

    ########################################################
    # return (array_1d_output, array_1d_output.shape[0], array_1d_output.shape[1])   
    return (array_1d_output)   


def pica_import_subject_to_array(data_path_input, file_name_input) :
    file_name = os.path.join(data_path_input, file_name_input)
    array_1d_output = []
    array_1d_output = np.loadtxt(file_name, dtype=str, comments="#", delimiter=",", unpack=False)
    return (array_1d_output)   
