'''
Parallel Independent Component Analysis (pICA): (Liu et al. 2009)
This script computes pICA using the INFOMAX criteria.
The preprocessing steps include demeaning and whitening.
'''
import numpy as np
from numpy import dot
from numpy.linalg import matrix_rank, inv, matrix_rank, pinv
import decimal
from numpy.random import permutation
from scipy.linalg import eigh,  sqrtm, eig
from scipy.sparse.linalg import eigsh
from scipy.stats import t
from scipy.linalg import sqrtm
from scipy.linalg import norm as mnorm
from scipy.special import expit, logit
import time
import unittest
# import pica_modality2_infomax_v4
import dpica_mask_v4 as dpica_mask
import dpica_report_v5 as dpica_report
import matplotlib.pyplot as plt
import math
import os
import copy
from datetime import datetime, timedelta
from itertools import accumulate
  


# Switch case
# NUM_SUBJECT = 2
# NUM_SUBJECT = 43
# NUM_SUBJECT = 63
# NUM_SUBJECT = 100
# NUM_SUBJECT = 500
NUM_SUBJECT = 777
# NUM_SUBJECT = 1000        
# NUM_SUBJECT = 7000    
# NUM_SUBJECT = 8662
NSUB_LIST = [777]  #1
# NSUB_LIST = [388,389]  #2
# NSUB_LIST = [258,259,260]  #3
# NSUB_LIST = [194,194,194,195]  #4
# NSUB_LIST = [155,155,155,155,157]  #5
# LOCAL_COM_X = 65
# LOCAL_COM_Y = 29
# GLOBAL_COM_X = 65
# GLOBAL_COM_Y = 29
LOCAL_COM_X = 130
LOCAL_COM_Y = 58
GLOBAL_COM_X = 65
GLOBAL_COM_Y = 29
# LOCAL_COM_X = 125
# LOCAL_COM_Y = 150
# GLOBAL_COM_X = 25
# GLOBAL_COM_Y = 20
# LOCAL_COM_X = 260
# LOCAL_COM_Y = 145
# GLOBAL_COM_X = 65
# GLOBAL_COM_Y = 29

# NSUB_LIST = [129,129,129,129,129,132]  #6
# LOCAL_COM_X = 125
# LOCAL_COM_Y = 125
# GLOBAL_COM_X = 25
# GLOBAL_COM_Y = 20

# NSUB_LIST = [111,111,111,111,111,111,111]  #7
# LOCAL_COM_X = 110
# LOCAL_COM_Y = 110
# GLOBAL_COM_X = 25
# GLOBAL_COM_Y = 20

# NSUB_LIST = [97,97,97,97,97,97,97,98]  #8
# LOCAL_COM_X = 96
# LOCAL_COM_Y = 96
# GLOBAL_COM_X = 25
# GLOBAL_COM_Y = 20

# NSUB_LIST =[77,77,77,77,77,77,77,77,77,84] # 10
# LOCAL_COM_X = 130
# LOCAL_COM_Y = 58
# GLOBAL_COM_X = 65
# GLOBAL_COM_Y = 29

# NSUB_LIST = [38,38,	38,	38,	38,	38,	38,	38,	38,	38,	38,	38,	38,	38,	38,	38,	38,	38,	38,	55] #20
# LOCAL_COM_X = 35
# LOCAL_COM_Y = 30
# GLOBAL_COM_X = 25
# GLOBAL_COM_Y = 20

# NSUB_LIST =[25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,52] # 30
# LOCAL_COM_X = 25
# LOCAL_COM_Y = 25
# GLOBAL_COM_X = 25
# GLOBAL_COM_Y = 20

# DATASET = "A"
# DATASET = "B"
# DATASET = "C"
# DATASET = "F"
DATASET = "J"
# Experiment
CONSTRAINED_CONNECTION_NUM =  0.12  #0.15 # 0.20912 # 1 # 0.3006049888 #0.5 #0.4 # 0.3006049888 # 0.2542  #1  # 0.20912
CONSTRAINED_COMPONENTS_NUM =  1 # NUMBER OF COMPONENTS FROM EACH DATASET BEING CONSTRAINED
# CONSTRAINED_CONNECTION_AUTO_B = True
CONSTRAINED_CONNECTION_AUTO_B = False
ICASSO = True 
# ICASSO = False  
ICA_RUN_NUMBERS =  5

B_MODALITY_CREATIONB_1TO3 = False
B_MODALITY_CREATION_FROM_MATLAB = False  # False = Load Modality X and Y from MATLAB.
B_MODALITY_CREATION_FROM_FILE = True
B_MODALITY_CREATION_CLEAN_DATA_FROM_FILE = False     # First time to crate /local/ dataset
B_DATASET_777 = True
B_DATASET_FIT = not B_DATASET_777
B_DATASET_ABCD = not B_DATASET_777
B_GLOBAL = True
B_MATLAB_CLEAN_DATA_VALIDATION = False
B_MATLAB_PCA_FLIP_SIGN_X = False
B_MATLAB_PCA_FLIP_SIGN_Y = False
B_MATLAB_PCA_VALIDATION = False
B_MATLAB_ICA_DATA_VALIDATION = False
B_MATLAB_ICA_SPHERE_VALIDATION = False
B_MATLAB_ICA_WEIGHTAA_VALIDATION = False

B_MATLAB_ICA_VALIDATION = False
B_MATLAB_U_VALIDATION = False
B_MATLAB_WEIGHT_VALIDATION = False
B_MATLAB_LOCAL_A_VALIDATION = False

# B_LOCAL_PCA_MEAN_X = True
B_LOCAL_PCA_MEAN_X = False
# B_LOCAL_PCA_WHITE_X = True
B_LOCAL_PCA_WHITE_X = False
B_GLOBAL_PCA_MEAN_X = True
# B_GLOBAL_PCA_MEAN_X = False
B_GLOBAL_PCA_WHITE_X = True
# B_GLOBAL_PCA_WHITE_X = False

B_LOCAL_PCA_MEAN_Y = B_LOCAL_PCA_MEAN_X
B_LOCAL_PCA_WHITE_Y = B_LOCAL_PCA_WHITE_X
B_GLOBAL_PCA_MEAN_Y = B_GLOBAL_PCA_MEAN_X
B_GLOBAL_PCA_WHITE_Y = B_GLOBAL_PCA_WHITE_X

if B_LOCAL_PCA_MEAN_X  :  LMX = "T" 
else : LMX = "F" 
if B_LOCAL_PCA_WHITE_X :  LMX = "T" 
else : LWX = "F" 
if B_GLOBAL_PCA_MEAN_X :  GMX = "T" 
else : GMX = "F" 
if B_GLOBAL_PCA_WHITE_X:  GWX = "T" 
else : GWX = "F" 

if B_LOCAL_PCA_MEAN_Y  :  LMY = "T" 
else : LMY = "F" 
if B_LOCAL_PCA_WHITE_Y :  LWY = "T" 
else : LWY = "F" 
if B_GLOBAL_PCA_MEAN_Y :  GMY = "T" 
else : GMY = "F" 
if B_GLOBAL_PCA_WHITE_Y:  GWY = "T" 
else : GWY = "F" 

B_MW = LMX + LMX + GMX + GWX + LMY + LWY + GMY + GWY

DATASET_FIT_data_path = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/63/"
# ENDURANCE_C =  -1       # -1
# ENDURANCE_C =  -1e-1    # -0.1
# ENDURANCE_C =  -1e-3    # % the maximumlly allowed descending trend of entropy;  #  -1e-3 or -5e-4, or -1e-4  
# ENDURANCE_C =  -5e-4    # -0.0005
ENDURANCE_C =  -1e-4    # -0.0001
    # -1   -0.1   -0.001    -0.0005  -0.0001       
ENDURANCE_ABS = abs(ENDURANCE_C)

if CONSTRAINED_CONNECTION_AUTO_B : CONAUTO = "A" 
else : CONAUTO = CONSTRAINED_CONNECTION_NUM

if ICASSO : Multi = "I" 
else : Multi = "A"
if ICA_RUN_NUMBERS == 1 : Multi = "1"

if B_GLOBAL : Glo = "T"
else : Glo ="F"

# Global constants
ANNEAL = 0.90        # if weights blowup, restart with lrate
MAX_STEP = 1000  # 1200
SITE_NUM = len(NSUB_LIST)




MYFILENAME = "213Nc1" + str(SITE_NUM) + "_" + str(DATASET) + "_" + str(B_MW) + "_ConCon" + str(CONAUTO) +  "_ConCom" + str(CONSTRAINED_COMPONENTS_NUM) + "_RUN" + str(ICA_RUN_NUMBERS) + "_" + str(Multi) +  "_E"  +  str(ENDURANCE_ABS) + "_Global" +  str(Glo) + "_"  +  str(LOCAL_COM_X) + "_" + str(LOCAL_COM_Y) + "_" + str(GLOBAL_COM_X) + "_"  + str(GLOBAL_COM_Y) + "/"  


# Parameters above



if NUM_SUBJECT == 43 :
    ## 43 ALL 
    DATA_PATH_FROM_FILE = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/43/"
    DATA_PATH_X = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/43/Fusion_Data/fmri_gene_full/Healthy/"  
    DATA_PATH_Y = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/43/Fusion_Data/fmri_gene_full/Healthy/"  
    DATA_PATH_OUTPUT = "/data/users2/cpanichvatana1/dataset/output/output1_siteAll43_" + MYFILENAME
    DATA_SITES_X = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/43/"
    DATA_SITES = "siteAll43.txt" 

elif NUM_SUBJECT == 63 :
    ## 63 ALL 
    DATA_PATH_FROM_FILE = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/63/"
    DATA_PATH_X = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/63/Fusion_Data/fmri_gene_full/HealthySZ/"  
    DATA_PATH_Y = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/63/Fusion_Data/fmri_gene_full/HealthySZ/"  
    DATA_PATH_OUTPUT = "/data/users2/cpanichvatana1/dataset/output/output1_siteAll63_" + MYFILENAME
    DATA_SITES_X = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/63/"
    DATA_SITES = "siteAll63.txt" 
    DATA_PATH_SITE1 = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/63_2_1/Fusion_Data/fmri_gene_full/HealthySZ/"  
    DATA_PATH_SITE2 = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/63_2_2/Fusion_Data/fmri_gene_full/HealthySZ/"    

elif NUM_SUBJECT == 2 :
    ## 10 ALL 
    DATA_PATH_FROM_FILE = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data2/2/"
    DATA_PATH_X = "/data/collaboration/NeuroMark2/Data/ABCD/Data_BIDS/Raw_Data/"
    DATA_PATH_Y = "/data/users2/cpanichvatana1/dataset/ABCD/Release3"
    DATA_PATH_OUTPUT = "/data/users2/cpanichvatana1/dataset/output/output1_siteAll2_" + MYFILENAME
    DATA_SITES_X = "/data/users2/cpanichvatana1/dataset/ABCD/Data_BIDS2"
    DATA_SITES = "siteAll1000A2A.txt" 

elif NUM_SUBJECT == 10 :
    ## 10 ALL 
    DATA_PATH_FROM_FILE = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/10/"
    DATA_PATH_X = "/data/collaboration/NeuroMark2/Data/ABCD/Data_BIDS/Raw_Data/"
    DATA_PATH_Y = "/data/users2/cpanichvatana1/dataset/ABCD/ImputedSNP_QC2"
    DATA_PATH_OUTPUT = "/data/users2/cpanichvatana1/dataset/output/output1_siteAll10_" + MYFILENAME
    DATA_SITES_X = "/data/users2/cpanichvatana1/dataset/ABCD/Data_BIDS/"
    DATA_SITES = "siteAll10.txt" 

elif NUM_SUBJECT == 100 :
    ## 100 ALL 
    DATA_PATH_FROM_FILE = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/100/"
    DATA_PATH_X = "/data/collaboration/NeuroMark2/Data/ABCD/Data_BIDS/Raw_Data/"
    DATA_PATH_Y = "/data/users2/cpanichvatana1/dataset/ABCD/ImputedSNP_QC2"
    DATA_PATH_OUTPUT = "/data/users2/cpanichvatana1/dataset/output/output1_siteAll100_" + MYFILENAME
    DATA_SITES_X = "/data/users2/cpanichvatana1/dataset/ABCD/Data_BIDS/"
    DATA_SITES = "siteAll100.txt" 

elif NUM_SUBJECT == 500 :
    ## 500 ALL 
    DATA_PATH_FROM_FILE = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/500/"
    DATA_PATH_X = "/data/collaboration/NeuroMark2/Data/ABCD/Data_BIDS/Raw_Data/"
    DATA_PATH_Y = "/data/users2/cpanichvatana1/dataset/ABCD/ImputedSNP_QC2"
    DATA_PATH_OUTPUT = "/data/users2/cpanichvatana1/dataset/output/output1_siteAll500_" + MYFILENAME
    DATA_SITES_X = "/data/users2/cpanichvatana1/dataset/ABCD/Data_BIDS/"
    DATA_SITES = "siteAll500.txt" 


elif NUM_SUBJECT == 777 :
    ## 777 ALL 
    DATA_PATH_FROM_FILE = "/data/users2/cpanichvatana1/dataset/Chan_SNP_sMRI_1402/"
    DATA_PATH_X = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/63/Fusion_Data/fmri_gene_full/HealthySZ/"  
    DATA_PATH_Y = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/63/Fusion_Data/fmri_gene_full/HealthySZ/"  
    DATA_PATH_OUTPUT = "/data/users2/cpanichvatana1/dataset/output2/output2_site" + str(NUM_SUBJECT) + "_" + str(SITE_NUM) + "_" + MYFILENAME
    DATA_SITES_X = "/data/users2/cpanichvatana1/dataset/Chan_SNP_sMRI/"
    DATA_SITES = "siteAll777.txt" 
    # DATA_PATH_SITE1 = "/data/users2/cpanichvatana1/dataset/Chan_SNP_sMRI/" 
    # DATA_PATH_SITE2 = "/data/users2/cpanichvatana1/dataset/Chan_SNP_sMRI/"   
    DATA_PATH_SITE = "/data/users2/cpanichvatana1/dataset/Chan_SNP_sMRI_1402/" + str(SITE_NUM) + "/"  

elif NUM_SUBJECT == 1000 :
    # ## 1000 ALL 
    DATA_PATH_FROM_FILE = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/1000_" + str(DATASET) + "/" 
    DATA_PATH_X = "/data/collaboration/NeuroMark2/Data/ABCD/Data_BIDS/Raw_Data/"
    DATA_PATH_Y = "/data/users2/cpanichvatana1/dataset/ABCD/ImputedSNP_QC2"
    DATA_PATH_OUTPUT = "/data/users2/cpanichvatana1/dataset/output/output1_site1000_All_" + MYFILENAME
    DATA_SITES_X = "/data/users2/cpanichvatana1/dataset/ABCD/Data_BIDS/"
    # DATA_SITES = "siteAll1000setA.txt" 
    # DATA_SITES = "siteAll1000setB.txt" 
    # DATA_SITES = "siteAll1000setC.txt" 
    DATA_SITES = "siteAll1000set" + str(DATASET)  + ".txt" 


elif NUM_SUBJECT == 7000 :
    # 7000 ALL 
    DATA_PATH_FROM_FILE = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/7000/"
    DATA_PATH_X = "/data/collaboration/NeuroMark2/Data/ABCD/Data_BIDS/Raw_Data/"
    DATA_PATH_Y = "/data/users2/cpanichvatana1/dataset/ABCD/ImputedSNP_QC2"
    DATA_PATH_OUTPUT = "/data/users2/cpanichvatana1/dataset/output/output1_siteAll7000_" + MYFILENAME  
    DATA_SITES_X = "/data/users2/cpanichvatana1/dataset/ABCD/Data_BIDS/"
    DATA_SITES = "siteAll7000.txt" 

elif NUM_SUBJECT == 8662 :
    # 7000 ALL 
    DATA_PATH_FROM_FILE = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/8803/"
    DATA_PATH_X = "/data/collaboration/NeuroMark2/Data/ABCD/Data_BIDS/Raw_Data/"
    DATA_PATH_Y = "/data/users2/cpanichvatana1/dataset/ABCD/ImputedSNP_QC2"
    DATA_PATH_OUTPUT = "/data/users2/cpanichvatana1/dataset/output/output1_siteAll8803_" + MYFILENAME  
    DATA_SITES_X = "/data/users2/cpanichvatana1/dataset/ABCD/Data_BIDS/"
    DATA_SITES = "ABCD2_8662.txt" 

elif NUM_SUBJECT == 8803 :
    # 7000 ALL 
    DATA_PATH_FROM_FILE = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/8803/"
    DATA_PATH_X = "/data/collaboration/NeuroMark2/Data/ABCD/Data_BIDS/Raw_Data/"
    DATA_PATH_Y = "/data/users2/cpanichvatana1/dataset/ABCD/ImputedSNP_QC2"
    DATA_PATH_OUTPUT = "/data/users2/cpanichvatana1/dataset/output/output1_siteAll8803_" + MYFILENAME  
    DATA_SITES_X = "/data/users2/cpanichvatana1/dataset/ABCD/Data_BIDS/"
    DATA_SITES = "ABCD2_8803.txt" 

if ICASSO   :
    ICA_RUN_AVERAGE = False
    ICA_RUN_ICASSO = True
else:
    ICA_RUN_AVERAGE = True
    ICA_RUN_ICASSO = False    

ICA_RUN_COMPONENT_VALIDATION =  True

# Global PATHs
b_mask_creation = False         # Boolean to define if mask file creation is needed to create.
# MODALITY_Y_RAW_FILE_NAME = "ABCD_impQCed_maf0p01_new_rsnum_updated_clear5_prune0p5_recode.raw"
# MODALITY_Y_RAW_FILE_NAME = "ABCD_impQCed_maf0p01_new_rsnum_updated_clear5_sub7000.raw"
# MODALITY_Y_RAW_FILE_NAME = "ABCD_impQCed_maf0p01_new_rsnum_updated_clear5_sub1000.raw"
# MODALITY_Y_RAW_FILE_NAME = "ABCD_test_v1_test.txt"
MODALITY_Y_RAW_FILE_NAME = "abcd_r3_impute_maf001_3_nonrel_prun02_recodeA.raw.residuals.csv"
if B_DATASET_FIT :
    MASK_PATH_FILE_NAME = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/63/Fusion_Data/fmri_gene_full/Mask/mask_fmri_pica_v4.nii.gz"
    MASK_PATH_X = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/63/Fusion_Data/fmri_gene_full/Mask/" 
else:
    MASK_PATH_FILE_NAME = "/data/users2/cpanichvatana1/dataset/ABCD/Mask/ABCD_mask_fmri_dpica_v2.nii.gz"
    MASK_PATH_X = "/data/users2/cpanichvatana1/dataset/ABCD/Mask/"  




##############  Declare defaults used below   ##############
MAX_WEIGHT           = 1e8;       # guess that weights larger than this have blown up
DEFAULT_STOP         = 0.000001;  # stop training if weight changes below this
DEFAULT_ANNEALDEG    = 60;        # when angle change reaches this value,
DEFAULT_ANNEALSTEP   = 0.90;      # anneal by multiplying lrate by this original 0,9 to 0.95 changed by JL
DEFAULT_EXTANNEAL    = 0.98;      # or this if extended-ICA
DEFAULT_MAXSTEPS     = MAX_STEP # 512;       # ]top training after this many steps 512
DEFAULT_MOMENTUM     = 0.0;       # default momentum weight

DEFAULT_BLOWUP       = 1000000000.0;   # = learning rate has 'blown up'
DEFAULT_BLOWUP_FAC   = 0.8;       # when lrate 'blows up,' anneal by this fac
DEFAULT_RESTART_FAC  = 0.9;       # if weights blowup, restart with lrate
# lower by this factor
MIN_LRATE            = 0.000001;  # if weight blowups make lrate < this, quit
MAX_LRATE            = 0.1;       # guard against uselessly high learning rate
# DEFAULT_LRATE        = 0.015./np.log(chans)


# heuristic default - may need adjustment
#   for large or tiny data sets!
# DEFAULT_BLOCK        = floor(sqrt(frames/3));  # heuristic default

# - may need adjustment!
# Extended-ICA option:
DEFAULT_EXTENDED     = 0;         # default off
DEFAULT_EXTBLOCKS    = 1;         # number of blocks per kurtosis calculation
DEFAULT_NSUB         = 1;         # initial default number of assumed sub-Gaussians
# for extended-ICA
DEFAULT_EXTMOMENTUM  = 0.5;       # momentum term for computing extended-ICA kurtosis
MAX_KURTSIZE         = 6000;      # max points to use in kurtosis calculation
MIN_KURTSIZE         = 2000;      # minimum good kurtosis size (flag warning)
SIGNCOUNT_THRESHOLD  = 25;        # raise extblocks when sign vector unchanged
# after this many steps
SIGNCOUNT_STEP       = 2;         # extblocks increment factor

DEFAULT_SPHEREFLAG   = 'on';      # use the sphere matrix as the default
#   starting weight matrix
DEFAULT_PCAFLAG      = 'off';     # don't use PCA reduction
DEFAULT_POSACTFLAG   = 'on';      # use posact()
DEFAULT_VERBOSE      = 1;         # write ascii info to calling screen
DEFAULT_BIASFLAG     = 1;         # default to using bias in the ICA update rule

#--constrained ICA parameters
# Experiment
# CONSTRAINED_COMPONENTS =  3 # NUMBER OF COMPONENTS FROM EACH DATASET BEING CONSTRAINED
# Experiment
CONSTRAINED_COMPONENTS =  CONSTRAINED_COMPONENTS_NUM # NUMBER OF COMPONENTS FROM EACH DATASET BEING CONSTRAINED
CONSTRAINED_CONNECTION =  CONSTRAINED_CONNECTION_NUM # 0.2542   #1  #0.5; # CORRELATION THRESHOLD TO BE CONSTRAINED; HIGH THRESHOLD WILL BE STRENGTHENED. 1 mean to do 2 x ICA X and Y
CONSTRAINED_CONNECTION_PROABILITY = 0.025 # 0.05
CONSTRAINED_CONNECTION_AUTO = CONSTRAINED_CONNECTION_AUTO_B #False     # True for calculating from subject number. p_to_r2 False for setting to 1 default
ENDURANCE = ENDURANCE_C #-1e-3 # % the maximumlly allowed descending trend of entropy;  #  -1e-3 or -5e-4, or -1e-4
CRATE_X = 1  # Weight change rate start point
CRATE_Y = 1  # Weight change rate start point
CRATE_PERCENT = 0.9              # Weight change rate 
CRATE_OUTBOUND_PERCENT = 0.9              # Weight change rate 

ICA_RUN_NUMBER =  ICA_RUN_NUMBERS


##############  Set up keyword default values  ##############


epochs = 1;							 # do not care how many epochs in data

pcaflag    = DEFAULT_PCAFLAG;
sphering   = DEFAULT_SPHEREFLAG;     # default flags
posactflag = DEFAULT_POSACTFLAG;
verbose    = DEFAULT_VERBOSE;
# block      = DEFAULT_BLOCK;          # heuristic default - may need adjustment!
# lrate      = DEFAULT_LRATE;
annealdeg  = DEFAULT_ANNEALDEG;
annealstep = 0;                      # defaults declared below
nochange   = DEFAULT_STOP;
momentum   = DEFAULT_MOMENTUM;
maxsteps   = DEFAULT_MAXSTEPS;

weights    = 0;                      # defaults defined below
# ncomps     = chans;
biasflag   = DEFAULT_BIASFLAG;

DEFAULT_EXTENDED   = DEFAULT_EXTENDED;
extblocks  = DEFAULT_EXTBLOCKS;
kurtsize_X   = MAX_KURTSIZE
kurtsize_Y   = MAX_KURTSIZE
signsbias  = 0.02;                   # bias towards super-Gaussian components
extmomentum= DEFAULT_EXTMOMENTUM;    # exp. average the kurtosis estimates
nsub       = DEFAULT_NSUB;

wts_blowup_X = 0;                      # flag =1 when weights too large
wts_blowup_Y = 0;                      # flag =1 when weights too large

wts_passed = 0;                      # flag weights passed as argument
#
Connect_threshold =CONSTRAINED_CONNECTION; # set a threshold to select columns constrained.
MaxComCon  =       CONSTRAINED_COMPONENTS
trendPara  = ENDURANCE; #depends on the requirement on connection; the more negative,the stronger the contrains ,that may cause overfitting





X_x_size = 53       # Given ABCD fMIR size
X_y_size = 63       # Given ABCD fMIR size
X_z_size = 46       # Given ABCD fMIR size


NCOM_X = LOCAL_COM_X          
NCOM_Y = LOCAL_COM_Y          

Global_NCOM_X = GLOBAL_COM_X
Global_NCOM_Y = GLOBAL_COM_Y


class test_ica_methods(unittest.TestCase):

    def setUp(self):
    

        print('==============Global Parameters==============')
        

        self.All_X = []
        self.All_Y = []

        self.clean_data_ALL_X = []
        self.NSUB_ALL_X = []
        self.NVOX_ALL_X = []
        self.NCOM_ALL_X = []            

        self.clean_data_ALL_Y = []
        self.NSUB_ALL_Y = []
        self.NVOX_ALL_Y = []
        self.NCOM_ALL_Y = []            

        self.U_ALL_X = []
        self.L_white_ALL_X = []
        self.L_dewhite_ALL_X = []

        self.U_ALL_Y = []
        self.L_white_ALL_Y = []
        self.L_dewhite_ALL_Y = []


        self.max_pair_output_X = []
        self.max_pair_output_Y = [] 
        self.max_pair_output_XY = []

        print('==============Global Parameters==============')
        print('SITE_NUM = ', SITE_NUM)    
        print('NUM_SUBJECT = ', NUM_SUBJECT)
        print('NSUB_LIST = ', NSUB_LIST)
        print('MASK_PATH_FILE_NAME = ', MASK_PATH_FILE_NAME)
        print('MODALITY_Y_RAW_FILE_NAME = ', MODALITY_Y_RAW_FILE_NAME)
        print('DATA_PATH_X = ', DATA_PATH_X)
        print('MASK_PATH_X = ', MASK_PATH_X)
        print('DATA_PATH_Y = ', DATA_PATH_Y)
        print('DATA_PATH_FROM_FILE = ', DATA_PATH_FROM_FILE)
        print('DATA_PATH_OUTPUT = ', DATA_PATH_OUTPUT)
        print('DATA_SITES_X = ', DATA_SITES_X)
        print('DATA_PATH_SITE = ', DATA_PATH_SITE)


        print('MAX_WEIGHT = ', MAX_WEIGHT)
        print('DEFAULT_STOP = ', DEFAULT_STOP)
        print('DEFAULT_ANNEALDEG = ', DEFAULT_ANNEALDEG)
        print('DEFAULT_ANNEALSTEP = ', DEFAULT_ANNEALSTEP)
        print('DEFAULT_EXTANNEAL = ', DEFAULT_EXTANNEAL)
        print('DEFAULT_MAXSTEPS = ', DEFAULT_MAXSTEPS)
        print('DEFAULT_MOMENTUM = ', DEFAULT_MOMENTUM)
        print('DEFAULT_BLOWUP = ', DEFAULT_BLOWUP)
        print('DEFAULT_BLOWUP_FAC = ', DEFAULT_BLOWUP_FAC)
        print('DEFAULT_RESTART_FAC = ', DEFAULT_RESTART_FAC)
        print('MIN_LRATE = ', MIN_LRATE)
        print('MAX_LRATE = ', MAX_LRATE)

        print('DEFAULT_EXTENDED = ', DEFAULT_EXTENDED)
        print('DEFAULT_EXTBLOCKS = ', DEFAULT_EXTBLOCKS)
        print('DEFAULT_NSUB = ', DEFAULT_NSUB)
        print('DEFAULT_EXTMOMENTUM = ', DEFAULT_EXTMOMENTUM)
        print('SIGNCOUNT_THRESHOLD = ', SIGNCOUNT_THRESHOLD)
        print('DEFAULT_SPHEREFLAG = ', DEFAULT_SPHEREFLAG)
        print('DATA_SDEFAULT_PCAFLAGITES = ', DEFAULT_PCAFLAG)
        print('DEFAULT_POSACTFLAG = ', DEFAULT_POSACTFLAG)
        print('DEFAULT_VERBOSE = ', DEFAULT_VERBOSE)
        print('DEFAULT_BIASFLAG = ', DEFAULT_BIASFLAG)
        print('CONSTRAINED_COMPONENTS = ', CONSTRAINED_COMPONENTS)
        print('CONSTRAINED_CONNECTION = ', CONSTRAINED_CONNECTION)
        print('CONSTRAINED_CONNECTION_AUTO = ', CONSTRAINED_CONNECTION_AUTO)
        print('DATA_ENDURANCESITES = ', ENDURANCE)
        print('NCOM_X = ', NCOM_X)
        print('NCOM_Y = ', NCOM_Y)  
   
        print('Global_NCOM_X = ', Global_NCOM_X)
        print('Global_NCOM_Y = ', Global_NCOM_Y)    
      
        print('signsbias = ', signsbias)
        print('ANNEAL = ', ANNEAL)
        print('MAX_STEP = ', MAX_STEP)
        print('DATA_SITES = ', DATA_SITES)
        print('ICA_RUN_NUMBER = ', ICA_RUN_NUMBER)
        print('ICA_RUN_AVERAGE = ', ICA_RUN_AVERAGE)
        print('ICA_RUN_ICASSO = ', ICA_RUN_ICASSO)
        print('ENDURANCE = ', ENDURANCE)
        print('CRATE_X = ', CRATE_X)
        print('CRATE_Y = ', CRATE_Y)
        print('CRATE_PERCENT = ', CRATE_PERCENT)
        print('CRATE_OUTBOUND_PERCENT = ', CRATE_OUTBOUND_PERCENT)
        print('B_DATASET_FIT = ', B_DATASET_FIT)
        print('B_DATASET_ABCD = ', B_DATASET_ABCD)
        print('B_DATASET_777 = ', B_DATASET_777)        

        print('B_LOCAL_PCA_MEAN_X = ', B_LOCAL_PCA_MEAN_X)        
        print('B_LOCAL_PCA_WHITE_X = ', B_LOCAL_PCA_WHITE_X)        
        print('B_GLOBAL_PCA_MEAN_X = ', B_GLOBAL_PCA_MEAN_X)       
        print('B_GLOBAL_PCA_WHITE_X = ', B_GLOBAL_PCA_WHITE_X)
        print('B_LOCAL_PCA_MEAN_Y = ', B_LOCAL_PCA_MEAN_Y)        
        print('B_LOCAL_PCA_WHITE_Y = ', B_LOCAL_PCA_WHITE_Y)        
        print('B_GLOBAL_PCA_MEAN_Y = ', B_GLOBAL_PCA_MEAN_Y)       
        print('B_GLOBAL_PCA_WHITE_Y = ', B_GLOBAL_PCA_WHITE_Y)
     

        #     # print('[LOG][def_setUp]+++++Set up finish+++++')

        #Setup        
        print('[LOG][def_setUp]+++++Set up start+++++')
        self.NCOM_X = NCOM_X  
        self.NCOM_Y = NCOM_Y  
 
        self.Global_NCOM_X = Global_NCOM_X  
        self.Global_NCOM_Y = Global_NCOM_Y  

       
        self.RUN_NUMBER = 1
        self.STEP=[0,0]
        self.maxsteps = maxsteps
        self.STOPSIGN=[0,0]
      
        self.mymaxcorr_list=[]
        self.myentropy_list_X=[]
        self.myentropy_list_Y=[]
        self.mySTEP_list_X=[]
        self.mySTEP_list_Y=[]
        self.mymaxcol_list=[]
        self.mymaxrow_list=[]

        if not os.path.exists(DATA_PATH_OUTPUT):
            os.makedirs(DATA_PATH_OUTPUT)


        print('[LOG][Flow_1_Setup]=====Creating default mask - Start =====')     
        #
        # Create mask file
        #
        # b_mask_creation = True
        b_mask_creation = False

        # Initial parameter values
        mask_file_location = MASK_PATH_FILE_NAME
        x_size = X_x_size
        y_size = X_y_size
        z_size = X_z_size
        if b_mask_creation :
            # Define mask input and output directory path
            # data_path = "D:\\data\\Fusion_Data\\fmri_gene_full\\Healthy\\"
            # mask_path = "D:\\data\\Fusion_Data\\fmri_gene_full\\Mask\\"   
            data_path = DATA_PATH_X
            mask_path = MASK_PATH_X 
            mask_path_file_name = MASK_PATH_FILE_NAME

            if B_DATASET_FIT :
                mask_path_file_name, x_size, y_size, z_size = dpica_mask.pica_mask_creation4(data_path, \
                      mask_path, mask_path_file_name, b_plot_mask=False, b_plot_nii=False)
            else:

                mask_path_file_name, x_size, y_size, z_size = dpica_mask.pica_mask_creation3(data_path, \
                     mask_path, mask_path_file_name, b_plot_mask=False, b_plot_nii=False)

        print('[LOG][Flow_1_Setup]=====Creating default mask - End  =====')      

        #Define initial parameter number
        # print('[LOG][Flow_1_Setup]=====Define initial parameter number=====')   

        if B_DATASET_777 :
            # # Create clean data X txt files
            b_Modality_clean_data_creation_X = B_MODALITY_CREATION_CLEAN_DATA_FROM_FILE

            if b_Modality_clean_data_creation_X and SITE_NUM > 1 :         

                print('[LOG][Flow_1_Setup]=====Creating Local Modality_X - From file =====')        
                ini_time = datetime.now()
                data_path_from_file = DATA_PATH_FROM_FILE
                file_name = "data_smri_T.csv"   
                    
                data_path_to_file = DATA_PATH_SITE 
                file_name_to_save = "data_smri_T_X.csv"
           
                if SITE_NUM > 1 :

                    self.clean_data_ALL_X =  dpica_mask.pica_Modality_clean_data_creation_from_file3(data_path_from_file, \
                            file_name, data_path_to_file, file_name_to_save, SITE_NUM, NSUB_LIST)        

                print('[LOG][Flow_1_Setup]=====Creating Local Modality_X - Created time is ' , str(datetime.now() - ini_time) ,  '=====')     
                


            # # Create clean data Y txt files
            b_Modality_clean_data_creation_Y = B_MODALITY_CREATION_CLEAN_DATA_FROM_FILE

            if b_Modality_clean_data_creation_Y and SITE_NUM > 1 :         

                print('[LOG][Flow_1_Setup]=====Creating Local Modality_Y - From file =====')        
                ini_time = datetime.now()
                data_path_from_file = DATA_PATH_FROM_FILE
                file_name = "data_snp.csv"   
                    
                data_path_to_file = DATA_PATH_SITE 
                file_name_to_save = "data_snp_Y.csv"
           
                if SITE_NUM > 1 :

                    self.clean_data_ALL_Y =  dpica_mask.pica_Modality_clean_data_creation_from_file3(data_path_from_file, \
                            file_name, data_path_to_file, file_name_to_save, SITE_NUM, NSUB_LIST)        

                print('[LOG][Flow_1_Setup]=====Creating Local Modality_Y - Created time is ' , str(datetime.now() - ini_time) ,  '=====')     
                
        print('[LOG][Flow_1_Setup]=====Loading Local Modality_X - Start =====')   

        if B_DATASET_777 :
            
            # # Option 1 from txt file
            b_Modality_creation_X = B_MODALITY_CREATION_FROM_FILE


            if b_Modality_creation_X :            

                # if SITE_NUM > 1 :                        
                for run in range (SITE_NUM):         
                    
                    ini_time = datetime.now()
                    print("[LOG][Flow_1_Setup]=====Loading Local Modality_X" + str(run+1) + " - From file =====")  
                    data_path = DATA_PATH_SITE + "/local" + str(run) + "/simulatorRun/"
                    file_name = "data_smri_T_X.csv"
                    # file_name = "data_smri_T_200_X.csv"  # For troubleshooting
                    clean_data_X1, NSUB_X1, NVOX_X1, NCOM_X1  = \
                        dpica_mask.pica_Modality_XY_creation_from_file(self.NCOM_X, data_path, file_name)   

                    self.NCOM_X = NCOM_X1 
                    self.clean_data_ALL_X.append(clean_data_X1) 
                    self.NSUB_ALL_X.append(NSUB_X1)
                    self.NVOX_ALL_X.append(NVOX_X1)
                    self.NCOM_ALL_X.append(NCOM_X1)     
                    if not os.path.exists(DATA_PATH_OUTPUT + "/local" + str(run) ):
                        os.makedirs(DATA_PATH_OUTPUT + "/local" + str(run) )                                                 
                    np.savetxt( DATA_PATH_OUTPUT + "/local" + str(run) + "/clean_data_X.csv", clean_data_X1, delimiter=",")

                    print("[LOG][Flow_1_Setup]=====Loading Local Modality_X" + str(run+1) + " - Loaded time is " + str(datetime.now() - ini_time) +  "=====")         
        
        # End if B_DATASET_777:


        if B_DATASET_FIT :
            # # Option 1 from txt file
            b_Modality_creation_X = B_MODALITY_CREATION_FROM_FILE

            if b_Modality_creation_X :            
                print('[LOG][Flow_1_Setup]=====Loading Local Modality_X1 - From file =====')        
                ini_time = datetime.now()
                data_path_from_file = DATA_PATH_FROM_FILE
                file_name = "clean_data_X1.csv"     # Site X1                          
                self.clean_data_X1, self.NSUB_X1, self.NVOX_X1 , self.NCOM_X1 = \
                    dpica_mask.pica_Modality_X_creation_from_file1(self.NCOM_X, data_path_from_file, \
                        file_name, x_size, y_size)        

                np.savetxt( DATA_PATH_OUTPUT + "clean_data_X1.csv", self.clean_data_X1, delimiter=",")
                print('[LOG][Flow_1_Setup]=====Loading Local Modality_X1 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')     
                
                self.NCOM_X = self.NCOM_X1 

                if SITE_NUM > 1 :                        

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X2 - From file =====')        
                    ini_time = datetime.now()
                    data_path_from_file = DATA_PATH_FROM_FILE
                    file_name = "clean_data_X2.csv"     # Site X2            
                    
                    self.clean_data_X2, self.NSUB_X2, self.NVOX_X2 , self.NCOM_X2 = \
                        dpica_mask.pica_Modality_X_creation_from_file1(self.NCOM_X, data_path_from_file, \
                            file_name, x_size, y_size)        
            
                    np.savetxt( DATA_PATH_OUTPUT + "clean_data_X2.csv", self.clean_data_X2, delimiter=",")
                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X2 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')    

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X3 - From file =====')        
                    ini_time = datetime.now()
                    data_path_from_file = DATA_PATH_FROM_FILE
                    file_name = "clean_data_X3.csv"     # Site X3            
                    
                    self.clean_data_X3, self.NSUB_X3, self.NVOX_X3 , self.NCOM_X3 = \
                        dpica_mask.pica_Modality_X_creation_from_file1(self.NCOM_X, data_path_from_file, \
                            file_name, x_size, y_size)        
                    
                    np.savetxt( DATA_PATH_OUTPUT + "clean_data_X3.csv", self.clean_data_X3, delimiter=",")
                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X3 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')    


            else :  # Option 2 from FIT folder

                # Define Modality_X 
                mask_path_file_name = MASK_PATH_FILE_NAME

                if SITE_NUM == 1 :  
                    # Site # 1           
                    data_path = DATA_PATH_X
                    self.clean_data_X1, self.clean_mat_data_X1, self.NSUB_X1, self.NVOX_X1 , self.NCOM_X1 = \
                        dpica_mask.pica_masked_Modality_X_creation(self.NCOM_X, data_path, \
                            mask_path_file_name, x_size, y_size, z_size)
                    self.NCOM_X = self.NCOM_X1 

                    np.savetxt( DATA_PATH_OUTPUT + "clean_data_X1.csv", self.clean_data_X1, delimiter=",")

                if SITE_NUM > 1 :                        
                    # Site # 1           
                    data_path = DATA_PATH_SITE1
                    self.clean_data_X1, self.clean_mat_data_X1, self.NSUB_X1, self.NVOX_X1 , self.NCOM_X1 = \
                        dpica_mask.pica_masked_Modality_X_creation(self.NCOM_X, data_path, \
                            mask_path_file_name, x_size, y_size, z_size)
                    self.NCOM_X = self.NCOM_X1 

                    np.savetxt( DATA_PATH_OUTPUT + "clean_data_X1.csv", self.clean_data_X1, delimiter=",")

                    # Site # 2
                    data_path = DATA_PATH_SITE2
                    self.clean_data_X2, self.clean_mat_data_X2, self.NSUB_X2, self.NVOX_X2 , self.NCOM_X2 = \
                        dpica_mask.pica_masked_Modality_X_creation(self.NCOM_X, data_path, \
                            mask_path_file_name, x_size, y_size, z_size)

                    np.savetxt( DATA_PATH_OUTPUT + "clean_data_X2.csv", self.clean_data_X2, delimiter=",")

                    # Site # 3
                    data_path = DATA_PATH_SITE3
                    self.clean_data_X3, self.clean_mat_data_X3, self.NSUB_X3, self.NVOX_X3 , self.NCOM_X3 = \
                        dpica_mask.pica_masked_Modality_X_creation(self.NCOM_X, data_path, \
                            mask_path_file_name, x_size, y_size, z_size)

                    np.savetxt( DATA_PATH_OUTPUT + "clean_data_X3.csv", self.clean_data_X3, delimiter=",")


                self.NCOM_X = self.NCOM_X1  
        
        # End if B_DATASET_FIT :

        if B_DATASET_ABCD :
            # # Option 1 from cvs file
            # b_Modality_creation_X = True
            b_Modality_creation_X = False
            if not (b_Modality_creation_X) :

            # Option 1 from cvs file  7000
                if NUM_SUBJECT == 7000 :
                    DATA_PATH_FROM_FILE2 = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/7000_3/"

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X1 - From file =====')        
                    ini_time = datetime.now()
                    data_path_from_file = DATA_PATH_FROM_FILE2
                    file_name = "clean_data_X1.csv"     # Site X1            
        
                    self.clean_data_X1, self.NSUB_X1, self.NVOX_X1 , self.NCOM_X1 = \
                        dpica_mask.pica_Modality_X_creation_from_file1(self.NCOM_X1, data_path_from_file, \
                            file_name, x_size, y_size)        
                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X1 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')     

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X2 - From file =====')        
                    ini_time = datetime.now()
                    data_path_from_file = DATA_PATH_FROM_FILE2
                    file_name = "clean_data_X2.csv"     # Site X2            
        
                    self.clean_data_X2, self.NSUB_X2, self.NVOX_X2 , self.NCOM_X2 = \
                        dpica_mask.pica_Modality_X_creation_from_file1(self.NCOM_X2, data_path_from_file, \
                            file_name, x_size, y_size)        
                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X2 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')    

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X3 - From file =====')        
                    ini_time = datetime.now()
                    data_path_from_file = DATA_PATH_FROM_FILE2
                    file_name = "clean_data_X3.csv"     # Site X3            
        
                    self.clean_data_X3, self.NSUB_X3, self.NVOX_X3 , self.NCOM_X3 = \
                        dpica_mask.pica_Modality_X_creation_from_file1(self.NCOM_X3, data_path_from_file, \
                            file_name, x_size, y_size)        
                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X3 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')    

                    ini_time = datetime.now()
                    self.clean_data_X1 = np.concatenate((self.clean_data_X1, self.clean_data_X2, self.clean_data_X3),axis = 0)
                    self.NSUB_X1 = 7000

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X - Concatenate time is ' , str(datetime.now() - ini_time) ,  '=====')          


                else:
        
                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X - From file =====')        
                    ini_time = datetime.now()

                    data_path_from_file = DATA_PATH_FROM_FILE
                    file_name = "clean_data_X1.csv"   
    
                    self.clean_data_X1, self.NSUB_X1, self.NVOX_X1 , self.NCOM_X1 = \
                        dpica_mask.pica_Modality_X_creation_from_file1(self.NCOM_X1, data_path_from_file, \
                            file_name, x_size, y_size)        

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')          


                if SITE_NUM > 1 :                        

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X2 - From file =====')        
                    ini_time = datetime.now()
                    data_path_from_file = DATA_PATH_FROM_FILE
                    file_name = "clean_data_X2.csv"     # Site X2            
                    
                    self.clean_data_X2, self.NSUB_X2, self.NVOX_X2 , self.NCOM_X2 = \
                        dpica_mask.pica_Modality_X_creation_from_file1(self.NCOM_X, data_path_from_file, \
                            file_name, x_size, y_size)        
            
                    np.savetxt( DATA_PATH_OUTPUT + "clean_data_X2.csv", self.clean_data_X2, delimiter=",")
                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X2 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')    

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X3 - From file =====')        
                    ini_time = datetime.now()
                    data_path_from_file = DATA_PATH_FROM_FILE
                    file_name = "clean_data_X3.csv"     # Site X3            
                    
                    self.clean_data_X3, self.NSUB_X3, self.NVOX_X3 , self.NCOM_X3 = \
                        dpica_mask.pica_Modality_X_creation_from_file1(self.NCOM_X, data_path_from_file, \
                            file_name, x_size, y_size)        
                    
                    np.savetxt( DATA_PATH_OUTPUT + "clean_data_X3.csv", self.clean_data_X3, delimiter=",")
                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_X3 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')    

            # Option 2 from ABCD folder
            # b_Modality_creation_X = True

            if b_Modality_creation_X : 
                data_path = DATA_PATH_X
                data_site_path = DATA_SITES_X
                data_sites = DATA_SITES
                mask_path_file_name = MASK_PATH_FILE_NAME

                self.clean_data_X1, self.clean_folder_file_name_data_X1, self.NSUB_X1, self.NVOX_X1 , self.NCOM_X1 = \
                    dpica_mask.pica_masked_Modality_X_creation4(self.NCOM_X1, data_path, data_site_path, data_sites, \
                        mask_path_file_name, x_size, y_size, z_size)
                np.savetxt( DATA_PATH_OUTPUT + "clean_data_X1.csv", self.clean_data_X1, delimiter=",")
        
        # End if B_DATASET_ABCD :


        # Loading Local Modality_Y
        print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y - Start =====')        

        if B_DATASET_777 :

            # # Option 1 from txt file
            b_Modality_creation_Y = B_MODALITY_CREATION_FROM_FILE

            if b_Modality_creation_Y :   

                for run in range (SITE_NUM):         
                    ini_time = datetime.now()
                    print("[LOG][Flow_1_Setup]=====Loading Local Modality_Y" + str(run+1) + " - From file =====")  
                    data_path = DATA_PATH_SITE + "/local" + str(run) + "/simulatorRun/"
                    file_name = "data_snp_Y.csv"
                    clean_data_Y1, NSUB_Y1, NVOX_Y1, NCOM_Y1  = \
                        dpica_mask.pica_Modality_XY_creation_from_file(self.NCOM_Y, data_path, file_name)   

                    self.NCOM_Y = NCOM_Y1 
                    self.clean_data_ALL_Y.append(clean_data_Y1) 
                    self.NSUB_ALL_Y.append(NSUB_Y1)
                    self.NVOX_ALL_Y.append(NVOX_Y1)
                    self.NCOM_ALL_Y.append(NCOM_Y1) 
                    if not os.path.exists(DATA_PATH_OUTPUT + "/local" + str(run) ):
                        os.makedirs(DATA_PATH_OUTPUT + "/local" + str(run) )                                                 
                    np.savetxt( DATA_PATH_OUTPUT + "/local" + str(run) + "/clean_data_Y.csv", clean_data_Y1, delimiter=",")

                    print("[LOG][Flow_1_Setup]=====Loading Local Modality_Y" + str(run+1) + " - Loaded time is " + str(datetime.now() - ini_time) +  "=====")    

        # End if B_DATASET_777 :

        if B_DATASET_FIT :

            # # Option 1 from txt file
            b_Modality_creation_Y = B_MODALITY_CREATION_FROM_FILE

            if b_Modality_creation_Y :   

                # Define Modality_Y      
                print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y1 - From file =====')   

                ini_time = datetime.now()
                data_path_from_file = DATA_PATH_FROM_FILE
                file_name = "clean_data_Y1.csv"     # Site Y1   
                        
                self.clean_data_Y1, self.NSUB_Y1, self.NVOX_Y1 , self.NCOM_Y1 = \
                    dpica_mask.pica_Modality_Y_creation_from_file1(self.NCOM_Y, data_path_from_file, \
                        file_name, x_size, y_size)

                print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y1 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')    


                if SITE_NUM > 1 : 
                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y2 - From file =====')     
                    ini_time = datetime.now()
                    data_path_from_file = DATA_PATH_FROM_FILE
                    file_name = "clean_data_Y2.csv"     # Site Y2            
                            
                    self.clean_data_Y2, self.NSUB_Y2, self.NVOX_Y2 , self.NCOM_Y2 = \
                        dpica_mask.pica_Modality_Y_creation_from_file1(self.NCOM_Y, data_path_from_file, \
                            file_name, x_size, y_size)

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y2 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')  


                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y3 - From file =====')     
                    ini_time = datetime.now()
                    data_path_from_file = DATA_PATH_FROM_FILE
                    file_name = "clean_data_Y3.csv"     # Site Y3            
                    
                    self.clean_data_Y3, self.NSUB_Y3, self.NVOX_Y3 , self.NCOM_Y3 = \
                        dpica_mask.pica_Modality_Y_creation_from_file1(self.NCOM_Y, data_path_from_file, \
                            file_name, x_size, y_size)

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y3 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')  

            else :  # Option 2 from FIT folder

                # Define Modality_Y      
                mask_path_file_name = MASK_PATH_FILE_NAME

                if SITE_NUM == 1 :     
                    # Site # 1   
                    data_path = DATA_PATH_Y
                    self.clean_data_Y1, self.NSUB_Y1, self.NVOX_Y1 , self.NCOM_Y1 = \
                        dpica_mask.pica_Modality_Y_creation(self.NCOM_Y, data_path)

                    self.NCOM_Y = self.NCOM_Y1   
                    np.savetxt( DATA_PATH_OUTPUT + "clean_data_Y1.csv", self.clean_data_Y1, delimiter=",")

                if SITE_NUM > 1 :
                     # Site # 1   
                    data_path = DATA_PATH_SITE1
                    self.clean_data_Y1, self.NSUB_Y1, self.NVOX_Y1 , self.NCOM_Y1 = \
                        dpica_mask.pica_Modality_Y_creation(self.NCOM_Y, data_path)

                    self.NCOM_Y = self.NCOM_Y1   
                    np.savetxt( DATA_PATH_OUTPUT + "clean_data_Y1.csv", self.clean_data_Y1, delimiter=",")
                    
                    # Site # 2 
                    data_path = DATA_PATH_SITE2
                    self.clean_data_Y2, self.NSUB_Y2, self.NVOX_Y2 , self.NCOM_Y2 = \
                        dpica_mask.pica_Modality_Y_creation(self.NCOM_Y, data_path)

                    np.savetxt( DATA_PATH_OUTPUT + "clean_data_Y2.csv", self.clean_data_Y2, delimiter=",")

                    # Site # 3
                    data_path = DATA_PATH_SITE3
                    self.clean_data_Y3, self.NSUB_Y3, self.NVOX_Y3 , self.NCOM_Y3 = \
                        dpica_mask.pica_Modality_Y_creation(self.NCOM_Y, data_path)

                    np.savetxt( DATA_PATH_OUTPUT + "clean_data_Y3.csv", self.clean_data_Y3, delimiter=",")
        
        # End if B_DATASET_FIT :

        if B_DATASET_ABCD :
            # # Option 1 from cvs file
            b_Modality_creation_Y = B_MODALITY_CREATION_FROM_MATLAB
            # b_Modality_creation_Y = True

            if not (b_Modality_creation_Y) :

                # Option 1 from cvs file  7000
                if NUM_SUBJECT == 7000 :

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y1 - From file =====')     
                    
                    DATA_PATH_FROM_FILE2 = "/data/users2/cpanichvatana1/dataset/ABCD/Clean_data/7000_3/"

                    ini_time = datetime.now()
                    data_path_from_file = DATA_PATH_FROM_FILE2
                    file_name = "clean_data_Y1.csv"     # Site Y1  

                    self.clean_data_Y1, self.NSUB_Y1, self.NVOX_Y1 , self.NCOM_Y1 = \
                        dpica_mask.pica_Modality_Y_creation_from_file1(self.NCOM_Y1, data_path_from_file, \
                            file_name, x_size, y_size)
                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y1 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')    


                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y2 - From file =====')     
                    ini_time = datetime.now()
                    data_path_from_file = DATA_PATH_FROM_FILE2
                    file_name = "clean_data_Y2.csv"     # Site Y2     

                    self.clean_data_Y2, self.NSUB_Y2, self.NVOX_Y2 , self.NCOM_Y2 = \
                        dpica_mask.pica_Modality_Y_creation_from_file1(self.NCOM_Y2, data_path_from_file, 
                            file_name, x_size, y_size)
                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y2 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')  


                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y3 - From file =====')     
                    ini_time = datetime.now()
                    data_path_from_file = DATA_PATH_FROM_FILE2
                    file_name = "clean_data_Y3.csv"     # Site Y3            

                    self.clean_data_Y3, self.NSUB_Y3, self.NVOX_Y3 , self.NCOM_Y3 = \
                        dpica_mask.pica_Modality_Y_creation_from_file1(self.NCOM_Y3, data_path_from_file, \
                            file_name, x_size, y_size)

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y3 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')  


                    ini_time = datetime.now()
                    self.clean_data_Y1 = np.concatenate((self.clean_data_Y1, self.clean_data_Y2, self.clean_data_Y3),axis = 0)
                    self.NSUB_Y1 = 7000

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y - Concatenate time is ' , str(datetime.now() - ini_time) ,  '=====')          

                else :

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y - From file =====')     
                    ini_time = datetime.now()

                    data_path_from_file = DATA_PATH_FROM_FILE
                    file_name = "clean_data_Y1.csv"     # Site Y All            
    
                    self.clean_data_Y1, self.NSUB_Y1, self.NVOX_Y1 , self.NCOM_Y1 = \
                        dpica_mask.pica_Modality_Y_creation_from_file1(self.NCOM_Y1, data_path_from_file, \
                            file_name, x_size, y_size)     

                    print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')          

                    if SITE_NUM > 1 : 
                        print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y2 - From file =====')     
                        ini_time = datetime.now()
                        data_path_from_file = DATA_PATH_FROM_FILE
                        file_name = "clean_data_Y2.csv"     # Site Y2            
                                
                        self.clean_data_Y2, self.NSUB_Y2, self.NVOX_Y2 , self.NCOM_Y2 = \
                            dpica_mask.pica_Modality_Y_creation_from_file1(self.NCOM_Y, data_path_from_file, \
                                file_name, x_size, y_size)

                        print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y2 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')  


                        print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y3 - From file =====')     
                        ini_time = datetime.now()
                        data_path_from_file = DATA_PATH_FROM_FILE
                        file_name = "clean_data_Y3.csv"     # Site Y3            
                        
                        self.clean_data_Y3, self.NSUB_Y3, self.NVOX_Y3 , self.NCOM_Y3 = \
                            dpica_mask.pica_Modality_Y_creation_from_file1(self.NCOM_Y, data_path_from_file, \
                                file_name, x_size, y_size)

                        print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y3 - Loaded time is ' , str(datetime.now() - ini_time) ,  '=====')  


            # Option 2 from ABCD folder
            b_Modality_creation_Y = B_MODALITY_CREATION_FROM_MATLAB
            # b_Modality_creation_Y = False
            if b_Modality_creation_Y :
                data_path = DATA_PATH_Y   
                snp_file_name = MODALITY_Y_RAW_FILE_NAME
                data_ID_path_input = DATA_SITES_X
                snp_ID_file_name = DATA_SITES

                # self.clean_data_Y1, self.NSUB_Y1, self.NVOX_Y1 , self.NCOM_Y1 = \
                #     dpica_mask.pica_Modality_Y_creation2(self.NCOM_Y1, data_path, \
                #         snp_file_name, self.clean_folder_file_name_data_X1, x_size, y_size)

                self.clean_data_Y1, self.NSUB_Y1, self.NVOX_Y1 , self.NCOM_Y1 = \
                    dpica_mask.pica_Modality_Y_creation3(self.NCOM_Y1, data_path, snp_file_name, \
                        data_ID_path_input, snp_ID_file_name, \
                        self.clean_folder_file_name_data_X1, x_size, y_size)



                np.savetxt( DATA_PATH_OUTPUT + "clean_data_Y1.csv", self.clean_data_Y1, delimiter=",")

        # End if B_DATASET_ABCD :

        print('[LOG][Flow_1_Setup]=====Loading Local Modality_Y - Finish =====')    

        #Print initial parameter number        
        print('[LOG][Flow_1_Setup]=====Print initial parameter number=====')    
        for run in range (1, SITE_NUM + 1):       
            print("NSUB_X" + str(run) + "  N_X" + str(run) + " = " + str(self.NSUB_ALL_X[run-1]) + \
                 "  NVOX_X" + str(run) + "  d_X" + str(run) + " = " + str(self.NVOX_ALL_X[run-1]) + \
                 "  NCOM_X" + str(run) + "  r_X" + str(run) + " = " + str(self.NCOM_ALL_X[run-1])  )

        for run in range (1, SITE_NUM + 1):       
            print("NSUB_Y" + str(run) + "  N_Y" + str(run) + " = " + str(self.NSUB_ALL_Y[run-1]) + \
                 "  NVOX_Y" + str(run) + "  d_Y" + str(run) + " = " + str(self.NVOX_ALL_Y[run-1]) + \
                 "  NCOM_Y" + str(run) + "  r_Y" + str(run) + " = " + str(self.NCOM_ALL_Y[run-1])  )



        # Print all Modality data
        print('[LOG][Flow_1_Setup]=====Print all Modality data=====')  

        for run in range (1, SITE_NUM + 1):       
            print("Modality_X" + str(run) + "  Input===X_X" + str(run) + ".shape (N_X" + str(run) + \
                " x d_X" + str(run) + ") = " + str(self.clean_data_ALL_X[run-1].shape) + \
                " NCOMP_X" + str(run) + " (r_X" + str(run) + ") = " + str(self.NCOM_ALL_X[run-1])  )

        for run in range (1, SITE_NUM + 1):       
            print("Modality_Y" + str(run) + "  Input===X_Y" + str(run) + ".shape (N_Y" + str(run) + \
                " x d_Y" + str(run) + ") = " + str(self.clean_data_ALL_Y[run-1].shape) + \
                " NCOMP_Y" + str(run) + " (r_Y" + str(run) + ") = " + str(self.NCOM_ALL_Y[run-1])  )


        print('[LOG][def_setUp]+++++Set up finish+++++')
        return (self)

    def p_to_r2(self, N):

        prb = CONSTRAINED_CONNECTION_PROABILITY
        df = N - 2
        x = abs(t.ppf(prb, df))   
        
        r = math.sqrt(1/((N-2) + x**2))*x
        return(r)

    def norm(self, x):
        """Computes the norm of a vector or the Frobenius norm of a
        matrix_rank

        """
        return mnorm(x.ravel())

    def local_reconstruction9(self, GlobalW_unmixer, GlobalA_mixer, Global_deWhite, Con_deWhite, NSUB_All, NCOM_All, SITE_NUMBER, \
        Global=False, Test=False):
        """Computes local A using Global_deWhite and Concatenated_local_deWhite
        *Input
        GlobalW_unmixer : ICA's UnMixing matrix or Weight matrix : srxr
        Global_deWhite :  Global deWhitening and output of Global PCA : srxr 
        Concatenated_local_deWhite :  Concatenate L_deWh to Con_deWh: sNxr
        NSUB_All : All subject numbers of each sites in this Modality 
        NCOM_All : All component numbers of each sites in this Modality     
        SITE_NUMBER : Number of local sites
        verbose: flag to print optimization updates
        *Output
        LocalA : Local mixing matrix : sNxr
        """
        LocalA_ALL = []        
        if SITE_NUMBER == 1 :
        # 1 site
            if not (Global) :
                if Test :
                    LocalA_1 = (dot(Global_deWhite, GlobalA_mixer))
                    LocalA_All = ((LocalA_1))
                else : 

                    LocalA_1 = np.linalg.lstsq(GlobalW_unmixer.T,Global_deWhite.T)
                    LocalA_1 = np.array(LocalA_1)[0].T
                    LocalA_All = (LocalA_1)
            else : # Global)

                # 1) Find Local_deWhite  N x r
                # 1 site
                L_dWh_1 = copy.copy(Con_deWhite[:int(NSUB_All[0]), :])
                # # 3 sites
                # L_dWh_2 = Con_deWhite[int(NSUB_All[0]):(int(NSUB_All[0])+int(NSUB_All[1])), :]
                # L_dWh_3 = Con_deWhite[(int(NSUB_All[0])+int(NSUB_All[1])):, :]

                # 2) Find [Local white A (All) sr x r] from [Glo_deWhite sr x r ] * [Glo_A r x r ] 
                LocalA_Wh_All = copy.copy(dot(Global_deWhite, GlobalA_mixer))

                # 3) Find/Seperate each Local_White_A  r x r 
                Local_Wh_A_1 = copy.copy(LocalA_Wh_All[:int(NCOM_All[0]), :])
                # # 3 sites
                # Local_Wh_A_2 = LocalA_Wh_All[int(NCOM_All[0]):(int(NCOM_All[0])+int(NCOM_All[1])), :]
                # Local_Wh_A_3 = LocalA_Wh_All[(int(NCOM_All[0])+int(NCOM_All[1])):, :]

                # 4) Find Local_A  N x r   : Nxr * rxr by [1)dewhite N x r] * [3) local_A r x r]
                # 1 site
                LocalA_1 = copy.copy(dot(L_dWh_1, Local_Wh_A_1))
                # 3 sites
                # LocalA_2 = dot(L_dWh_2, Local_Wh_A_2)
                # LocalA_3 = dot(L_dWh_3, Local_Wh_A_3)

                # 5) Concatenate LocalA_All sN x r = np.concatenate((LocalA_1),axis = 0)
                # 1 site
                LocalA_All = copy.copy(LocalA_1)
                # 3 sites        
                # LocalA_All = np.concatenate((LocalA_1, LocalA_2, LocalA_3),axis = 0)

        else :
            L_dWh_ALL = []
            Local_Wh_A_ALL = []
            NSUB_len = len(NSUB_All)
            NCOM_len = len(NCOM_All)
            NSUB_Accumate_All = list(accumulate(NSUB_All))
            NCOM_Accumate_All = list(accumulate(NCOM_All))


            # 1) Find Local_deWhite  N x r of each sites
            #       INPUT  :  Con_deWhite
            #       OUTPUT :  List of local_deWhtie into L_dWh_ALL
            for run in range (NSUB_len):    
                if run == 0 :   
                    NSUB_start = 0
                    NSUB_end = NSUB_All[run]
                    L_dWh_ALL.append( Con_deWhite[NSUB_start:NSUB_end , :] )
                else :
                    NSUB_start =  int(NSUB_Accumate_All[run-1])
                    NSUB_end = int(NSUB_Accumate_All[run])                    
                    L_dWh_ALL.append(Con_deWhite[NSUB_start:NSUB_end , :])

            # 2) Find [Local white A (All) sr x r] from [Glo_deWhite sr x r ] * [Glo_A r x r ] 
            #       INPUT  :  Global_deWhite, GlobalA_mixer
            #       OUTPUT :  LocalA_Wh_All
            LocalA_Wh_All = dot(Global_deWhite, GlobalA_mixer)

            # 3) Find/Seperate each Local_White_A  r x r 
            #       INPUT  :  LocalA_Wh_All
            #       OUTPUT :  List of Local_White_A  into Local_Wh_A_ALL            
            for run in range (NCOM_len):    
                if run == 0 :   
                    NCOM_start = 0
                    NCOM_end = NCOM_All[run]
                    Local_Wh_A_ALL.append( LocalA_Wh_All[NCOM_start:NCOM_end , :] )
                else :
                    NCOM_start = int(NCOM_Accumate_All[run-1])
                    NCOM_end = int(NCOM_Accumate_All[run])
                    Local_Wh_A_ALL.append(LocalA_Wh_All[NCOM_start:NCOM_end , :])

            # 4) Find Local_A  N x r   : Nxr * rxr by [1)dewhite N x r] * [3) local_A r x r]
            #       to concatenate them into LocalA_All sN x r 
            #       INPUT  :  L_dWh_ALL and Local_Wh_A_ALL
            #       OUTPUT :  List of Local A that is concatenated into LocalA_ALL     
            for run in range (NSUB_len):    
                if run == 0 :
                    LocalA_All = dot(L_dWh_ALL[run], Local_Wh_A_ALL[run])
                else:
                    LocalA_All = np.concatenate((LocalA_All, dot(L_dWh_ALL[run], Local_Wh_A_ALL[run])),axis = 0)


        # print("Local_Reconstruction...Return====LocalA===")     
        # print("Local_Reconstruction...Finish")     

        return (LocalA_All)

    def global_reconstruction9(self, LocalA_All, Global_White, Con_White, NSUB_All, NCOM_All, SITE_NUMBER, Global=False):
        """Computes Global A using Global_White and Concatenated_local_White
        *Input
        LocalA_All : LocalA_All : Local A (mixing) matrix of this Modality : sNxr
        Global_White :  Global Whitening as output of Global PCA : rxsr 
        Concatenated_local_White :  Concatenate L_Wh to be Con_White: rxsN
        NSUB_All : All subject numbers of each sites in this Modality : Array of interger numbers
        NCOM_All : All component numbers of each sites in this Modality : Array of interger numbers
        SITE_NUMBER : Number of local sites        
        verbose: flag to print optimization updates
        *Output
        GlobalA_mixer : Global A mixing matrix or Loading parameter matrix : srxr
        """

        # print("Global_Reconstruction...Start")      
        # Initialization
        # print("Global_Reconstruction...Initialization")    
        # print("Global_Reconstruction...NSUB_All =", NSUB_All) 

        if SITE_NUMBER == 1 :
        # 1 site
            if not (Global) :        
                GlobalA_mixer = dot(Global_White, LocalA_All)
            else :
                L_Wh_1 = Con_White[:,:int(NSUB_All[0])]
                LocalA_1 = LocalA_All[:int(NSUB_All[0]) ,: ]      

                Local_Wh_A_1 = dot(L_Wh_1, LocalA_1)
                Local_Wh_A_All = Local_Wh_A_1

                GlobalA_mixer = dot(Global_White, Local_Wh_A_All)

        else :
            L_Wh_ALL = []
            LocalA_ALL = []
            NSUB_len = len(NSUB_All)
            NSUB_Accumate_All = list(accumulate(NSUB_All))

            # 1) Find Local_White of each sites
            #       INPUT  :  Con_White
            #       OUTPUT :  List of local_Whtie into L_Wh_ALL
            for run in range (NSUB_len):    
                if run == 0 :   
                    NSUB_start = 0
                    NSUB_end = NSUB_All[run]
                    L_Wh_ALL.append( Con_White[:, NSUB_start:NSUB_end] )
                else :
                    NSUB_start =  int(NSUB_Accumate_All[run-1])
                    NSUB_end = int(NSUB_Accumate_All[run])                    
                    L_Wh_ALL.append(Con_White[:,NSUB_start:NSUB_end])

            # 2) Find/Seperate each Local_White_A  r x r 
            #       INPUT  :  LocalA_All
            #       OUTPUT :  List of Local A   into LocalA_ALL            
            for run in range (NSUB_len):    
                if run == 0 :   
                    NSUB_start = 0
                    NSUB_end = NSUB_All[run]
                    LocalA_ALL.append( LocalA_All[NSUB_start:NSUB_end , :] )
                else :
                    NSUB_start =  int(NSUB_Accumate_All[run-1])
                    NSUB_end = int(NSUB_Accumate_All[run])                    
                    LocalA_ALL.append(LocalA_All[NSUB_start:NSUB_end , :])

            # 3) Find [Local white A (All) 
            #       INPUT  :  L_dWh_ALL and LocalA_ALL
            #       OUTPUT :  List of Local A that is concatenated into Local_Wh_A_All     
            for run in range (NSUB_len):    
                if run == 0 :
                    Local_Wh_A_All = dot(L_Wh_ALL[run], LocalA_ALL[run])  
                    # ValueError: shapes (125,777) and (258,25) not aligned: 777 (dim 1) != 258 (dim 0)
                else:
                    Local_Wh_A_All = np.concatenate((Local_Wh_A_All, dot(L_Wh_ALL[run], LocalA_ALL[run])),axis = 0)

            # 4) Find Gloabl A (All) 
            #       INPUT  :  Global_White, Local_Wh_A_All
            #       OUTPUT :  GlobalA_mixer
            GlobalA_mixer = dot(Global_White, Local_Wh_A_All)

      
        return (GlobalA_mixer)
 
    def test_dpICA_infomax_clean(self):
 
        start = time.time()
        LocalA_Corr_A_ALL_X = []
        LocalA_Corr_A_ALL_Y = []

        #
        # Local PCA start
        #
        print('[LOG][Flow_2_Local_PCA]=====Start=====')



        # print('[LOG][Flow_2_Local_PCA]=====Local PCA of Modality_X=====')



        for run in range (SITE_NUM):       
            U_X1, L_white_X1, L_dewhite_X1 = pca_whiten8(self.clean_data_ALL_X[run], self.NCOM_ALL_X[run], B_LOCAL_PCA_MEAN_X, B_LOCAL_PCA_WHITE_X) #Remove mean and Don't whitening

            self.U_ALL_X.append(U_X1)
            self.L_white_ALL_X.append(L_white_X1)
            self.L_dewhite_ALL_X.append(L_dewhite_X1)                            

            if not os.path.exists(DATA_PATH_OUTPUT + "/local" + str(run) ):
                os.makedirs(DATA_PATH_OUTPUT + "/local" + str(run) )                                                 
            np.savetxt( DATA_PATH_OUTPUT + "/local" + str(run) + "/U_X" + str(run) + ".csv", U_X1, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "/local" + str(run) + "/L_white_X" + str(run) + ".csv", L_white_X1, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "/local" + str(run) + "/L_dewhite_X" + str(run) + ".csv", L_dewhite_X1, delimiter=",")

        # print('[LOG][Flow_2_Local_PCA]=====Local PCA of Modality_Y=====')


        for run in range (SITE_NUM):       
            U_Y1, L_white_Y1, L_dewhite_Y1 = pca_whiten8(self.clean_data_ALL_Y[run], self.NCOM_ALL_Y[run], B_LOCAL_PCA_MEAN_Y, B_LOCAL_PCA_WHITE_Y) #Remove mean and Don't whitening

            self.U_ALL_Y.append(U_Y1)
            self.L_white_ALL_Y.append(L_white_Y1)
            self.L_dewhite_ALL_Y.append(L_dewhite_Y1)                            

            if not os.path.exists(DATA_PATH_OUTPUT + "/local" + str(run) ):
                os.makedirs(DATA_PATH_OUTPUT + "/local" + str(run) )   

            np.savetxt( DATA_PATH_OUTPUT + "/local" + str(run) + "/U_Y" + str(run) + ".csv", U_Y1, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "/local" + str(run) + "/L_white_Y" + str(run) + ".csv", L_white_Y1, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "/local" + str(run) + "/L_dewhite_Y" + str(run) + ".csv", L_dewhite_Y1, delimiter=",")



        # Print all Modality Local PCA shape
        print('[LOG][Flow_2_Local_PCA]=====Print all Modality Local PCA shape=====')  

        for run in range (SITE_NUM):     
            print("Modality_X" + str(run+1) + " === U_X" + str(run+1) + ".shape (r_X" + str(run+1) + \
                " x d_X" + str(run+1) + ")" + str(self.U_ALL_X[run].shape) +  \
                " L_white_X" + str(run+1) + " (r_X" + str(run+1) + " x N_X" + str(run+1) + ") = " + str(self.L_white_ALL_X[run].shape) + \
                " L_dewhite_X" + str(run+1) + " (N_X" + str(run+1) + " x r_X" + str(run+1) + ") = " + \
                str(self.L_dewhite_ALL_X[run].shape) )  

  
        for run in range (SITE_NUM):     
            print("Modality_Y" + str(run+1) + " === U_Y" + str(run+1) + ".shape (r_Y" + str(run+1) + \
                " x d_Y" + str(run+1) + ")" + str(self.U_ALL_Y[run].shape) +  \
                " L_white_Y" + str(run+1) + " (r_Y" + str(run+1) + " x N_Y" + str(run+1) + ") = " + str(self.L_white_ALL_Y[run].shape) + \
                " L_dewhite_Y" + str(run+1) + " (N_Y" + str(run+1) + " x r_Y" + str(run+1) + ") = " + \
                str(self.L_dewhite_ALL_Y[run].shape) )  


        print('[LOG][Flow_2_Local_PCA]=====End=====')          


        #
        # Global Global_U, White, and deWhite start
        #
        print('[LOG][Flow_3_Global_U, White, and deWhite]=====Start=====')    

        # # 1 site
        Global_U_X = self.U_ALL_X[0]
        Global_U_Y = self.U_ALL_Y[0] 

        # # n sites
        for run in range (1, SITE_NUM):       
            Global_U_X = np.concatenate((Global_U_X, self.U_ALL_X[run]),axis = 0)
            Global_U_Y = np.concatenate((Global_U_Y, self.U_ALL_Y[run]),axis = 0)

        # Print shape of concatenated local U = Global U : sr x d 
        # print('[LOG][Flow_3_Global_U, White, and deWhite]=====Print shape of concatenated local U = Global U : sr x d =====')   
        print('Modality_X === Global_U_X.shape (sr x d) ',Global_U_X.shape )
        print('Modality_Y === Global_U_Y.shape (sr x d) ',Global_U_Y.shape )



        # 1 site
        self.Con_White_X = self.L_white_ALL_X[0]
        self.Con_White_Y = self.L_white_ALL_Y[0]

        
        # n sites
        for run in range (1, SITE_NUM):       
            self.Con_White_X = np.concatenate((self.Con_White_X, self.L_white_ALL_X[run]),axis = 1)
            self.Con_White_Y = np.concatenate((self.Con_White_Y, self.L_white_ALL_Y[run]),axis = 1)
        # Print shape of concatenated local white  = Con_White : r x sN 
        # print('[LOG][Flow_3_Global_U, White, and deWhite]=====Print shape of concatenated local white = Con_White : r x sN =====')   
        print('Modality_X === Con_White_X.shape (r x sN) ',self.Con_White_X.shape )
        print('Modality_Y === Con_White_Y.shape (r x sN) ',self.Con_White_Y.shape )



        # 1 site
        self.Con_deWhite_X = self.L_dewhite_ALL_X[0]
        self.Con_deWhite_Y = self.L_dewhite_ALL_Y[0]

        # n sites
        for run in range (1, SITE_NUM):         
            self.Con_deWhite_X = np.concatenate((self.Con_deWhite_X, self.L_dewhite_ALL_X[run]),axis = 0)
            self.Con_deWhite_Y = np.concatenate((self.Con_deWhite_Y, self.L_dewhite_ALL_Y[run]),axis = 0)
        # Print shape of concatenated local dewhite  = Con_deWhite : sN x r
        # print('[LOG][Flow_3_Global_U, White, and deWhite]=====Print shape of concatenated local dewhite = Con_deWhite : sN x r =====')   
        print('Modality_X === Con_deWhite_X.shape (sN x r) ',self.Con_deWhite_X.shape )
        print('Modality_Y === Con_deWhite_Y.shape (sN x r) ',self.Con_deWhite_Y.shape )

        print("NSUB_ALL_X =", self.NSUB_ALL_X) 
        print("NSUB_ALL_Y =", self.NSUB_ALL_Y) 
        print("NVOX_ALL_X =", self.NVOX_ALL_X) 
        print("NVOX_ALL_Y =", self.NVOX_ALL_Y) 
        print("NCOM_ALL_X =", self.NCOM_ALL_X) 
        print("NCOM_ALL_Y =", self.NCOM_ALL_Y) 

        if not os.path.exists(DATA_PATH_OUTPUT + "/remote/" ):
            os.makedirs(DATA_PATH_OUTPUT + "/remote/" )    
        np.savetxt( DATA_PATH_OUTPUT + "/remote/" + "Global_U_X.csv", Global_U_X, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "/remote/" + "Con_White_X.csv", self.Con_White_X, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "/remote/" + "Con_deWhite_X.csv", self.Con_deWhite_X, delimiter=",")

        np.savetxt( DATA_PATH_OUTPUT + "/remote/" + "Global_U_Y.csv", Global_U_Y, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "/remote/" + "Con_White_Y.csv", self.Con_White_Y, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "/remote/" + "Con_deWhite_Y.csv", self.Con_deWhite_Y, delimiter=",")


        print('[LOG][Flow_3_Global_U, White, and deWhite]=====Stop=====')      

        #
        # Global PCA start
        #
        print('[LOG][Flow_4_Global_PCA]=====Start=====')   

        # Global PCA of reach Modality         
        print('[LOG][Flow_4_Global_PCA]=====Global PCA of Modality_X and Modality_Y=====')
        

        if B_GLOBAL :
            self.GlobalPCA_U_X, self.GlobalPCA_White_X, self.GlobalPCA_dewhite_X  = pca_whiten8(Global_U_X, self.Global_NCOM_X, B_GLOBAL_PCA_MEAN_X, B_GLOBAL_PCA_WHITE_X) #Keep mean and DO whitening

            self.GlobalPCA_U_Y, self.GlobalPCA_White_Y, self.GlobalPCA_dewhite_Y  = pca_whiten8(Global_U_Y, self.Global_NCOM_Y, B_GLOBAL_PCA_MEAN_Y, B_GLOBAL_PCA_WHITE_Y) #Keep mean and DO whitening
        else :
            self.GlobalPCA_U_X = Global_U_X
            self.GlobalPCA_White_X = self.Con_White_X
            self.GlobalPCA_dewhite_X = self.Con_deWhite_X

            self.GlobalPCA_U_Y = Global_U_Y
            self.GlobalPCA_White_Y = self.Con_White_Y
            self.GlobalPCA_dewhite_Y = self.Con_deWhite_Y
            print('There is no GLOBAL. ')
            print('' + \
                'GlobalPCA_White (r x sr) = Local_White and ' + \
                'GlobalPCA_dewhite (sr x r ) = Local_deWhite' )


        print('  ')   

        # Print all Modality Global PCA shape
        # print('[LOG][Flow_4_Global_PCA]=====Print all Modality Global PCA shape=====')
        print('Modality_X === GlobalPCA_U_X.shape (r_X x d_X)', self.GlobalPCA_U_X.shape , \
            'GlobalPCA_White_X (r_X x sr_X) = ', self.GlobalPCA_White_X.shape  , \
            'GlobalPCA_dewhite_X (sr_X x r_X ) = ', self.GlobalPCA_dewhite_X.shape  )  
        print('Modality_Y === GlobalPCA_U_Y.shape (r_Y x d_Y)',self.GlobalPCA_U_Y.shape , \
            'GlobalPCA_White_Y (r_Y x sr_Y) = ', self.GlobalPCA_White_Y.shape  , \
            'GlobalPCA_dewhite_Y (sr_Y x r_Y ) = ', self.GlobalPCA_dewhite_Y.shape  )  



        np.savetxt( DATA_PATH_OUTPUT + "/remote/"  + "GlobalPCA_U_X.csv", self.GlobalPCA_U_X, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "/remote/"  + "GlobalPCA_White_X.csv", self.GlobalPCA_White_X, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "/remote/"  + "GlobalPCA_dewhite_X.csv", self.GlobalPCA_dewhite_X, delimiter=",")

        np.savetxt( DATA_PATH_OUTPUT + "/remote/"  + "GlobalPCA_U_Y.csv", self.GlobalPCA_U_Y, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "/remote/"  + "GlobalPCA_White_Y.csv", self.GlobalPCA_White_Y, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "/remote/"  + "GlobalPCA_dewhite_Y.csv", self.GlobalPCA_dewhite_Y, delimiter=",")


        print('[LOG][Flow_4_Global_PCA]=====End=====') 


      
        ###########################################################################
        # ICA Infomax flow
        #   - Flow 5
        #   - Flow 6
        #   - Flow 7
        #   - Flow 8
        #   - Flow 8a
        #
        
        # ICA Infomax function   

        if ICA_RUN_NUMBER == 1 : 

            self.run = 1            
            self, GlobalW_unmixer_X, sphere_X, \
                  GlobalW_unmixer_Y, sphere_Y = pica_infomax8(self) 

            weight_X = copy.copy(np.dot(GlobalW_unmixer_X, sphere_X))    
            weight_Y = copy.copy(np.dot(GlobalW_unmixer_Y, sphere_Y))

            S_sources_X =    copy.copy(np.dot(weight_X, self.GlobalPCA_U_X))
            S_sources_Y =    copy.copy(np.dot(weight_Y, self.GlobalPCA_U_Y))

            GlobalA_mixer_X = copy.copy(pinv(weight_X))
            GlobalA_mixer_Y = copy.copy(pinv(weight_Y))

            np.savetxt( DATA_PATH_OUTPUT + "ICA_GlobalA_mixer_X.csv", GlobalA_mixer_X, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "ICA_S_sources_X.csv", S_sources_X, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "ICA_GlobalW_unmixer_X.csv", GlobalW_unmixer_X, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "ICA_Global_sphere_X.csv", sphere_X, delimiter=",")

            np.savetxt( DATA_PATH_OUTPUT + "ICA_GlobalA_mixer_Y.csv", GlobalA_mixer_Y, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "ICA_S_sources_Y.csv", S_sources_Y, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "ICA_GlobalW_unmixer_Y.csv", GlobalW_unmixer_Y, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "ICA_Global_sphere_Y.csv", sphere_Y, delimiter=",")


            LocalA_Corr_All_X = (self.local_reconstruction9(weight_X, GlobalA_mixer_X, \
                self.GlobalPCA_dewhite_X, self.Con_deWhite_X, self.NSUB_ALL_X, self.NCOM_ALL_X, SITE_NUM, B_GLOBAL, False))      

            LocalA_Corr_All_Y = (self.local_reconstruction9(weight_Y, GlobalA_mixer_Y, \
                self.GlobalPCA_dewhite_Y, self.Con_deWhite_Y, self.NSUB_ALL_Y, self.NCOM_ALL_Y, SITE_NUM, B_GLOBAL, False))          

            NSUB_len = len(self.NSUB_ALL_X) 
            NSUB_Accumate_All = list(accumulate(self.NSUB_ALL_X))

            for run in range (NSUB_len):    
                if run == 0 :   
                    NSUB_start = 0
                    NSUB_end = NSUB_Accumate_All[run]
                    LocalA_Corr_A_ALL_X.append( LocalA_Corr_All_X[NSUB_start:NSUB_end , :] )
                    LocalA_Corr_A_ALL_Y.append( LocalA_Corr_All_Y[NSUB_start:NSUB_end , :] )
                else :
                    NSUB_start =  int(NSUB_Accumate_All[run-1])
                    NSUB_end = int(NSUB_Accumate_All[run])                    
                    LocalA_Corr_A_ALL_X.append(LocalA_Corr_All_X[NSUB_start:NSUB_end , :])
                    LocalA_Corr_A_ALL_Y.append(LocalA_Corr_All_Y[NSUB_start:NSUB_end , :])


        else :
            # b_infomax_creation = True
            # if b_infomax_creation :
              
            weight_X_All = []
            S_sources_X_All = []
            sphere_X_All = []
            

            weight_Y_All = []
            S_sources_Y_All = []
            sphere_Y_All = []


            for run in range (ICA_RUN_NUMBER):
                self.run = run
                self, GlobalW_unmixer_X_1 , sphere_X_1, \
                    GlobalW_unmixer_Y_1, sphere_Y_1,= (pica_infomax8(self) )         

                weight_X_1 = copy.copy(np.dot(GlobalW_unmixer_X_1, sphere_X_1) )
                weight_Y_1 = copy.copy(np.dot(GlobalW_unmixer_Y_1, sphere_Y_1) )
                S_sources_X_1 = copy.copy(np.dot(weight_X_1, self.GlobalPCA_U_X) )
                S_sources_Y_1 = copy.copy(np.dot(weight_Y_1, self.GlobalPCA_U_Y) )

                print('run =', run)
                print('weight_X_1.shape =', np.array(weight_X_1).shape)
                print('S_sources_X_1.shape =', np.array(S_sources_X_1).shape)
                print('weight_Y_1.shape =', np.array(weight_Y_1).shape)
                print('S_sources_Y_1.shape =', np.array(S_sources_Y_1).shape)
                print('  ')

                weight_X_All.append(weight_X_1)
                S_sources_X_All.append(S_sources_X_1)
                sphere_X_All.append(sphere_X_1)
                weight_Y_All.append(weight_Y_1)
                S_sources_Y_All.append(S_sources_Y_1)
                sphere_Y_All.append(sphere_Y_1)

                np.savetxt( DATA_PATH_OUTPUT + "ICA_weight_X_" + str(run) + ".csv", weight_X_1, delimiter=",")
                np.savetxt( DATA_PATH_OUTPUT + "ICA_S_sources_X_" + str(run) + ".csv", S_sources_X_1, delimiter=",")
                np.savetxt( DATA_PATH_OUTPUT + "ICA_sphere_X_" + str(run) + ".csv", sphere_X_1, delimiter=",")
                np.savetxt( DATA_PATH_OUTPUT + "ICA_weight_Y_" + str(run) + ".csv", weight_Y_1, delimiter=",")
                np.savetxt( DATA_PATH_OUTPUT + "ICA_S_sources_Y_" + str(run) + ".csv", S_sources_Y_1, delimiter=",")
                np.savetxt( DATA_PATH_OUTPUT + "ICA_sphere_Y_" + str(run) + ".csv", sphere_Y_1, delimiter=",")


                print('weight_X_All.shape =', np.array(weight_X_All).shape)
                print('S_sources_X_All.shape =', np.array(S_sources_X_All).shape)
                print('weight_X_All[run,:,:].shape =', (np.array(weight_X_All)[run,:,:]).shape)
                print('S_sources_X_All[run,:,:].shape =', (np.array(S_sources_X_All)[run,:,:]).shape)
                print('weight_Y_All[run,:,:].shape =', (np.array(weight_Y_All)[run,:,:]).shape)
                print('S_sources_Y_All[run,:,:].shape =', (np.array(S_sources_Y_All)[run,:,:]).shape)

                # end of for

                print('  ')

            if ICA_RUN_COMPONENT_VALIDATION :

                print('[LOG][Flow_8b_Parallel_ICA-Global] Multi-run component validation of  ', str(self.run), " run. Start.")               
                self, Coef_max_pair_output_X, Coef_max_pair_output_Y, Coef_max_pair_output_XY\
                    = pica_multi_run_component_validation2(self, ICA_RUN_NUMBER,   \
                            weight_X_All, weight_Y_All, S_sources_X_All, S_sources_Y_All )
                           
                print('[LOG][Flow_8b_Parallel_ICA-Global] Multi-run component validation of  ', str(self.run), " run. Finish.")               


            if ICA_RUN_AVERAGE :
                print('[LOG][Flow_8b_Parallel_ICA-Global] Multi-run ICA AVERAGE of ', str(ICA_RUN_NUMBER), " X run. Start.") 
                self, GlobalA_mixer_X, S_sources_X, GlobalW_unmixer_X \
                    = pica_infomax_run_average5(self, "X", ICA_RUN_NUMBER, self.GlobalPCA_U_X , \
                            weight_X_All, S_sources_X_All )
                print('[LOG][Flow_8b_Parallel_ICA-Global] Multi-run ICA AVERAGE of ', str(ICA_RUN_NUMBER),  " X run. Finish.") 


                print('[LOG][Flow_8b_Parallel_ICA-Global] Multi-run ICA AVERAGE of ', str(ICA_RUN_NUMBER),  " Y run. Start.") 
                self, GlobalA_mixer_Y, S_sources_Y, GlobalW_unmixer_Y \
                    = pica_infomax_run_average5(self, "Y", ICA_RUN_NUMBER, self.GlobalPCA_U_Y , \
                            weight_Y_All, S_sources_Y_All )

                print('[LOG][Flow_8b_Parallel_ICA-Global] Multi-run ICA AVERAGE of ', str(ICA_RUN_NUMBER), " Y run. Finish.") 

            if ICA_RUN_ICASSO :

                print('[LOG][Flow_8b_Parallel_ICA-Global] Multi-run ICA ICASSO of ', str(self.run), " X run. Start.")               
                self, GlobalA_mixer_X, S_sources_X, GlobalW_unmixer_X \
                    = pica_infomax_run_icasso6(self, "X", ICA_RUN_NUMBER, self.GlobalPCA_U_X , \
                            weight_X_All, S_sources_X_All )
                print('[LOG][Flow_8b_Parallel_ICA-Global] Multi-run ICA ICASSO of ', str(self.run), " X run. Finish.")  


                print('[LOG][Flow_8b_Parallel_ICA-Global] Multi-run ICA ICASSO of ', str(self.run), " Y run. Start.")   
                self, GlobalA_mixer_Y, S_sources_Y, GlobalW_unmixer_Y \
                    = pica_infomax_run_icasso6(self, "Y", ICA_RUN_NUMBER, self.GlobalPCA_U_Y , \
                            weight_Y_All, S_sources_Y_All )
                print('[LOG][Flow_8b_Parallel_ICA-Global] Multi-run ICA ICASSO of ', str(self.run), " Y run. Finish.") 


            np.savetxt( DATA_PATH_OUTPUT + "ICA_GlobalA_mixer_X.csv", GlobalA_mixer_X, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "ICA_S_sources_X.csv", S_sources_X, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "ICA_GlobalW_unmixer_X.csv", GlobalW_unmixer_X, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "ICA_Global_sphere_X.csv", sphere_X_1, delimiter=",")

            np.savetxt( DATA_PATH_OUTPUT + "ICA_GlobalA_mixer_Y.csv", GlobalA_mixer_Y, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "ICA_S_sources_Y.csv", S_sources_Y, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "ICA_GlobalW_unmixer_Y.csv", GlobalW_unmixer_Y, delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "ICA_Global_sphere_Y.csv", sphere_Y_1, delimiter=",")


            LocalA_Corr_All_X = (self.local_reconstruction9(GlobalW_unmixer_X, GlobalA_mixer_X, \
                self.GlobalPCA_dewhite_X, self.Con_deWhite_X, self.NSUB_ALL_X, self.NCOM_ALL_X, SITE_NUM, B_GLOBAL, False))      

            LocalA_Corr_All_Y = (self.local_reconstruction9(GlobalW_unmixer_Y, GlobalA_mixer_Y, \
                self.GlobalPCA_dewhite_Y, self.Con_deWhite_Y, self.NSUB_ALL_Y, self.NCOM_ALL_Y, SITE_NUM, B_GLOBAL, False))          

            
            NSUB_len = len(self.NSUB_ALL_X) 
            NSUB_Accumate_All = list(accumulate(self.NSUB_ALL_X))

            for run in range (NSUB_len):    
                if run == 0 :   
                    NSUB_start = 0
                    NSUB_end = NSUB_Accumate_All[run]
                    LocalA_Corr_A_ALL_X.append( LocalA_Corr_All_X[NSUB_start:NSUB_end , :] )
                    LocalA_Corr_A_ALL_Y.append( LocalA_Corr_All_Y[NSUB_start:NSUB_end , :] )
                else :
                    NSUB_start =  int(NSUB_Accumate_All[run-1])
                    NSUB_end = int(NSUB_Accumate_All[run])                    
                    LocalA_Corr_A_ALL_X.append(LocalA_Corr_All_X[NSUB_start:NSUB_end , :])
                    LocalA_Corr_A_ALL_Y.append(LocalA_Corr_All_Y[NSUB_start:NSUB_end , :])

        ##########################################################################

        print('[LOG][Flow_9_Looping the next iteration of Global ICA (Finding W) via Infomax and pICA (Finding ∆A)]. Finish.')   

        ###########################################################################
        # Report Analysis    
        print('[LOG][Flow_10_Report_correlation_data-Local_reconstruction]=====Modality_X : Find local A using GlobalPCA_dewhite_X and Con_deWhite_X   =====')
        print('[LOG][Flow_10_Report_correlation_data-Local_reconstruction]=====Modality_Y : Find local A using GlobalPCA_dewhite_Y and Con_deWhite_Y   =====')      


        print('[LOG][Flow_10_Report]=====Print all Modality shape=====')
        print('Modality_X === GlobalA_mixer_X.shape (r_X x r_X) = ',GlobalA_mixer_X.shape , \
            'S_sources_X (r_X x d_X) = ', S_sources_X.shape  , \
            'GlobalW_unmixer_X (r_X x r_X ) = ', GlobalW_unmixer_X.shape  )  
        print('Modality_Y === GlobalA_mixer_Y.shape (r_Y x r_Y) = ',GlobalA_mixer_Y.shape , \
            'S_sources_Y (r_Y x d_Y) = ', S_sources_Y.shape  , \
            'GlobalW_unmixer_Y (r_Y x r_Y ) = ', GlobalW_unmixer_Y.shape  )  

        # Report correlation data - Local reconstruction

        print('[LOG][Flow_10_Report_correlation_data-Local_reconstruction]=====Start=====')   

        # Report_correlation_data- Find local A using GlobalPCA_dewhite_X and Con_deWhite_X    
        # Called Local-reconstruction ; from-Global-to-Local     

        print('[LOG][Flow_10_Report_correlation_data-Local_reconstruction]=====Print all Modality local A shape=====')
        for run in range (SITE_NUM):  
            print("Modality_X === LocalA_Corr_A" + str(run+1) + "_X (sN_X x r_X) " + str(LocalA_Corr_A_ALL_X[run].shape) )
            print("Modality_Y === LocalA_Corr_A" + str(run+1) + "_Y (sN_Y x r_Y) " + str(LocalA_Corr_A_ALL_Y[run].shape) )
            

        print('[LOG][Flow_10_Report_correlation_data-Local_reconstruction]=====Finish=====')  


        print('[LOG][Flow_10_Report_correlation_between_Python_and_MathLab]=====LocalA_Start=====')     


        # Save S_X, S_Y, local_A_X, and local_A_Y
        np.savetxt( DATA_PATH_OUTPUT + "GlobalA_mixer_X.csv", GlobalA_mixer_X, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "S_sources_X.csv", S_sources_X, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "GlobalW_unmixer_X.csv", GlobalW_unmixer_X, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "GlobalA_mixer_Y.csv", GlobalA_mixer_Y, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "S_sources_Y.csv", S_sources_Y, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "GlobalW_unmixer_Y.csv", GlobalW_unmixer_Y, delimiter=",")

        for run in range (SITE_NUM):  
            np.savetxt( DATA_PATH_OUTPUT + "LocalA_Corr_A" + str(run) + "_X.csv", LocalA_Corr_A_ALL_X[run], delimiter=",")
            np.savetxt( DATA_PATH_OUTPUT + "LocalA_Corr_A" + str(run) + "_Y.csv", LocalA_Corr_A_ALL_Y[run], delimiter=",")


        np.savetxt( DATA_PATH_OUTPUT  + "mymaxcorr_list.csv", self.mymaxcorr_list, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT  + "mymax_pair_XY.csv", self.max_pair_output_XY, fmt='%s')               
        np.savetxt( DATA_PATH_OUTPUT  + "mymax_pair_X.csv", self.max_pair_output_X, fmt='%s')  
        np.savetxt( DATA_PATH_OUTPUT  + "mymax_pair_Y.csv", self.max_pair_output_Y, fmt='%s')           
        np.savetxt( DATA_PATH_OUTPUT  + "myentropy_list_X.csv", self.myentropy_list_X, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT  + "myentropy_list_Y.csv", self.myentropy_list_Y, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT  + "mySTEP_list_X.csv", self.mySTEP_list_X, delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT  + "mySTEP_list_Y.csv", self.mySTEP_list_Y, delimiter=",")

        print('======================================')
        print('======================================')
        print('DATA_PATH_OUTPUT = ', DATA_PATH_OUTPUT)
        print('maxcorr = ', self.mymaxcorr_list)
        print('max_pair_XY = ', self.max_pair_output_XY)     
        print('max_pair_X = ', self.max_pair_output_X)     
        print('max_pair_Y = ', self.max_pair_output_Y)          
        print('ENTROPY_X =  ' , self.myentropy_list_X)
        print('ENTROPY_Y =  ' , self.myentropy_list_Y)
        print('STEP_X =  ' , self.mySTEP_list_X)
        print('STEP_Y =  ' , self.mySTEP_list_Y)
        print('======================================')
        print('======================================')

        return (self)

def diagsqrts2(w):

    """
    Returns direct and inverse square root normalization matrices
    """
    Di = np.diag(1. / (np.sqrt(w) + np.finfo(float).eps))
    Di = Di.real.round(4)
    D = np.diag(np.sqrt(w))
    D = D.real.round(4)

    return D, Di


def pca_whiten8(x2d_input, n_comp, b_mean, b_whitening):    
    """ data Whitening  ==> pca_whiten6 is pca_whiten3 without whitening.
    *Input
    x2d : 2d data matrix of observations by variables
    n_comp: Number of components to retain
    b_mean : To decide to remove mean if False
    b_whitening : to decide to whitening if True
    *Output
    Xwhite : Whitened X
    white : whitening matrix (Xwhite = np.dot(white,X))
    dewhite : dewhitening matrix (X = np.dot(dewhite,Xwhite))
    """
    # PCA...Start    
    # print("PCA...Start")      
    if b_mean :     # Keep mean
        x2d = x2d_input
    else :          # Remove mean
        x2d = x2d_input - x2d_input.mean(axis=1).reshape((-1, 1))   
        
    NSUB, NVOX = x2d.shape
    if NSUB > NVOX:
        cov = np.dot(x2d.T, x2d) / (NSUB - 1)    
        w, v = eigh(cov, eigvals=(NVOX - n_comp, NVOX - 1))    
        D, Di = diagsqrts2(w)
        u = np.dot(dot(x2d, v), Di)
        x_white = v.T
        white = np.dot(Di, u.T)
        dewhite = np.dot(u, D)
    else:
        # print("PCA...cov++')            
        cov = np.dot(x2d, x2d.T) / (NVOX - 1)    
        
        # cov = np.cov(x2d , rowvar=False)      
        # print("PCA...Eigenvector++')            
        w, u = eigh(cov, eigvals = (NSUB - n_comp, NSUB - 1))

        # w, u = np.linalg.eig(cov)

        # print("PCA...w eigenvalue rx1',w.shape )
        # print("PCA...u eigenvector Nxr ',u.shape)           # This Global u must be identity matrix
        if b_whitening :
            # print("PCA...diagsqrts(w)=D rxr',D.shape )
            D, Di = diagsqrts2(w) # Return direct and inverse square root normalization matrices
            # print("PCA...white=Whitening X')
            white = np.dot(Di, u.T)
            x_white = np.dot(white, x2d)
            # print("PCA...white=x_white (rxd)',x_white.shape )         
            dewhite = np.dot(u, D)
        else :       
            white = u.T
            x_white = np.dot(white, x2d) # c x v        # newVectors = data * whiteningMatrix'; newVectors = newVectors';  
            dewhite = u
        # print("PCA...return====x_white, white, dewhite ===')   
        # print("PCA...Finish')      
    return (x_white, white, dewhite)

def weight_update4(weights, x_white, bias1, lrate1, b_exp):
    """ Update rule for infomax
    This function recieves parameters to update W1
    * Input
    weights : unmixing matrix (must be a square matrix)
    x_white: whitened data
    bias1: current estimated bias
    lrate1: current learning rate
    b_exp : experiment 

    * Output
    weights : updated mixing matrix
    bias: updated bias
    lrate1: updated learning rate
    """
    NCOMP, NVOX = (x_white.shape)
    block1 = (int(np.floor(np.sqrt(NVOX / 3))))       # Matlab [138,11]   Python  138  
    last1 = (int(np.fix((NVOX/block1-1)*block1+1)))   # Matlab [57568,357]  Python 57568.0  357.0  


    # Experiment
    if not b_exp :
        permute1 = permutation(NVOX) 
    else :
        permute1 = range(NVOX)
    for start in range(0, last1, block1):
        if start + block1 < NVOX:
            tt2 = (start + block1 )

        else:
            tt2 = (NVOX)
            block1 = (NVOX - start)
        
        unmixed = (np.dot(weights, x_white[:, permute1[start:tt2]]) + bias1)
        logit = 1 / (1 + np.exp(-unmixed))
        weights = (weights + lrate1 * np.dot( 
            block1 * np.eye(NCOMP) + np.dot( (1-2*logit), unmixed.T), weights))

        bias1 = (bias1 + lrate1 * (1-2*logit).sum(axis=1).reshape(bias1.shape))
        # Checking if W blows up

        if (np.isnan(weights)).any() or np.max(np.abs(weights)) > MAX_WEIGHT:
            print("Weight is outside the range. Restarting.")
            weights = (np.eye(NCOMP))
            bias1 = (np.zeros((NCOMP, 1)))
            error = 1
           
            if lrate1 > 1e-6 and \
            matrix_rank(x_white) < NCOMP:
                print("Data 1 is rank defficient"
                    ". I cannot compute " +
                    str(NCOMP) + " components.")
                return (None, None, None, 1)

            if lrate1 < 1e-6:
                print("Weight matrix may"
                    " not be invertible...")
                return (None, None, None, 1)

            break
        else:
            error = 0

    return (weights, bias1, lrate1, error)

def ica_fuse_falsemaxdetect(self, data, trendPara, LtrendPara = 0.0):
        # function [Overindex ] = ica_fuse_falsemaxdetect(data, trendPara,LtrendPara)
        # % false maximun detect fucntion is to detect if a flase maximum occurs
        # % by checking entroy's trend along time.
        # % if there is  a decreaseing trend , then the false maximum occur;
        # % if osciilation for a long time without increasing occurs, Then the false
        # % maximum occur; 

        if  trendPara is None: 
            LtrendPara= 1e-4
            trendPara= -1e-3 #% the parameter -1e-3 or -5e-4, or -1e-4; need to test on simulation for overfitting problem with  low correlation
        elif LtrendPara == 0.0 :
            LtrendPara= 1e-4
        # end if

        if not (LtrendPara) :
            LtrendPara = 0.0001 
        if not (trendPara) :
            trendPara= -0.001
        
        Overindex = 0

        # % if osciilation for a long time without increasing occurs, Then the false maximum occur; 
        # aX + b ; degree=1. 
        # Check absolute of "a" to be less than LtrendPara        
        n = np.count_nonzero(data)

        if  n > 60 :
            x = np.arange(50)
            y = data[n-49:n+1] 
            y = data[n-49:n+1] - np.mean(data[n-49:n+1])
            p = np.polyfit(x,y,1)            
            # datat = data[n-49:n] - np.mean(data[n-49:n])
            # p = np.polyfit(data[0:49],datat,1)
            if abs(p[0]) < LtrendPara :     # Check "a" to be less than LtrendPara, then Overindex=1 to reduce Crate (Lamda= Crate)
                Overindex = 1
            # end if
        # end if

        # Allow the slop of entropy trend line to be as trendPara 
        # aX + b ; degree=1. 
        # Check "a" to be less than trendPara
        if not Overindex : 
            x = np.arange(5)       
            y = data[n-4:n+1]      
            y = data[n-4:n+1] - np.mean(data[n-4:n+1])
            p = np.polyfit(x,y,1)
            # r = datat - np.polyval(p,data[0:4])
            if p[0] < trendPara :           # Check "a" to be less than trendPara, then Overindex=1 to reduce Crate (Lamda= Crate)
                    Overindex = 1
            # end if        
        # end if

        return (Overindex)

def ica_fuse_corr(self, x, y): 
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

    # Loop over rows

    for ii in range(0,nx2):  
        for jj in range(0,ny2):    
            x_ii = x[:, ii]
            y_jj = y[:, jj]
            c[ii, jj] = ica_fuse_corr2(self, x[:, ii], y[:, jj]) # Find corr each colume
    # End loop over rows

    return c

def ica_fuse_corr2(self, x, y): 
    # computes correlation coefficient
    meanX = np.mean(x)      
    meanY = np.mean(y)      

    # Remove mean
    x = x - meanX       
    y = y - meanY       

    corr_coeff = np.sum(np.sum(x*y, axis=0)) / math.sqrt(np.sum(np.sum(x*x)) * np.sum(np.sum(y*y)))

    return corr_coeff  

def find_argmax_v2( coef_1_2, axis_input=1): # 
    # Calculate the indices of the maximum values along an axis
    # Input :
    #  - coef_1_2 : corrcoef between matrix 1 and matrix 2
    #  - axis_input : By default, the index is into the flattened array, otherwise along the specified axis. Default = 1 is by row.
    # Output :
    #  - max_index_row : maximum values along an axis


    NCOMP = coef_1_2.shape[0]
    Corr_matrix = abs(coef_1_2)
    coef_max_index_array = np.zeros(NCOMP).astype(int)
    for i in range(NCOMP) :        
        amax  = np.amax(Corr_matrix)
        amax_index = np.where(Corr_matrix == amax)
        amax_row = amax_index[0]
        amax_column = amax_index[1]
        amax_row = amax_row[0]
        amax_column = amax_column[0]

        # print('[LOG][Multirun ICA] Finding argmax -amax_index_pair  =', amax_index_pair , \
        #         'amax_index[0] = ', amax_index[0], '  amax_index[1]', amax_index[1])
        if axis_input == 1 : 
            coef_max_index_array[amax_row] = int(amax_column)
        elif axis_input == 0 :                     
            coef_max_index_array[amax_column] = int(amax_row)    
        Corr_matrix[amax_row,:] = 0
        Corr_matrix[:,amax_column] = 0

    # print('[LOG][Multirun ICA] Finding argmax - Result = ', coef_max_index_array )
    # print('[LOG][Multirun ICA] Finding argmax - Finish' )

    return coef_max_index_array   # 

def find_argmax_v3( coef_1_2, axis_input=1): # 
    # Same function with find_argmax_v2, but the goal is to find max value from both column and row.
    # Input :
    #  - coef_1_2 : corrcoef between matrix 1 and matrix 2
    #  - axis_input : By default, the index is into the flattened array, otherwise along the specified axis. Default = 1 is by row.
    # Output :
    #  - max_index_row : maximum values along an axis
    #     
    NCOMP = coef_1_2.shape[0]
    Corr_matrix = abs(coef_1_2)
    coef_max_index_array = np.zeros(NCOMP).astype(int)
    for i in range(NCOMP) :        
            amax  = np.amax(Corr_matrix)
            amax_index = np.where(Corr_matrix == amax)
            amax_row = amax_index[0]
            amax_column = amax_index[1]
            amax_row = int(amax_row[0])
            amax_column = int(amax_column[0])

            # print('[LOG][Multirun ICA] Finding argmax -amax_index_pair  =', amax_index_pair , \
            #         'amax_index[0] = ', amax_index[0], '  amax_index[1]', amax_index[1])
            if axis_input == 1 : 
                    coef_max_index_array[amax_row] = int(amax_column)
                    Corr_matrix[amax_row,:] = 0
                    Corr_matrix[:,amax_column] = 0                                
            elif axis_input == 0 :                     
                    coef_max_index_array[amax_column] = int(amax_row)    
                    Corr_matrix[amax_row,:] = 0
                    Corr_matrix[:,amax_column] = 0
    return coef_max_index_array   

def flip_negative( m, coef_flip_array): 
    NCOMP, r = m.shape
    for i in range(NCOMP) :
        if coef_flip_array[i] < 0 : 
            m[i] = -1 * m[i]
    # End or for i loop.

    return m

def pica_infomax_run_average5(self, XY, num_ica_runs, GlobalPCA_U,  w, s):    
    """Computes average ICA 
    *Input
        w : Globa W unmixer matrix from Infomax (r x r) or (components x components)
        s : Source matrix from Infomax (r x d) or (components x variable voxel)
        num_ica_runs : Number of times to run ica    
        XY  : Modality X or Y
    *Output
        A : GlobalA_mixer_X : mixing matrix
        S : S_sources_X : source matrix
        W : GlobalW_unmixer : unmixing matrix
    """

    NCOMP, r = np.array(w)[0,:,:].shape
    coef_All = []
    # max_index_row_All = []
    w_ordered_All = []
    s_ordered_All = []
    coef_s0_sj_ordered_All = []

    #########################
    # Normalizing
    #########################
    # Normalize weight_All by removing mean and by removing standard deviation of the array
    print('[LOG][Multirun ICA] De-mean ')

    #########################
    # Clustering
    #########################
    # Using the ordered cross-correlation algorithm
    print('[LOG][Multirun ICA] Clustering ')
    data_path_save = DATA_PATH_OUTPUT    
    # Define w0 index 
    w0_ordered = np.array(w)[0,:,:]
    s0_ordered = np.array(s)[0,:,:]
    w_ordered_All.append(w0_ordered)
    s_ordered_All.append(s0_ordered)

    for j in range (1, num_ica_runs):
        # Finding correlation from Source matrix
        w1 = np.array(w)[j,:,:]
        s1 = np.array(s)[j,:,:]        
        data_file_save =  "Correlation_Graph_"  + XY + "_s0_s" + str(j) + ".jpeg"    
        coef_s0_sj = dpica_report.pica_2d_correlate7(s0_ordered, s1, data_path_save, data_file_save, True, False, True)    
        coef_All.append(coef_s0_sj)

        # Finding numximum pair index from Source matrix
        max_index_row = find_argmax_v2(coef_s0_sj, 1) 
        w1_ordered = w1[max_index_row,:]        # Rearrange w1 in max_index_row(w0) order.
        s1_ordered = s1[max_index_row,:]        # Rearrange s1 in max_index_row(w0) order.
        data_file_save =  "Correlation_Graph_"  + XY + "_s0_s" + str(j) + "_ordered.jpeg"     
        coef_s0_sj_ordered = dpica_report.pica_2d_correlate7(s0_ordered, s1_ordered, data_path_save, data_file_save, True, False, True)     
        w_ordered_All.append(w1_ordered)
        s_ordered_All.append(s1_ordered)
        coef_s0_sj_ordered_All.append(coef_s0_sj_ordered)

    # end of for

    print('[LOG][Multirun ICA] Clustering - Finish')

    #########################
    # Re-arrange  positive/negative pattern
    #########################
    # Using the -1 algorithm (not absolute algorithm)
    print('[LOG][Multirun ICA] Re-arranging - pattern')

    for i in range(NCOMP) :
        ## Switch current row to opposite sign
        for j in range (1, num_ica_runs):
            # np.array(w)[j,:,:]
            coef_s0_sj_ordered = np.array(coef_s0_sj_ordered_All)[j-1,:,:]  
            w_ordered = np.array(w_ordered_All)[j,:,:]  
            if coef_s0_sj_ordered[i,i] < 0 : 
                w_ordered[i] = -1 * w_ordered[i]
                print("[LOG][Multirun ICA] Re-arranging - Component ", i , "of w", (j), "_[",i,"] is applied -1 as coef_s0_s",(j), " =", coef_s0_sj_ordered[i,i] )    
            # end of if
        # end of for j loop
    # end of for i loop


    # Save each Weight Correlation matrix
    for j in range (1, num_ica_runs):
        w_ordered = np.array(w_ordered_All)[j,:,:]  
        data_file_save =  "Correlation_Graph_"  + XY + "_w0_w" + str(j) + "_ordered_flipped.jpeg"   
        coef_w0_wj_ordered = dpica_report.pica_2d_correlate7(w0_ordered, w_ordered, data_path_save, data_file_save, True, False, True)    
        # print('[LOG][Multirun ICA] Re-arranging - pattern. Save ', data_file_save) 
    # End or for j loop.


    print('[LOG][Multirun ICA] Re-arranging - Finished')


    #########################
    # Computing the final weight W
    #########################
    # Using the Average algorithm
    print('[LOG][Multirun ICA] Average algorithm')

    GlobalW_unmixer = []
 

    w_ordered_All = np.array(w_ordered_All)

    # Compute GlobalA 
    GlobalW_unmixer = np.average(w_ordered_All, axis=0)  # [5, 8, 8] ==> [8, 8] Average by row 
    print('[LOG][Multirun ICA] Average algorithm - Done')

    # Compute GlobalA_mixer and S_source
    GlobalA_mixer = inv(GlobalW_unmixer)     
    S_sources = np.dot(GlobalW_unmixer, GlobalPCA_U)       

    return (self, GlobalA_mixer, S_sources, GlobalW_unmixer)

def pica_infomax_run_icasso6(self, XY,  num_ica_runs, GlobalPCA_U,  w, s):   
    """Computes ICASSO ICA with find_argmax function
    *Input
        w : Globa W unmixer matrix from Infomax (r x r) or (components x components)
        s : Source matrix from Infomax (r x d) or (components x variable voxel)
        num_ica_runs : Number of times to run ica    
        XY  : Modality X or Y
    *Output
        A : GlobalA_mixer_X : mixing matrix
        S : S_sources_X : source matrix
        W : GlobalW_unmixer : unmixing matrix
    """
    NCOMP, r = np.array(w)[0,:,:].shape
    coef_All = []
    # max_index_row_All = []
    w_ordered_All = []
    s_ordered_All = []
    coef_s0_sj_ordered_All = []

    #########################
    # Normalizing
    #########################
    # Normalize m1, m2, m3, m4, m5 by removing mean and by removing standard deviation of the array
            # x2d_demean = x2d - x2d.mean(axis=1).reshape((-1, 1))  
    print('[LOG][Multirun ICA] De-mean ')

    #########################
    # Clustering
    #########################
    # Using the ordered cross-correlation algorithm
    print('[LOG][Multirun ICA] Clustering ')
    data_path_save = DATA_PATH_OUTPUT    
    # Define w0 index 
    w0_ordered = np.array(w)[0,:,:]
    s0_ordered = np.array(s)[0,:,:]
    w_ordered_All.append(w0_ordered)
    s_ordered_All.append(s0_ordered)

    for j in range (1, num_ica_runs):
        # Finding correlation from Source matrix
        w1 = np.array(w)[j,:,:]
        s1 = np.array(s)[j,:,:]        
        data_file_save =  "Correlation_Graph_"  + XY + "_s0_s" + str(j) + ".jpeg"    
        coef_s0_sj = dpica_report.pica_2d_correlate7(s0_ordered, s1, data_path_save, data_file_save, True, False, True)    
        coef_All.append(coef_s0_sj)

        # Finding numximum pair index from Source matrix
        max_index_row = find_argmax_v2(coef_s0_sj, 1) 
        w1_ordered = w1[max_index_row,:]        # Rearrange w1 in max_index_row(w0) order.
        s1_ordered = s1[max_index_row,:]        # Rearrange s1 in max_index_row(w0) order.
        data_file_save =  "Correlation_Graph_"  + XY + "_s0_s" + str(j) + "_ordered.jpeg"     
        coef_s0_sj_ordered = dpica_report.pica_2d_correlate7(s0_ordered, s1_ordered, data_path_save, data_file_save, True, False, True)     
        w_ordered_All.append(w1_ordered)
        s_ordered_All.append(s1_ordered)
        coef_s0_sj_ordered_All.append(coef_s0_sj_ordered)

    # end of for


    print('[LOG][Multirun ICA] Clustering - Finish')

    #########################
    # Re-arrange  positive/negative pattern
    #########################
    # Using the -1 algorithm (not absolute algorithm)
    print('[LOG][Multirun ICA] Re-arranging - pattern')

    for i in range(NCOMP) :
        ## Switch current row to opposite sign
        for j in range (1, num_ica_runs):
            # np.array(w)[j,:,:]
            coef_s0_sj_ordered = np.array(coef_s0_sj_ordered_All)[j-1,:,:]  
            w_ordered = np.array(w_ordered_All)[j,:,:]  
            if coef_s0_sj_ordered[i,i] < 0 : 
                w_ordered[i] = -1 * w_ordered[i]
                print("[LOG][Multirun ICA] Re-arranging - Component ", i , "of w", (j), "_[",i,"] is applied -1 as coef_s0_s",(j), " =", coef_s0_sj_ordered[i,i] )    
            # end of if
        # end of for j loop
    # end of for i loop

    # Save each Weight Correlation matrix
    for j in range (1, num_ica_runs):
        w_ordered = np.array(w_ordered_All)[j,:,:]  
        data_file_save =  "Correlation_Graph_"  + XY + "_w0_w" + str(j) + "_ordered_flipped.jpeg"   
        coef_w0_wj_ordered = dpica_report.pica_2d_correlate7(w0_ordered, w_ordered, data_path_save, data_file_save, True, False, True)    
        # print('[LOG][Multirun ICA] Re-arranging - pattern. Save ', data_file_save) 
    # End or for j loop.

 
    print('[LOG][Multirun ICA] Re-arranging - Finished')

    #########################
    # Computing the final weight W
    #########################
    # Using the Centrotype algorithm
    print('[LOG][Multirun ICA] Centrotype algorithm')

    GlobalW_unmixer = []
    list_m = []


    for i in range(NCOMP) :        
        coef_max_sum_list = []
        coef_max_sum = -9999
        print("[LOG][Multirun ICA] Centrotype - Component ", i , "========================")

        for j in range(num_ica_runs) : 
            for k in range(num_ica_runs) : 
                if j != k and j < k:
                    w_j_ordered = np.array(w_ordered_All)[j,:,:]  
                    w_k_ordered = np.array(w_ordered_All)[k,:,:]  

                    data_file_save =  "Correlation_Graph_"  + XY + "_ICASSO_component_" + str(i) + "_w" + str(j) + "_w" + str(k) + ".jpeg"   
                    coef_component = dpica_report.pica_2d_correlate6(w_j_ordered[i], w_k_ordered[i], data_path_save, data_file_save, False, False, True)           
                    coef_component_wj_wk = np.corrcoef ( w_j_ordered[i], w_k_ordered[i])        
                    coef_component_wj_wk_sum = np.sum(coef_component_wj_wk[0,1])
                    print("[LOG][Multirun ICA] Centrotype - Component ", i , " coef_component_w" + str(j) + "_w" + str(k) + "_sum = " , coef_component_wj_wk_sum)    

                    if coef_component_wj_wk_sum > coef_max_sum :
                            coef_max_sum_list = w_k_ordered[i]
                            coef_max_sum = coef_component_wj_wk_sum                            
                            print("[LOG][Multirun ICA] Centrotype - Component ", i , "w" + str(j) + "_w" + str(k) + "_coef_max_sum = " , coef_max_sum)    

                    # end if w_j Vs w_k

        # print("[LOG][Multirun ICA] Centrotype - Component ", i , "coef_max_sum_list = " , coef_max_sum_list)    

        list_m.append(coef_max_sum_list)
        print("[LOG][Multirun ICA] Centrotype - Component ", i , "list_m = " , len(list_m), ".")
        print("[LOG][Multirun ICA] Centrotype - Component ", i , "========================")


    # End or for loop.


    # Compute GlobalA 
    GlobalW_unmixer = np.array(list_m)

    print('[LOG][Multirun ICA] Centrotype algorithm - Done')

    # Compute GlobalA_mixer and S_source
    GlobalA_mixer = inv(GlobalW_unmixer)     
    S_sources = np.dot(GlobalW_unmixer, GlobalPCA_U)       

    return (self, GlobalA_mixer, S_sources, GlobalW_unmixer)

def pica_multi_run_component_validation2(self, num_ica_runs,  w_X, w_Y, s_X, s_Y):   
    """Validating top component of pICA of each run
    *Input
        w : Globa W unmixer matrix from Infomax (r x r) or (components x components). Of X and Y
        s : Source matrix from Infomax (r x d) or (components x variable voxel). Of X and Y
        num_ica_runs : Number of times to run ica   
    *Output
        max_pair_X : List of Top component X
        max_pair_Y : List of Top component Y
    """

    #########################

    data_path_save = DATA_PATH_OUTPUT    

    self.max_pair_output_X = []
    self.max_pair_output_X.append([1, ""]) 
    self.max_pair_output_Y = []
    self.max_pair_output_Y.append([1, ""]) 
    self.max_pair_output_XY = []
    # self.max_pair_output_XY.append([1, ""]) 



    #########################
    # Reconstruction top component
    #########################

    print('[LOG][Top pair ICA] top component - Start')

    LocalA_Corr_All_ordered_X = []
    LocalA_Corr_All_ordered_Y = []

    for run in range (num_ica_runs):
        print("[LOG][Top pair ICA] top component - Local reconstruction of run " + str(run+1))

        LocalA_Corr_A0_X = (self.local_reconstruction9(w_X[run], inv(w_X[run]), \
            self.GlobalPCA_dewhite_X, self.Con_deWhite_X, self.NSUB_ALL_X, self.NCOM_ALL_X, SITE_NUM, B_GLOBAL, False))      
        LocalA_Corr_All_ordered_X.append(LocalA_Corr_A0_X)

        LocalA_Corr_A0_Y = (self.local_reconstruction9(w_Y[run], inv(w_Y[run]), \
            self.GlobalPCA_dewhite_Y, self.Con_deWhite_Y, self.NSUB_ALL_Y, self.NCOM_ALL_Y, SITE_NUM, B_GLOBAL, False))      
        LocalA_Corr_All_ordered_Y.append(LocalA_Corr_A0_Y)


    # Save run=0
    run = 0
    if run == 0 :
        np.savetxt( DATA_PATH_OUTPUT + "ICA_S_sources_ordered_X_" + str(run) + ".csv", np.array(s_X)[run,:,:], delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "ICA_LocalA_Corr_ordered_X_" + str(run) + ".csv", LocalA_Corr_All_ordered_X[run], delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "ICA_S_sources_ordered_Y_" + str(run) + ".csv", np.array(s_Y)[run,:,:], delimiter=",")
        np.savetxt( DATA_PATH_OUTPUT + "ICA_LocalA_Corr_ordered_Y_" + str(run) + ".csv", LocalA_Corr_All_ordered_Y[run], delimiter=",")

        print("[LOG][Top pair ICA] top component - Local Top Pair of run1 ")

# A_XY                
        data_file_save =  "Top_Pair_Graph_"  +  "_A1_ordered_flipped.jpeg"   
        label_X = "Local A of Centralized A" + str(run+1) + " of X"
        label_Y = "Local A of Centralized A" + str(run+1) + " of Y"
        title_name = 'Centralized : ' + str(NUM_SUBJECT) + ' subject \n Local A of XY'                  

        Coef_max_pair_output_A0_Ai_XY = dpica_report.pica_2d_correlate15(LocalA_Corr_All_ordered_X[run].T, LocalA_Corr_All_ordered_Y[run].T, \
                data_path_save, data_file_save, title_name , label_X, label_Y, 1, True, True, True, True)   

        self.max_pair_output_XY.append([run+1, (Coef_max_pair_output_A0_Ai_XY)])  


    for run in range (1, num_ica_runs):
        print("[LOG][Top pair ICA] top component - Local Top Pair of run1 and run" + str(run+1))

# S_X
        file_name_X = "ICA_S_sources_X_1.csv"     
        file_name_Y = "ICA_S_sources_X_" + str(run+1) + ".csv"     

        Data_X = np.array(s_X)[0,:,:]
        Data_Y = np.array(s_X)[run,:,:]

        data_file_save =  "Correlation_Graph_" + file_name_X + "_" + file_name_Y + "_size_" + str(Data_Y.shape[0]) + ".jpeg"
        title_name = 'Centralized Vs Centralized \n Source S of ' + str(NUM_SUBJECT) + ' subject'
        label_X = "Source S of Centralized X_1"
        label_Y = "Source S of Centralized X_" + str(run+1) 
        
        Coef_2d_output = dpica_report.pica_2d_correlate10(Data_X, Data_Y, \
                data_path_save, data_file_save, title_name , label_X, label_Y, 1, False, False, False, False)         
        max_index_row_X = (find_argmax_v3(Coef_2d_output, 1))  
        Data_Y = Data_Y[max_index_row_X,:]   
        np.savetxt( DATA_PATH_OUTPUT + "ICA_S_sources_ordered_X_" + str(run) + ".csv", Data_Y, delimiter=",")

# A_X                
        data_file_save =  "Top_Pair_Graph_"  +  "_A1_A" + str(run+1) + "_ordered_flipped.jpeg"   
        label_X = 'Local A of Centralized A1 of X'
        label_Y = "Local A of Centralized A" + str(run+1) + " of X"
        title_name = 'Centralized : ' + str(NUM_SUBJECT) + ' subject \n Local A'                  

        L_A = LocalA_Corr_All_ordered_X[run]
        LocalA_Corr_All_A1_X = L_A[:,max_index_row_X] 

        Coef_max_pair_output_A0_Ai_X = dpica_report.pica_2d_correlate15(LocalA_Corr_All_ordered_X[0].T, LocalA_Corr_All_A1_X.T, \
                data_path_save, data_file_save, title_name , label_X, label_Y, 1, True, True, True, True)   

        self.max_pair_output_X.append([run+1, (Coef_max_pair_output_A0_Ai_X)])  

        np.savetxt( DATA_PATH_OUTPUT + "ICA_LocalA_Corr_ordered_X_" + str(run) + ".csv", LocalA_Corr_All_A1_X, delimiter=",")
        


# S_Y
        file_name_X = "ICA_S_sources_Y_1.csv"     
        file_name_Y = "ICA_S_sources_Y_" + str(run+1) + ".csv"     

        Data_X = np.array(s_Y)[0,:,:]
        Data_Y = np.array(s_Y)[run,:,:]

        data_file_save =  "Correlation_Graph_" + file_name_X + "_" + file_name_Y + "_size_" + str(Data_Y.shape[0]) + ".jpeg"
        title_name = 'Centralized Vs Centralized \n Source S of ' + str(NUM_SUBJECT) + ' subject'
        label_X = "Source S of Centralized Y_1"
        label_Y = "Source S of Centralized Y_" + str(run+1) 
        
        Coef_2d_output = dpica_report.pica_2d_correlate10(Data_X, Data_Y, \
                data_path_save, data_file_save, title_name , label_X, label_Y, 1, False, False, False, False)         
        max_index_row_Y = (find_argmax_v3(Coef_2d_output, 1))  
        Data_Y = Data_Y[max_index_row_Y,:]   
        np.savetxt( DATA_PATH_OUTPUT + "ICA_S_sources_ordered_Y_" + str(run) + ".csv", Data_Y, delimiter=",")

# A_Y                
        data_file_save =  "Top_Pair_Graph_"  +  "_A1_A" + str(run+1) + "_ordered_flipped.jpeg"   
        label_X = 'Local A of Centralized A1 of Y'
        label_Y = "Local A of Centralized A" + str(run+1) + " of Y"
        title_name = 'Centralized : ' + str(NUM_SUBJECT) + ' subject \n Local A Y'                  

        L_A_Y = LocalA_Corr_All_ordered_Y[run]
        LocalA_Corr_All_A1_Y = L_A_Y[:,max_index_row_Y] 

        Coef_max_pair_output_A0_Ai_Y = dpica_report.pica_2d_correlate15(LocalA_Corr_All_ordered_Y[0].T, LocalA_Corr_All_A1_Y.T, \
                data_path_save, data_file_save, title_name , label_X, label_Y, 1, True, True, True, True)   

        self.max_pair_output_Y.append([run+1, (Coef_max_pair_output_A0_Ai_Y)])  

        np.savetxt( DATA_PATH_OUTPUT + "ICA_LocalA_Corr_ordered_Y_" + str(run) + ".csv", LocalA_Corr_All_A1_Y, delimiter=",")


# A_XY                
        data_file_save =  "Top_Pair_Graph_"  +  "_A1_A" + str(run+1) + "_ordered_flipped.jpeg"   
        label_X = "Local A of Centralized A" + str(run+1) + " of X"
        label_Y = "Local A of Centralized A" + str(run+1) + " of Y"
        title_name = 'Centralized : ' + str(NUM_SUBJECT) + ' subject \n Local A of XY'                  

        Coef_max_pair_output_A0_Ai_XY = dpica_report.pica_2d_correlate15(LocalA_Corr_All_A1_X.T, LocalA_Corr_All_A1_Y.T, \
                data_path_save, data_file_save, title_name , label_X, label_Y, 1, True, True, True, True)   

        self.max_pair_output_XY.append([run+1, (Coef_max_pair_output_A0_Ai_XY)])  




    print('[LOG][Top pair ICA] top component - Finish')

    return (self, Coef_max_pair_output_A0_Ai_X, Coef_max_pair_output_A0_Ai_Y, Coef_max_pair_output_A0_Ai_XY)

def findsteplength1(fk, deltafk, x1, x2, alphak, P, c1, c2):

    # Initialization    
    con = 1
    coml = len(x1)

    while (con) and (con < 100) :
        xnew = x1 + alphak * P
        # fk1 = corrcoef(xnew, x2)
        fk0 = np.corrcoef(xnew, x2)        
        
        # print ("fk1 = " ,  fk1)
        fk1 = - fk0[ 0, 1] ** 2
        # tcov = ((xnew - np.mean(xnew)) * (x2 - np.mean(x2)).T)   /    (coml-1)
        # tcov_x2T = (x2 - np.mean(x2)).T
        # tcov_x2 = (x2 - np.mean(x2))
        tcov_dot = np.dot((xnew - np.mean(xnew)) , (x2 - np.mean(x2)).T)
        # tcov_dotT = np.dot((xnew - np.mean(xnew)) , (x2 - np.mean(x2)))
        tcov_dot_coml = tcov_dot /    (coml-1)

        tcov = tcov_dot_coml

        comterm = 2*(tcov)/np.var(xnew,ddof=1)/np.var(x2,ddof=1)
        deltafk1 = -comterm*(x2-np.mean(x2) + tcov*(np.mean(xnew)-xnew)/np.var(xnew,ddof=1))    # 1st order derivative  #PT=-1.960159432381532  #ML=-1.960159432381532
                # firstterm1=(fk1-fk)/c1;                
        firstterm1 = (fk1-fk) / c1  # PT=-2678.735196765395  ML =  -2.678735196765395e+03
                # firstterm2 = deltafk * P[:]
        firstterm2 = np.dot(deltafk , P[:] )   
        firstterm2_temp = np.dot(deltafk , P )   
                # secondterm1 = deltafk1 * P[:]
        secondterm1 = np.dot(deltafk1 , P[:])
                # secondterm2 = deltafk * P[:]
        secondterm2 = np.dot(deltafk , P[:])

        if (firstterm1 > alphak*firstterm2) :
            if firstterm1 <0 :
                alphak = 0.9 * abs(firstterm1/firstterm2)
            else:
                alphak = 0.1 * alphak
            
            if alphak < 1e-6 :
                alphak = 0
                con = 0
            else:  
                con = con + 1
        
            
        elif (secondterm1 < 0) and (secondterm1 < c2*secondterm2):
            # %alphak=abs(secondterm1/secondterm2/c2)*alphak;
            con = con + 1
            alphak = 1.1*alphak
        elif (secondterm1 > 0) and (secondterm2 < 0):
            alphak = 0.9*alphak
            con = con + 1
        else:
            con = 0

        # End of if 
    # End of While

    if (con >= 50 ):
        alphak = 0
        print("Clearning rate searching for fMRI data failed!")

    return alphak

def pica_infomax8(self):    

    """Computes ICA infomax in whitened data
    Decomposes x_white as x_white=AS
    *Input
    STEP : Array of STEP number of Modality         <== Seems not use. No need to pass this parameter
    STOPSIGN : Array of STOPSIGN Boolean flag of Modality <== Seems not use. No need to pass this parameter
    x_white: whitened data (Use PCAwhiten)
    verbose: flag to print optimization updates
    *Output
    A : mixing matrix
    S : source matrix
    W : unmixing matrix
    STEP : Number of STEP
    STOPSIGN : Flag of STOPSIGN to stop updating Weight
    """
    # print("INFOMAX...Start")      



    # Initialization
    self.maxsteps = DEFAULT_MAXSTEPS

    data1 = copy.copy(self.GlobalPCA_U_X)
    STEP_X = 0
    STOPSIGN_X = 0


    self.Global_NCOM_X = self.GlobalPCA_U_X.shape[0]        # Global_NCOM
    self.old_weight_X = np.eye(self.Global_NCOM_X)
    bias_X = np.zeros((self.Global_NCOM_X, 1))
    self.d_weight_X = np.zeros(self.Global_NCOM_X)
    self.old_d_weight_X = np.zeros(self.Global_NCOM_X)
    sphere_X = []
    weight_X = []
    chans_X, frames_X =  data1.shape #% determine the data size
    urchans_X = chans_X    #% remember original data channels
    datalength_X = frames_X
    DEFAULT_BLOCK_X = int(np.floor(np.sqrt(frames_X/3)))
    DEFAULT_LRATE_X = 0.015/np.log(chans_X)
    delta_X = []
    wts_blowup_X = 0
    activations_X = []
    winvout_X = []



    data2 = copy.copy(self.GlobalPCA_U_Y)
    STEP_Y = 0
    STOPSIGN_Y = 0        

    self.Global_NCOM_Y = self.GlobalPCA_U_Y.shape[0]        # Global_NCOM
    self.old_weight_Y = np.eye(self.Global_NCOM_Y)
    bias_Y = np.zeros((self.Global_NCOM_Y, 1))
    self.d_weight_Y = np.zeros(self.Global_NCOM_Y)
    self.old_d_weight_Y = np.zeros(self.Global_NCOM_Y)
    sphere_Y = []
    weight_Y = []
    chans_Y, frames_Y = data2.shape #% determine the data size
    urchans_Y = chans_Y    #% remember original data channels
    datalength_Y = frames_Y
    DEFAULT_BLOCK_Y = int(np.floor(np.sqrt(frames_Y/3)))
    DEFAULT_LRATE_Y = 0.015/np.log(chans_Y)
    delta_Y = []
    wts_blowup_Y = 0
    activations_Y = []
    winvout_Y = []

    DEFAULT_BLOCK = [DEFAULT_BLOCK_X,DEFAULT_BLOCK_Y]
    block = DEFAULT_BLOCK
    DEFAULT_LRATE = [DEFAULT_LRATE_X,DEFAULT_LRATE_Y]
    lrate = DEFAULT_LRATE

    # %%%%%%%%%%%%%%%%%%%%%% Declare defaults used below %%%%%%%%%%%%%%%%%%%%%%%%
    # %

    # %
    # %%%%%%%%%%%%%%%%%%%%%%% Set up keyword default values %%%%%%%%%%%%%%%%%%%%%%%%%
    # %

    # %
    # %%%%%%%%%% Collect keywords and values from argument list %%%%%%%%%%%%%%%
    # %
    Keyword = ''    # Keyword = eval(['p',int2str((i-3)/2 +1)]);
    Value = ''      #Value = eval(['v',int2str((i-3)/2 +1)]);
    Keyword = Keyword.lower() #% convert upper or mixed case to lower

    weights = 0             # fprintf(...'runica() weights value must be a weight matrix or sphere')
    wts_passed =1

    ncomps = self.Global_NCOM_X    # fprintf(..'runica() pca value should be the number of principal components to retain')
    pcaflag = 'off'          # fprintf(..'runica() pca value should be the number of principal components to retain')
    posactflag = DEFAULT_POSACTFLAG        # fprintf('runica() posact value must be on or off')
    lrate = DEFAULT_LRATE   # fprintf('runica() lrate value is out of bounds');
    block = DEFAULT_BLOCK   # fprintf('runica() block size value must be a number')
    nochange = DEFAULT_STOP # fprintf('runica() stop wchange value must be a number')
    maxsteps   = DEFAULT_MAXSTEPS # fprintf('runica() maxsteps value must be a positive integer')
    annealstep = 0          # fprintf('runica() anneal step value must be (0,1]')
    annealdeg = DEFAULT_ANNEALDEG  # fprintf('runica() annealdeg value is out of bounds [0,180]')
    momentum = 0            # fprintf('runica() momentum value is out of bounds [0,1]')
    sphering = 'on'         # fprintf('runica() sphering value must be on or off')
    biasflag = 1            # fprintf('runica() bias value must be on or off')    ## 1 or 0
    srate = 0               # fprintf('runica() specgram srate must be >=0')
    loHz = 0                # fprintf('runica() specgram loHz must be >=0 and <= srate/2')
    hiHz = 1                # fprintf('runica() specgram hiHz must be >=loHz and <= srate/2')
    Hzinc = 1               # fprintf('runica() specgram Hzinc must be >0 and <= hiHz-loHz')
    Hzframes = self.GlobalPCA_U_X.shape[1] / 2 # fprintf('runica() specgram frames must be >=0 and <= data length')
    
 
    extended = 0 #1 #0            # % turn on extended-ICA
    extblocks = DEFAULT_EXTBLOCKS           # % number of blocks per kurt() compute

    # %%%%%%%%%%%%%%%%%%%%%%%% Connect_threshold computation %%%%%%%%%%%%%%%%%%%%%%%%


    N = sum(self.NSUB_ALL_X)
    # 1 site
    # N = self.NSUB_X1
    # # 3 sites
    # if SITE_NUM == 3 :
    #     N = self.NSUB_X1 + self.NSUB_X2 + self.NSUB_X3    
    # elif SITE_NUM == 2 :
    #     N = self.NSUB_X1 + self.NSUB_X2    
    # Experiment    

    # Experiment    

    if CONSTRAINED_CONNECTION_AUTO :
        Connect_threshold =  self.p_to_r2 (N)  # % set a threshold to select columns constrained. # 0.20912 

    else:
        Connect_threshold =  CONSTRAINED_CONNECTION  # Set to 1

    print('    ')
    print('    ')
    print('[LOG]=====INFOMAX=====')
    print('[LOG]Number of subject =  ', N )
    print('[LOG]Global_NCOM_X =  ', Global_NCOM_X)  
    print('[LOG]Global_NCOM_Y =  ', Global_NCOM_Y)  
    print('[LOG]CONSTRAINED_CONNECTION =  %10.10f,' %(Connect_threshold))
    print('[LOG]CONSTRAINED_CONNECTION_PROABILITY =  ', CONSTRAINED_CONNECTION_PROABILITY)  # CONSTRAINED_CONNECTION_PROABILITY = 0.025

    # %%%%%%%%%%%%%%%%%%%%%%%% Initialize weights, etc. %%%%%%%%%%%%%%%%%%%%%%%%

    if not extended :
        annealstep = DEFAULT_ANNEALSTEP     # 0.90;DEFAULT_ANNEALSTEP   = 0.90
    else:    
        annealstep = DEFAULT_EXTANNEAL      # 0.98;DEFAULT_EXTANNEAL    = 0.98


    if annealdeg :
        annealdeg  = DEFAULT_ANNEALDEG - momentum*90    #; % heuristic DEFAULT_ANNEALDEG    = 60; 
        if annealdeg < 0 :
            annealdeg = 0

    if ncomps >  chans_X or ncomps < 1 :
        print ('runica(): number of components must be 1 to %d.' %chans_X)
        return

    #% initialize weights
    #if weights ~= 0,   # weights =0

    # %
    # %%%%%%%%%%%%%%%%%%%%% Check keyword values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %
    if frames_X < chans_X  :
        print ('runica(): X : data length %d < data channels %f!' %(frames_X,chans_X) )
        return
    elif frames_Y < chans_Y :
        print ('runica(): X : data length %d < data channels %f!' %(frames_Y,chans_Y) )
        return
    # elseif block < 2,
    #     fprintf('runica(): block size %d too small!\n',block)
    #     return
    # elseif block > frames,
    #     fprintf('runica(): block size exceeds data length!\n');
    #     return
    # elseif floor(epochs) ~= epochs,
    #     fprintf('runica(): data length is not a multiple of the epoch length!\n');
    #     return
    # elseif nsub > ncomps
    #     fprintf('runica(): there can be at most %d sub-Gaussian components!\n',ncomps);
    #     return
    # end;

    # %
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Process the data %%%%%%%%%%%%%%%%%%%%%%%%%%
    # %

    if verbose :
        print (' Input data X size [%d,%d] = %d channels, %d frames.' \
            %(chans_X, frames_X,chans_X, frames_X))
        print (' Input data Y size [%d,%d] = %d channels, %d frames.' \
            %(chans_Y, frames_Y,chans_Y, frames_Y))
        
        if pcaflag == 'on' :
            print (' After PCA dimension reduction,  finding ')
        else:
            print (' Finding ')
        
        if ~extended :
            print (' %d ICA components using logistic ICA.' %ncomps)
        else : #% if extended
            print (' %d ICA components using extended ICA.',ncomps)
            if extblocks > 0 :
                print ('Kurtosis will be calculated initially every %d blocks using %d data points.' %(extblocks,MAX_KURTSIZE))
            else :
                print ('Kurtosis will not be calculated. Exactly %d sub-Gaussian components assumed.'% nsub)
            # end of if extblocks > 0 :
        # end of if ~extended :

        print ('Initial X learning rate will be %g, block size %d.'%(lrate[0],block[0]))
        print ('Initial Y learning rate will be %g, block size %d.'%(lrate[1],block[1]))

        if momentum > 0: 
            print ('Momentum will be %g.\n'%momentum)

        print ('Learning rate will be multiplied by %g whenever angledelta >= %g deg.'%(annealstep,annealdeg))
        print ('Training will end when wchange < %g or after %d steps.' %(nochange,maxsteps))
        if biasflag :
            print ('Online bias adjustment will be used.')
        else:
            print ('Online bias adjustment will not be used.')
        # end of if biasflag :
    # end of  if verbose :
    # %
    # %%%%%%%%%%%%%%%%%%%%%%%%% Remove overall row means %%%%%%%%%%%%%%%%%%%%%%%%
    # %
    # %if verbose,
    # %    fprintf('Removing mean of each channel ...\n');
    # %end
    print ('Not removing mean of each channel!!!')
    # %data = data - mean(data')'*ones(1,frames);      % subtract row means

    if verbose :
        print ('Final training data1 range: %g to %g' % (np.amin(data1),np.amax(data1)))
        print ('Final training data1 range: %g to %g' % (np.amin(data2),np.amax(data2)))

    # %
    # %%%%%%%%%%%%%%%%%%% Perform PCA reduction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %
    if pcaflag =='on' :
        print ('Reducing the data to %d principal dimensions...\n',ncomps)
        # [eigenvectors1,eigenvalues1,data1] = pcsquash(data1,ncomps(1)); % changed for two dsatasets
        # [eigenvectors2,eigenvalues2,data2] = pcsquash(data2,ncomps(2)); % changed for two datasets
        # % make data its projection onto the ncomps-dim principal subspace
    # end of if pcaflag =='on' :

    # %
    # %%%%%%%%%%%%%%%%%%% Perform specgram transformation %%%%%%%%%%%%%%%%%%%%%%%
    # %
    if srate > 0 :
        print ('srate > 0 ')
        # % [P F T] = SPECGRAM(A,NFFT,Fs,WINDOW,NOVERLAP)
        # Hzwinlen =  fix(srate/Hzinc);
        # Hzfftlen = 2^(ceil(log(Hzwinlen)/log(2)));
        # if (Hzwinlen>Hzframes)
        #     Hzwinlen = Hzframes;
        # end
        # Hzoverlap = 0;
        # if (Hzwinlen>Hzframes)
        #     Hzoverlap = Hzframes - Hzwinlen;
        # end
        # % get freqs and times
        # [tmp,freqs,tms] = specgram(data(1,:),Hzfftlen,srate,Hzwinlen,Hzoverlap);
        # fs = find(freqs>=loHz & freqs <= hiHz);
        # % fprintf('   size(fs) = %d,%d\n',size(fs,1),size(fs,2));
        # % fprintf('   size(tmp) = %d,%d\n',size(tmp,1),size(tmp,2));
        # specdata = reshape(tmp(fs,:),1,length(fs)*size(tmp,2));
        # specdata = [real(specdata) imag(specdata)];
        # for ch=2:chans
        #     [tmp] = specgram(data(ch,:),Hzwinlen,srate,Hzwinlen,Hzoverlap);
        #     tmp = reshape((tmp(fs,:)),1,length(fs)*size(tmp,2));
        #     specdata = [specdata;[real(tmp) imag(tmp)]]; % channels are rows
        # end
        # fprintf('Converted data to %d channels by %d=2*%dx%d points spectrogram data.\n',chans,2*length(fs)*length(tms),length(fs),length(tms));
        # fprintf('   Low Hz %g, high Hz %g, Hz inc %g, window length %d\n',freqs(fs(1)),freqs(fs(end)),freqs(fs(2))-freqs(fs(1)),Hzwinlen);
        # data = specdata;
        # datalength=size(data,2);
    # end of if srate > 0 
    # %
    # %%%%%%%%%%%%%%%%%%% Perform sphering %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %        



    if sphering == 'on' :
        if verbose :
            print ('Computing the sphering matrix...')
        sphere_X_1 = np.cov(data1.T,rowvar=False )     
        # sphere_X_2 = (sqrtm( sphere_X_1   ))     
        sphere_X_2 = (sqrtm( sphere_X_1   )).real
        sphere_X_3 = inv(sphere_X_2)     
        sphere_X = 2.0*sphere_X_3    


        # sphere_X = 2.0*inv(sqrtm(np.cov(data1.T,rowvar=False )))     
        # sphere_Y = 2.0*inv(sqrtm(np.cov(data2.T,rowvar=False )))     

        sphere_Y_1 = np.cov(data2.T,rowvar=False )     
        # sphere_Y_2 = (sqrtm( sphere_Y_1   ))     
        sphere_Y_2 = (sqrtm( sphere_Y_1   )).real     
        sphere_Y_3 = inv(sphere_Y_2)     
        sphere_Y = 2.0*sphere_Y_3   #   find the "sphering" matrix = spher()

        if not weights :
            if verbose :
                print (' Starting weights are the identity matrix ...')

            weights=1

            weight_X = np.eye(self.Global_NCOM_X, chans_X) #% begin with the identity matrix
            weight_Y = np.eye(self.Global_NCOM_Y, chans_Y) #% begin with the identity matrix

        else  :  #% weights given on commandline
            if verbose :
                print (' Using starting weights named on commandline ...')
            
        # end of if not weights :
        if verbose :
            print (' Sphering the data ...')
                    
        data1 = copy.copy(np.dot(sphere_X,data1))     # % actually decorrelate the electrode signals
        data2 = copy.copy(np.dot(sphere_Y,data2))     # % actually decorrelate the electrode signals
    elif sphering == 'off' : # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if  not weights :
            if verbose : 
                print (' Using the sphering matrix as the starting weight matrix ...')
                print (' Returning the identity matrix in variable "sphere" ...')

            # ExperimentPCAComponent                  
            sphere_X = 2.0*np.inv(sqrtm(np.cov(data1.T,rowvar=False))) # % find the "sphering" matrix = spher()
            weight_X = np.eye(self.Global_NCOM_X,chans_X) * sphere_X # % begin with the identity matrix
            sphere_X = np.eye(chans_X)               #   % return the identity matrix

            sphere_Y = 2.0*np.inv(sqrtm(np.cov(data2.T,rowvar=False)))  # ; % find the "sphering" matrix = spher()
            weight_Y = np.eye(self.Global_NCOM_Y,chans_Y) * sphere_Y # % begin with the identity matrix
            sphere_Y = np.eye(chans_Y)               #   % return the identity matrix
            
        else : # % weights ~= 0
            if verbose :
                print (' Using starting weights named on commandline ...')
                print (' Returning the identity matrix in variable "sphere" ...')
            
            sphere_X = np.eye(chans_X)             #  % return the identity matrix
            sphere_Y = np.eye(chans_Y)             #  % return the identity matrix
        # end not weights :
    elif sphering == 'none':
        sphere_X = np.eye(chans_X)             #  % return the identity matrix
        sphere_Y = np.eye(chans_Y)             #  % return the identity matrix
        if not weights  : 
            if verbose :
                print (' Starting weights are the identity matrix ...')
                print (' Returning the identity matrix in variable "sphere" ...')
            # end of if verbose :

            weight_X = np.eye(self.Global_NCOM_X, chans_X) #% begin with the identity matrix
            weight_Y = np.eye(self.Global_NCOM_Y, chans_Y) #% begin with the identity matrix

        else : # % weights ~= 0
            if verbose : 
                print (' Using starting weights named on commandline ...')
                print (' Returning the identity matrix in variable "sphere" ...')
            # end of if verbose :
        # end not weights :
        sphere_X = np.eye(chans_X,chans_X)              #  % return the identity matrix
        sphere_Y = np.eye(chans_Y,chans_Y)              #  % return the identity matrix            
        if verbose :
            print ('Returned variable "sphere" will be the identity matrix.')
        # end of if verbose 
    #end sphering == 'on' :



    # %
    # %%%%%%%%%%%%%%%%%%%%%%%% Initialize ICA training %%%%%%%%%%%%%%%%%%%%%%%%%
    # %


    # 
    lastt_X = np.fix((datalength_X/block[0]-1)*block[0]+1)   # Matlab [57568,357]  Python 57568.0  357.0  
    lastt_Y = np.fix((datalength_Y/block[1]-1)*block[1]+1)
    degconst = 180/np.pi


    BI_X = block[0] * np.eye(self.Global_NCOM_X,self.Global_NCOM_X)
    BI_Y = block[1] * np.eye(self.Global_NCOM_Y,self.Global_NCOM_Y) 
    delta_X = np.zeros((1,chans_X * chans_X))
    delta_Y = np.zeros((1,chans_Y * chans_Y))
    change_X = 1
    change_Y = 1
    oldchange_X = 0
    oldchange_Y = 0
    startweight_X = copy.copy(weight_X)
    startweight_Y = copy.copy(weight_Y)
    prevweight_X = copy.copy(startweight_X)
    prevweight_Y = copy.copy(startweight_Y)
    oldweight_X = copy.copy(startweight_X)
    oldweight_Y = copy.copy(startweight_Y)

    prevwtchange_X = np.zeros((chans_X,self.Global_NCOM_X))
    prevwtchange_Y = np.zeros((chans_Y,self.Global_NCOM_Y))      
    oldwtchange_X = np.zeros((chans_X,self.Global_NCOM_X))       
    oldwtchange_Y = np.zeros((chans_Y,self.Global_NCOM_Y))

    lrates_X = np.zeros((1,maxsteps))
    lrates_Y = np.zeros((1,maxsteps))
    onesrow_X = np.ones((1,block[0]))
    onesrow_Y = np.ones((1,block[1]))

    signs_X = np.ones((1,self.Global_NCOM_X)) #    % initialize signs to nsub -1, rest +1
    signs_Y = np.ones((1,self.Global_NCOM_Y)) #    % initialize signs to nsub -1, rest +1

    for k in range(1,nsub) : 
        signs_X[k] = -1
        signs_Y[k] = -1
    # end for
    
    if extended and extblocks < 0 and verbose :
        print('Fixed extended-ICA sign assignments:  ')

    # end if

    signs_X = np.diag(signs_X) # % make a diagonal matrix
    signs_Y = np.diag(signs_Y) # % make a diagonal matrix

    oldsigns_X = np.zeros(signs_X.size)
    oldsigns_Y = np.zeros(signs_Y.size)
    change_X = 0.0
    change_Y = 0.0
    signcount_X = 0 #   % counter for same-signs
    signcount_Y = 0 #   % counter for same-signs
    signcounts_X = 0
    signcounts_Y = 0
    urextblocks = copy.copy(extblocks) #  % original value, for resets


    old_kk_X = np.zeros((1,self.Global_NCOM_X)) #   % for kurtosis momemtum
    old_kk_Y = np.zeros((1,self.Global_NCOM_Y)) #   % for kurtosis momemtum

    # %
    # %%%%%%%% ICA training loop using the logistic sigmoid %%%%%%%%%%%%%%%%%%%
    # %

    if verbose :
        print('Beginning ICA training ...')
        if extended :
            print(' first training step may be slow ...')
        else:
            print('\n')
        # end if
    # end if
    STEP_X = 0
    STEP_Y = 0
    blockno_X = 0  # MATLAB = 1
    blockno_Y = 0  # MATLAB = 1
    STOPSIGN_X  = 0
    STOPSIGN_Y  = 0

    alphak_X = 1
    alphak_Y = 1  # %alphak_R=[1,1];
    Crate_X = copy.copy(CRATE_X)
    Crate_Y = copy.copy(CRATE_Y)


    lrate_X = DEFAULT_LRATE_X  #Dataset X step 1 - lrate 0.000014
    lrate_Y = DEFAULT_LRATE_Y
    lossf_X = np.zeros(maxsteps+1)
    lossf_Y = np.zeros(maxsteps+1)

    angledelta_X = 0        
    angledelta_Y = 0        

    entropy_X = 0
    entropy_Y = 0

    entropychange_X = 0
    entropychange_Y = 0

    # Entropy   
    Loop_num = 1
    Loop_list = []
    Loop_list_X = []
    Loop_list_Y = []
    STEP_list_X = []
    STEP_list_Y = []
    STOPSIGN_list_X = []
    STOPSIGN_list_Y = []
    Crate_list_X = []
    Crate_list_Y = []
    entropy_list_X = []
    entropy_list_Y = []
    entropychange_list_X = []
    entropychange_list_Y = []
    costfunction_corrcoef_list_maxcorr_XY = []
    costfunction_corrcoef_list_maxcorr_test_XY = []
    mymaxcorr = 0
    mymaxcorr_test = 0
    myentropy_X = 0
    myentropy_Y = 0



    print("[LOG]=====INFOMAX=====maxsteps=", maxsteps)

    # while ((STEP_X < maxsteps or STEP_Y < maxsteps) and \
    #     (not STOPSIGN_X and not STOPSIGN_X) ):
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    while (STEP_X < maxsteps or STEP_Y < maxsteps) :

        # Entropy
        Loop_list.append(Loop_num)
        Loop_list_X.append([Loop_num,Loop_num])
        Loop_list_Y.append([Loop_num,Loop_num])

        if not STOPSIGN_X :

            eps = np.finfo(np.float32).eps
            u = copy.copy(np.dot(weight_X , data1) + np.dot( bias_X , np.ones((1,frames_X)) ))
            y = copy.copy(1 / (1 + np.exp(-u)  ))

            yy = copy.copy((1-y))                              
            temp1 = copy.copy(np.dot(weight_X,y))              
            temp2 = copy.copy((temp1 * yy))                    
            temp = copy.copy(np.log( abs( temp2 ) + eps))  

            entropy_X = np.mean(temp)      
            # Entropy            
            entropy_list_X.append([Loop_num,entropy_X])        
            myentropy_X = copy.copy(entropy_X)


            # ------------

            # begin to update weight matrix
            weight_X, bias_X, lrate_X, wts_blowup_X = weight_update4(weight_X, data1, bias_X, lrate_X, False)


            # %---------------------------
            # % if weight is not  blowup, update
            if not wts_blowup_X :
                
                oldwtchange_X  = copy.copy(weight_X - oldweight_X)
                STEP_X = copy.copy(STEP_X + 1)
                lrates_X[0,STEP_X] = copy.copy(lrate_X)
                angledelta_X = 0 
                delta_X = copy.copy(oldwtchange_X.reshape(1 , chans_X* self.Global_NCOM_X ,order='F'))                
                change_X = copy.copy(np.dot(delta_X,delta_X.T))
                
            # end if not wts_blowup_X
            #%DATA1 blow up restart-------------------------------
            if wts_blowup_X or np.isnan(change_X) or np.isinf(change_X) : #  % if weights blow up,
                print(' ')
                STEP_X = 0
                STOPSIGN_X = 0 #                % start again
                change_X = copy.copy(nochange)
                wts_blowup_X = 0 #                    % re-initialize variables
                blockno_X = 1
                lrate_X = copy.copy(lrate_X * DEFAULT_RESTART_FAC) #; % with lower learning rate
                weight_X  = copy.copy(startweight_X)  #            % and original weight matrix
                oldweight_X  = copy.copy(startweight_X)
                oldwtchange_X = np.zeros((chans_X,self.Global_NCOM_X))  
                delta_X = np.zeros((1,chans_X * chans_X))
                olddelta_X = copy.copy(delta_X)
                extblocks = copy.copy(urextblocks)
                prevweight_X  = copy.copy(startweight_X)
                prevwtchange_X = np.zeros((chans_X,self.Global_NCOM_X))
                bias_X = copy.copy(np.zeros((self.Global_NCOM_X, 1)))
                lrates_X = copy.copy(np.zeros((1,maxsteps)))

                # Entropy                    
                entropychange_list_X.append([Loop_num,2.0])

                if extended : 
                    signs_X = copy.copy(np.ones((1,self.Global_NCOM_X)))  #% initialize signs to nsub -1, rest +1
            
                    for k in range(1,nsub) :
                        signs_X[k] = -1
                    # end for
                    signs_X = np.diag(signs_X) # % make a diagonal matrix
                    oldsigns_X = np.zeros(signs_X.size)
                # end if extended
                if lrate_X > MIN_LRATE :
                    r =  copy.copy(matrix_rank(data1))
                    if r < self.Global_NCOM_X :
                        print('Data has rank %d. Cannot compute %d components.' %( r,self.Global_NCOM_X))
                        return
                    else : 
                        print('Lowering learning rate to %g and starting again.' %lrate_X)
                    #end if
                else :
                    print('XXXXX runica(): QUITTING - weight matrix may not be invertible! XXXXXX')
                    return
                #end if 
            else  : #% if DATA1 weights in bounds
                # %testing the trend of entropy term, avoiding the overfitting of correlation

                u = copy.copy(np.dot(weight_X , data1 [:, :]) + bias_X * np.ones((1,frames_X)))
                y = copy.copy(1/(1 + np.exp(-u)))              
                temp = copy.copy(np.log(abs( (np.dot(weight_X,y) * (1-y) ) + eps)))
                lossf_X[STEP_X] = copy.copy(np.mean(temp) )
                
                #%changes of entropy term added by jingyu
                if STEP_X > 1 :
                    entropychange_X = lossf_X[STEP_X] - entropy_X
                else :
                    entropychange_X = 1
                # Entropy                    
                entropychange_list_X.append([Loop_num,entropychange_X])               
                #end
                #%--------
                
                if STEP_X > 5 :
                    a = 1 + 1
                    index_X = copy.copy(ica_fuse_falsemaxdetect(self, lossf_X,trendPara))
                    if index_X :
                        Crate_X  = copy.copy(Crate_X*CRATE_PERCENT) #         % anneal learning rate empirical
                    # end if
                #end % end of test------------------------

                # %%%%%%%%%%%%% Print weight update information %%%%%%%%%%%%%%%%%%%%%%
                # %

                if STEP_X  > 2 and not STOPSIGN_X :
                    change_temp = copy.copy(float( np.sqrt( float(change_X.real) * float(oldchange_X.real) )))
                    delta_temp = copy.copy(np.dot(delta_X , olddelta_X.T ))
                    angledelta_X = copy.copy(math.acos(delta_temp/  change_temp    ))  # (1, 64) x (1, 64).T

                #end
                if verbose :
                    if STEP_X > 2 :
                        if not extended :
                             
                            print('Dataset X step %d - lrate %7.6f, wchange %7.6f, angledelta %4.1f deg, Crate_X %f, eX %f, eY %f, maxC %f, mc_test %f' 
                                %( STEP_X, lrate_X, change_X, degconst*angledelta_X, Crate_X, entropy_X, entropy_Y, mymaxcorr, mymaxcorr_test) )                            
                        else :

                            print('Dataset X step %d - lrate %7.6f, wchange %7.6f, angledelta %4.1f deg, %d subgauss' 
                                %( STEP_X, lrate_X, change_X, degconst*angledelta_X, (self.Global_NCOM_X - sum(np.diag(signs_X)))/2)) 
                        #end
                    elif not extended :
                        print('Dataset X step %d - lrate %7.6f, wchange %7.6f' %(STEP_X, lrate_X, float(change_X.real) ))
                    else:

                        print('Dataset X step %d - lrate %5f, wchange %7.6f, %d subgauss' 
                            %( STEP_X, lrate_X, float(change_X.real), (self.Global_NCOM_X - sum(np.diag(signs_X)))/2))
                    #end % step > 2
        
                # %%%%%%%%%%%%%%%%%%%% Anneal learning rate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # %
                if entropychange_X < 0  : #%| degconst*angledelta(1) > annealdeg,
                    lrate_X = copy.copy(lrate_X * annealstep) #          % anneal learning rate
                    olddelta_X = copy.copy(delta_X)  #                % accumulate angledelta until
                    oldchange_X  = copy.copy(change_X) #              %  annealdeg is reached
                elif STEP_X == 1 : #                     % on first step only
                    olddelta_X   = copy.copy(delta_X) #                % initialize
                    oldchange_X  = copy.copy(change_X)
                # end
                
                #%%%%%%%%%%%%%%%%%%%% Apply stopping rule %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # Apply stopping rule               
                if STEP_X  > 2 and change_X < nochange:    # apply stopping rule
                    STOPSIGN_X  = 1                    # stop when weights stabilize
                    print ("STOPSIGN_X = True")                        
                elif STEP_X  >= maxsteps :
                    STOPSIGN_X  = 1                    # max step
                    print ("STOPSIGN_X = True")                        
                elif change_X > DEFAULT_BLOWUP :       # if weights blow up,
                    lrate_X = copy.copy(lrate_X * DEFAULT_BLOWUP_FAC)    # keep trying with a smaller learning rate
                # end if
                # %%%%%%%%%%%%%%%%%% Save current values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                oldweight_X  = (weight_X)


            #end; % end if weights in bounds
        # end if ~stopsign(1)

        # Entropy
        STEP_list_X.append([Loop_num,STEP_X])
        STOPSIGN_list_X.append([Loop_num,STOPSIGN_X])
        Crate_list_X.append([Loop_num,Crate_X])       


        if not STOPSIGN_Y :
            # Global ICA - Find Global A (aka find Global W) using Infomax         
            # print('[LOG][Flow_5_Global_ICA]=====Modality_Y : Find Global A (aka find Global W) using Infomax =====')

            eps = np.finfo(np.float32).eps
            u = copy.copy(np.dot(weight_Y , data2) + np.dot( bias_Y , np.ones((1,frames_Y)) ))
            y = copy.copy(1 / (1 + np.exp (-u)))

            yy = copy.copy((1-y))                              
            temp1 = copy.copy(np.dot(weight_Y,y))              
            temp2 = copy.copy((temp1 * yy))                    
            temp = copy.copy(np.log( abs( temp2 ) + eps))      

            entropy_Y = np.mean(temp)     
            # Entropy            
            entropy_list_Y.append([Loop_num,entropy_Y]) 
            myentropy_Y = copy.copy(entropy_Y)



            # ------------

            # begin to update weight matrix
            weight_Y, bias_Y, lrate_Y, wts_blowup_Y = weight_update4(weight_Y, data2, bias_Y, lrate_Y, False)

            # %---------------------------
            # % if weight is not  blowup, update
            if not wts_blowup_Y :
                
                oldwtchange_Y  = copy.copy (weight_Y - oldweight_Y)
                STEP_Y = STEP_Y + 1
                lrates_Y[0,STEP_Y] = copy.copy (lrate_Y)
                angledelta_Y = 0 
                # delta_Y = copy.copy(oldwtchange_Y.reshape(1 , chans_Y* self.Global_NCOM_Y ) )
                delta_Y = copy.copy(oldwtchange_Y.reshape(1 , chans_Y* self.Global_NCOM_Y ,order='F'))
                change_Y = copy.copy(np.dot(delta_Y,delta_Y.T))
                
            # end if not wts_blowup_Y
            #%DATA1 blow up restart-------------------------------
            if wts_blowup_Y or np.isnan(change_Y) or np.isinf(change_Y) : #  % if weights blow up,
                print(' ')
                STEP_Y = 0
                STOPSIGN_Y = 0 #                % start again
                change_Y = copy.copy (nochange)
                wts_blowup_Y = 0 #                    % re-initialize variables
                blockno_Y = 1
                lrate_Y = copy.copy (lrate_Y * DEFAULT_RESTART_FAC) #; % with lower learning rate
                weight_Y  = copy.copy (startweight_Y)  #            % and original weight matrix
                oldweight_Y  = copy.copy (startweight_Y)
                oldwtchange_Y = np.zeros((chans_Y,self.Global_NCOM_Y))  
                delta_Y = np.zeros((1,chans_Y * chans_Y))
                olddelta_Y = copy.copy (delta_Y)
                extblocks = copy.copy (urextblocks)
                prevweight_Y  = copy.copy (startweight_Y)
                prevwtchange_Y = np.zeros((chans_Y,self.Global_NCOM_Y))
                bias_Y = np.zeros((self.Global_NCOM_Y, 1))
                lrates_Y = np.zeros((1,maxsteps))

                # Entropy                    
                entropychange_list_Y.append([Loop_num,2.0])

                if extended : 
                    signs_Y = copy.copy(np.ones((1,self.Global_NCOM_Y)))  #% initialize signs to nsub -1, rest +1
            
                    for k in range(1,nsub) :
                        signs_Y[k] = -1
                    # end for
                    signs_Y = np.diag(signs_Y) # % make a diagonal matrix
                    oldsigns_Y = np.zeros(signs_Y.size)
                # end if extended
                if lrate_Y > MIN_LRATE :
                    r =  matrix_rank(data2) 
                    if r < self.Global_NCOM_Y :
                        print('Data has rank %d. Cannot compute %d components.' %( r,self.Global_NCOM_Y))
                        return
                    else : 
                        print('Lowering learning rate to %g and starting again.' %lrate_Y)
                    #end if
                else :
                    print('XXXXX runica(): QUITTING - weight matrix may not be invertible! XXXXXX')
                    return
                #end if 
            else  : #% if DATA1 weights in bounds
                # %testing the trend of entropy term, avoiding the overfitting of correlation

                u = copy.copy(np.dot(weight_Y , data2 [:, :]) + bias_Y * np.ones((1,frames_Y)))
                y = copy.copy(1/(1 + np.exp(-u)))
                # y = 1/(1 + expit(-u))
                temp = copy.copy(np.log(abs( (np.dot(weight_Y,y) * (1-y) ) + eps)))
                lossf_Y[STEP_Y] = copy.copy(np.mean(temp) )     # MATLAB -2.05926744085665  PYTHON = -2.0592651746
                
                #%changes of entropy term added by jingyu
                if STEP_Y > 1 :
                    entropychange_Y = copy.copy(lossf_Y[STEP_Y] - entropy_Y)
                else :
                    entropychange_Y = 1
                # Entropy                    
                entropychange_list_Y.append([Loop_num,entropychange_Y])               
                #end
                #%--------
                
                if STEP_Y > 5 :
                    index_Y = copy.copy(ica_fuse_falsemaxdetect(self, lossf_Y,trendPara)) # Test Entropy deciding to reduce Change_rate
                    if index_Y :
                        # Crate_Y  = Crate_Y*0.9 #         % anneal learning rate empirical
                        Crate_Y  = copy.copy(Crate_Y*CRATE_PERCENT) #         % anneal learning rate empirical
                        # print('Dataset Y step %d - ica_fuse_falsemaxdetect index_Y %5f, Crate_Y %f, trendPara %f ' %(STEP_Y, index_Y, Crate_Y, trendPara))

                    # end if
                #end % end of test------------------------

                # %%%%%%%%%%%%% Print weight update information %%%%%%%%%%%%%%%%%%%%%%
                # %

                if STEP_Y  > 2 and not STOPSIGN_Y :

                    change_temp = copy.copy(float( np.sqrt( float(change_Y.real) * float(oldchange_Y.real) )))
                    delta_temp = copy.copy(np.dot(delta_Y , olddelta_Y.T ))
                    angledelta_Y = copy.copy(math.acos(delta_temp/  change_temp    ))  # (1, 64) x (1, 64).T

                #end
               
                if verbose :
                    if STEP_Y > 2 :
                        if not extended :
                            print('Dataset Y step %d - lrate %7.6f, wchange %7.6f, angledelta %4.1f deg, Crate_Y %f, eX %f, eY %f, maxC %f, mc_test %f' 
                                %( STEP_Y, lrate_Y, change_Y, degconst*angledelta_Y, Crate_Y, entropy_X, entropy_Y, mymaxcorr, mymaxcorr_test) )   
                        else :
                            print('Dataset Y step %d - lrate %7.6f, wchange %7.6f, angledelta %4.1f deg, %d subgauss' 
                                %( STEP_Y, lrate_Y, change_Y, degconst*angledelta_Y, (self.Global_NCOM_Y - sum(np.diag(signs_Y)))/2)) 
                        #end
                    elif not extended :
                        print('Dataset Y step %d - lrate %7.6f, wchange %7.6f' %(STEP_Y, lrate_Y, float(change_Y.real) ))
                    else:
 
                        print('Dataset Y step %d - lrate %7.6f, wchange %7.6f, %d subgauss' 
                            %( STEP_Y, lrate_Y, float(change_Y.real), (self.Global_NCOM_Y - sum(np.diag(signs_Y)))/2))
                    #end % step > 2
        
                # %%%%%%%%%%%%%%%%%%%% Anneal learning rate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # %
                if entropychange_Y < 0  : #%| degconst*angledelta(1) > annealdeg,
                    lrate_Y = copy.copy (lrate_Y * annealstep) #          % anneal learning rate
                    olddelta_Y = copy.copy (delta_Y)  #                % accumulate angledelta until
                    oldchange_Y  = copy.copy (change_Y) #              %  annealdeg is reached
                elif STEP_Y == 1 : #                     % on first step only
                    olddelta_Y   = copy.copy (delta_Y) #                % initialize
                    oldchange_Y  = copy.copy (change_Y)
                # end
                
                #%%%%%%%%%%%%%%%%%%%% Apply stopping rule %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # Apply stopping rule               
                if STEP_Y  > 2 and change_Y < nochange:    # apply stopping rule
                    STOPSIGN_Y  = 1                    # stop when weights stabilize
                    print ("STOPSIGN_Y = True")                        
                elif STEP_Y  >= maxsteps :
                    STOPSIGN_Y  = 1                    # max step
                    print ("STOPSIGN_Y = True")                        
                elif change_Y > DEFAULT_BLOWUP :       # if weights blow up,
                    lrate_Y = copy.copy (lrate_Y * DEFAULT_BLOWUP_FAC)    # keep trying with a smaller learning rate
                # end if
                # %%%%%%%%%%%%%%%%%% Save current values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                oldweight_Y  =  (weight_Y)


            #end; % end if weights in bounds
        # end if ~stopsign(2) 
        # Entropy
        STEP_list_Y.append([Loop_num,STEP_Y])
        STOPSIGN_list_Y.append([Loop_num,STOPSIGN_Y])
        Crate_list_Y.append([Loop_num,Crate_Y])       


        GlobalW_unmixer_X =  (weight_X)
        GlobalA_mixer_X  =  (inv(GlobalW_unmixer_X))
        GlobalW_unmixer_Y =  (weight_Y)
        GlobalA_mixer_Y  = (inv(GlobalW_unmixer_Y)   )
  

        # print('[LOG][Flow_5_Global_ICA]=====End=====')   

        # Check_I_X = np.dot(self.GlobalPCA_White_X, self.GlobalPCA_dewhite_X )
        # print('[LOG][Flow_5_Global_ICA]===== Check_I_X=====')   

        

        #
        # Parrelle ICA - Correlation A
        #
        # print('[LOG][Flow_7_Parallel_ICA-Correlation_A]=====Start=====')   

        # Parallel ICA - Find Correlation local A       

        # print('[LOG][Flow_7_Parallel_ICA-Correlation_A]=====Modality_X and _Y : Find Correlation A====='

        # Parrelle ICA start
        # Parrelle ICA - Local reconstruction

        # % -------------------------
        # % modifying weights based on correlation between data1 A Matrix and data2 A Matrix
        # % %%%%%%%%%%%%%%%%%nudging

        if (STEP_X >2 and STEP_Y > 2) and ( not STOPSIGN_X or not STOPSIGN_Y ) :

            #
            # print('[LOG][Flow_6_Parallel_ICA-Local_reconstruction]=====Start=====')   

            # Parallel ICA - Find local A using GlobalPCA_dewhite_X and Con_deWhite_X    
            # Called Local-reconstruction ; from-Global-to-Local     
            # print('[LOG][Flow_6_Parallel_ICA-Local_reconstruction]=====Modality_X : Find local A using GlobalPCA_dewhite_X and Con_deWhite_X   =====')
            LocalA_All_X = (self.local_reconstruction9(GlobalW_unmixer_X, GlobalA_mixer_X, \
                self.GlobalPCA_dewhite_X, self.Con_deWhite_X, self.NSUB_ALL_X, self.NCOM_ALL_X, SITE_NUM, B_GLOBAL, False))      

 
            # print('[LOG][Flow_6_Parallel_ICA-Local_reconstruction]=====Modality_Y : Find local A using GlobalPCA_dewhite_Y and Con_deWhite_Y   =====')            
            LocalA_All_Y = (self.local_reconstruction9(GlobalW_unmixer_Y, GlobalA_mixer_Y, \
                self.GlobalPCA_dewhite_Y, self.Con_deWhite_Y, self.NSUB_ALL_Y, self.NCOM_ALL_Y, SITE_NUM, B_GLOBAL, False))                  



            # print('[LOG][Flow_7_Parallel_ICA-Correlation_A]=====Start=====')   

            # Parallel ICA - Find Correlation local A       

            # print('[LOG][Flow_7_Parallel_ICA-Correlation_A]=====Modality_X and _Y : Find Correlation A=====')


            mx = copy.copy (LocalA_All_X) 
            sx = copy.copy (LocalA_All_Y)  
            Corr_matrix = copy.copy(np.abs(ica_fuse_corr(self, mx,sx)))  # 8 x 8
            #  % calculate the correlation of all componentsts   % match tow modality components
            
            j_min = copy.copy(min(int(self.Global_NCOM_X),int(self.Global_NCOM_Y)))
            maxcorr=np.zeros(j_min)
            maxcol=np.zeros(j_min)
            maxrow=np.zeros(j_min)

            for j in range(j_min)  :
                [mm,ss] = copy.copy(np.unravel_index(np.argmax(Corr_matrix, axis=None), Corr_matrix.shape))
                maxcol[j] = copy.copy(mm)      
                maxrow[j] = copy.copy(ss)      
                maxcorr[j] = copy.copy(Corr_matrix[mm,ss])        
                Corr_matrix[mm,:]=0
                Corr_matrix[:,ss]=0

            costfunction_corrcoef_list_maxcorr_XY.append([Loop_num, maxcorr[0]])
            mymaxcorr =  copy.copy (maxcorr[0])


            # Fix corr
            ix = copy.copy(np.array(np.where (maxcorr > Connect_threshold))) #ix = find(temp>Connect_threshold)     # Connect_threshold 0.2542 ix = 1x0


            if not (np.size(ix)==0) :  # ~isempty(ix) :
                if (np.size(ix)) > MaxComCon :
                    ix = copy.copy(np.resize(ix,(1,MaxComCon)) )                               
                
                # print('[LOG] np.size(ix)   =  ', np.size(ix) )  
                
                # If not empty, do here      
                a =[]
                a_X = []
                a_Y = []
                u = []
                u_X = []
                u_Y = []
                for Cons_com in range(np.size(ix)) : # % constraned componenets
                    # print('[LOG]  Cons_com    =  ',  Cons_com )    

                    #% Updata the weights
                    
                    a = copy.copy (mx[:,int(maxcol[Cons_com])])
                    a_X = copy.copy(a)
                    a_Y = copy.copy(a)
                    u = copy.copy (sx[:,int(maxrow[Cons_com])])
                    u_X = copy.copy(u)
                    u_Y = copy.copy(u)

                    b1 = copy.copy(np.cov(a,u))  
                    b = b1[0,1]     

                    tmcorr = copy.copy(b/np.std(a, ddof=1) / np.std(u, ddof=1))  
                    comterm = copy.copy(2*b/np.var(a,ddof=1)/np.var(u,ddof=1))   

                    coml = len(a)
                    
                    if not STOPSIGN_X : # %& ~Overindex1
                        deltaf_X = copy.copy(comterm * ( u_X - np.mean(u_X) + (b * (np.mean(a_X)-a_X)  /  (np.var(a_X,ddof=1)) ) )) # 1st order derivative 
                        P_X = copy.copy(deltaf_X / np.linalg.norm(deltaf_X))     
                        alphak_X = copy.copy(findsteplength1 (-tmcorr**2, -deltaf_X, a_X, u_X, alphak_X, P_X, 0.0001, 0.999)) 
                        aweights_X = copy.copy(Crate_X * alphak_X * P_X)   
                        mx[:,int(maxcol[Cons_com])] = copy.copy (mx[:, int(maxcol[Cons_com])] + aweights_X)
                    # end if not STOPSIGN_X 

                    if not STOPSIGN_Y : # not Overindex1 
                        deltaf_Y = copy.copy((comterm * (a_Y - np.mean(a_Y) + b/np.var(u,ddof=1)*(np.mean(u_Y) - u_Y))))   # 1st order derivative         
                        P_Y = copy.copy(deltaf_Y / np.linalg.norm(deltaf_Y))      # (H2*deltaf')'
                        alphak_Y = copy.copy(findsteplength1 (-tmcorr**2, -deltaf_Y, u_Y, a_Y, alphak_Y, P_Y, 0.0001, 0.999))
                        aweights_Y = copy.copy(Crate_Y * alphak_Y * P_Y)
                        sx[:,int(maxrow[Cons_com])] = copy.copy (sx[:,int(maxrow[Cons_com])] + aweights_Y)
                    # end if not STOPSIGN_Y     


                # end for Cons_com 

                #
                # Parrelle ICA - Global reconstruction
                #
                # print('[LOG][Flow_8_Parallel_ICA-Global_reconstruction]=====Start=====')   

                if not STOPSIGN_X :
                    temp = copy.copy (weight_X )

                    # Parallel ICA - Find Global A (Flow#5) using Con_White_X and GlobalPCA_White_X 
                    # Called Global-reconstruction ; from-Local-to-Global     
                    # print('[LOG][Flow_8_Parallel_ICA-Global_reconstruction]=====Modality_X : Find Global A (Flow#5) using Con_White_X and GlobalPCA_White_X====')

                    GlobalA_mixer_X = (self.global_reconstruction9(mx, 
                        self.GlobalPCA_White_X, self.Con_White_X, self.NSUB_ALL_X, self.NCOM_ALL_X, SITE_NUM, B_GLOBAL)  )


                    # Print all Modality Correlation A shape
                    # print('[LOG][Flow_8_Parallel_ICA-Global_reconstruction]=====Print all Modality Global A shape=====')
                    # print('Modality_X === GlobalA_mixer_X (r_X x r_X)',GlobalA_mixer_X.shape )


                    #
                    # Parrelle ICA - Global Weight update
                    #
                    # print('[LOG][Flow_8a_Parallel_ICA-Global_Weight_update]=====Start=====')   

                    # Parallel ICA - Update Global Weight of all Modality from Global A mixer
                    # print('[LOG][Flow_8a_Parallel_ICA-Global_Weight_update]=====Update Global Weight of all Modality from Global A mixer=====')
                    weight_X = copy.copy (inv(GlobalA_mixer_X)) #% weights{1} \ eye(size(weights{1}));


                    if np.amax(abs(weight_X)) > MAX_WEIGHT :
                        Crate_X = Crate_X *0.95
                        weight_X = temp
                        print ('weight_X > MAX_WEIGHT !!!! Crate_X')                              
                    # end if
                # end if not STOPSIGN_X :

                if not STOPSIGN_Y :
                    temp = copy.copy(weight_Y )

                    # Parallel ICA - Find Global A (Flow#5) using Con_White_X and GlobalPCA_White_X 
                    # Called Global-reconstruction ; from-Local-to-Global  
                    # print('[LOG][Flow_8_Parallel_ICA-Global_reconstruction]=====Modality_Y : Find Global A (Flow#5) using Con_White_Y and GlobalPCA_White_Y====')

                    GlobalA_mixer_Y = (self.global_reconstruction9(sx, 
                        self.GlobalPCA_White_Y, self.Con_White_Y, self.NSUB_ALL_Y, self.NCOM_ALL_Y, SITE_NUM, B_GLOBAL) )


                    # Print all Modality Correlation A shape
                    # print('[LOG][Flow_8_Parallel_ICA-Global_reconstruction]=====Print all Modality Global A shape=====')
                    # print('Modality_Y === GlobalA_mixer_Y (r_Y x r_Y)',GlobalA_mixer_Y.shape )

                    # print('[LOG][Flow_8_Parallel_ICA-Global_reconstruction]=====End=====')   

                    #
                    # Parrelle ICA - Global Weight update
                    #
                    # print('[LOG][Flow_8a_Parallel_ICA-Global_Weight_update]=====Start=====')   

                    # Parallel ICA - Update Global Weight of all Modality from Global A mixer
                    # print('[LOG][Flow_8a_Parallel_ICA-Global_Weight_update]=====Update Global Weight of all Modality from Global A mixer=====')
                    weight_Y = copy.copy(inv(GlobalA_mixer_Y)) #% weights{1} \ eye(size(weights{1}));


                    if np.amax(abs(weight_Y)) > MAX_WEIGHT :
                        Crate_Y = Crate_Y *0.95
                        weight_Y = copy.copy(temp)
                        print ('weight_Y > MAX_WEIGHT !!!! Crate_Y')                        
                    # end if
                # end if not STOPSIGN_Y :

                # Test - Validation of Weight 
                #%             test ---------------------
                # This is for testing and validating if Weight is in Wrong direction or NOT.

                LocalA_All_X_test = copy.copy(self.local_reconstruction9(weight_X, GlobalA_mixer_X, \
                    self.GlobalPCA_dewhite_X, self.Con_deWhite_X, self.NSUB_ALL_X, self.NCOM_ALL_X, SITE_NUM, B_GLOBAL, False))

                LocalA_All_Y_test = copy.copy(self.local_reconstruction9(weight_Y, GlobalA_mixer_Y, \
                    self.GlobalPCA_dewhite_Y, self.Con_deWhite_Y, self.NSUB_ALL_Y, self.NCOM_ALL_Y, SITE_NUM, B_GLOBAL, False))

                mx_test = copy.copy (LocalA_All_X_test)   
                sx_test = copy.copy (LocalA_All_Y_test)  
                Corr_matrix_test = np.abs(ica_fuse_corr(self, mx_test,sx_test))  

                #  % calculate the correlation of all componentsts   % match tow modality components
                j_min_test = copy.copy(min(int(self.NCOM_X),int(self.NCOM_Y)))
                maxcorr_test =np.zeros(j_min_test)
                maxcol_test =np.zeros(j_min_test)
                maxrow_test =np.zeros(j_min_test)

                for j in range(j_min)  :
                    [mm,ss] = copy.copy(np.unravel_index(np.argmax(Corr_matrix_test, axis=None), Corr_matrix_test.shape))
                    maxcol_test[j] = copy.copy(mm)      
                    maxrow_test[j] = copy.copy(ss)      
                    maxcorr_test[j] = copy.copy(Corr_matrix_test[mm,ss])    
                    Corr_matrix_test[mm,:]=0
                    Corr_matrix_test[:,ss]=0
                mymaxcorr_test =  maxcorr_test[0]

                costfunction_corrcoef_list_maxcorr_test_XY.append([Loop_num, maxcorr_test[0]])


                if maxcorr_test[0] < maxcorr[0] : 
                    print ('Wrong direction !!!! ')
                    print ('/')                  
                    
                # end if

                #%             -----------------end test
                oldweight_Y = copy.copy (weight_Y) # 8x8
                oldweight_X = copy.copy (weight_X) # 8x8
                # print('[LOG][Flow_8a_Parallel_ICA-Global_Weight_update] Testing and validating ===== Finish =====')

            #end if ~ isempty(ix) :


        # Entropy
        else :
            costfunction_corrcoef_list_maxcorr_XY.append([Loop_num, 0])
            costfunction_corrcoef_list_maxcorr_test_XY.append([Loop_num,0])  

        # end if if (STEP_X >2 and STEP_Y > 2) and ( not STOPSIGN_X or not STOPSIGN_Y ) 


        if STOPSIGN_X == 1 and STOPSIGN_Y == 1 :
            laststep = max(STEP_X, STEP_Y)
            STEP_LAST_X = STEP_X
            STEP_LAST_Y = STEP_Y
            STEP_X = maxsteps                #% stop when weights stabilize
            STEP_Y = maxsteps
        # end if 
        

        #
        # Parrelle ICA - Global Weight update
        #
        # print('[LOG][Flow_8a_Parallel_ICA-Global_Weight_update]=====Start=====')   

        # Parallel ICA - Update Global Weight of all Modality from Global A mixer
        # print('[LOG][Flow_8a_Parallel_ICA-Global_Weight_update]=====Update Global Weight of all Modality from Global A mixer=====')

        GlobalW_unmixer_X = copy.copy (weight_X)
        GlobalW_unmixer_Y = copy.copy (weight_Y)

        # print('[LOG][Flow_8a_Parallel_ICA-Global_Weight_update]=====End=====')   
    
        Loop_num = Loop_num + 1

    # End of IF Flow 6 - Flow 8a


    # End while (STEP_X < maxsteps or STEP_Y < maxsteps) :
    ###########################################################################

    # %%%%%%%%%%%%%% Orient components towards positive activation %%%%%%%%%%%
    # %

    GlobalW_unmixer_X = copy.copy (weight_X)
    GlobalW_unmixer_Y = copy.copy (weight_Y)


    # Save all list into files.
    import datetime
    today = datetime.datetime.today() 
    YYYYMMDD = today.strftime('%Y%m%d%H%M')
    np.savetxt( DATA_PATH_OUTPUT  + "Loop_list" + "_" + YYYYMMDD +".csv", Loop_list, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "Loop_list_X" + "_" + YYYYMMDD +".csv", Loop_list_X, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "Loop_list_Y" + "_" + YYYYMMDD +".csv", Loop_list_Y, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "STEP_list_X" + "_" + YYYYMMDD +".csv", STEP_list_X, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "STEP_list_Y" + "_" + YYYYMMDD +".csv", STEP_list_Y, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "STOPSIGN_list_X" + "_" + YYYYMMDD +".csv", STOPSIGN_list_X, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "STOPSIGN_list_Y" + "_" + YYYYMMDD +".csv", STOPSIGN_list_Y, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "Crate_list_X" + "_" + YYYYMMDD +".csv", Crate_list_X, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "Crate_list_Y" + "_" + YYYYMMDD +".csv", Crate_list_Y, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "entropy_list_X" + "_" + YYYYMMDD +".csv", entropy_list_X, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "entropy_list_Y" + "_" + YYYYMMDD +".csv", entropy_list_Y, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "entropychange_list_X" + "_" + YYYYMMDD +".csv", entropychange_list_X, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "entropychange_list_Y" + "_" + YYYYMMDD +".csv", entropychange_list_Y, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "costfunction_corrcoef_list_maxcorr_XY" + "_" + YYYYMMDD +".csv", costfunction_corrcoef_list_maxcorr_XY, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "costfunction_corrcoef_list_maxcorr_test_XY" + "_" + YYYYMMDD +".csv", costfunction_corrcoef_list_maxcorr_test_XY, delimiter=",")

    self.mymaxcorr_list.append([self.RUN_NUMBER, round(mymaxcorr,4)])   
    self.myentropy_list_X.append([self.RUN_NUMBER, round(myentropy_X,4)])   
    self.myentropy_list_Y.append([self.RUN_NUMBER, round(myentropy_Y,4)])   
    self.mySTEP_list_X.append([self.RUN_NUMBER, STEP_LAST_X])   
    self.mySTEP_list_Y.append([self.RUN_NUMBER, STEP_LAST_Y])   
    self.mymaxcol_list.append([self.RUN_NUMBER, int(maxcol[0])])   
    self.mymaxrow_list.append([self.RUN_NUMBER, int(maxrow[0])])    
    self.RUN_NUMBER = self.RUN_NUMBER + 1

    if ((change_X  > nochange) or (change_Y > nochange)) : 
        if ( (STEP_X) == (maxsteps)  or (STEP_Y) == (maxsteps) ):
            print('!!!Reached max steps. Please reduce the constrained components in setup options and restart parallel ica.')
        # end if
    # end if


    print("[LOG]=====INFOMAX=====Current_STEP_X = ", STEP_LAST_X  , " and STEP_Y  =" , STEP_LAST_Y ,\
        "STOPSIGN_X = ", STOPSIGN_X , " STOPSIGN_Y  = ", STOPSIGN_Y )     

    # print('INFOMAX...Return====A,S,W,STEP,STOPSIGN as inv(weights), dot(weights, x_white), weights, STEP, STOPSIGN.===')     
    print("[LOG]=====INFOMAX=====Finish")        

    print('[LOG][Flow_5_Global_ICA]')
    print('[LOG][Flow_6_Parallel_ICA-Local_reconstruction]')   
    print('[LOG][Flow_7_Parallel_ICA-Correlation_A]')           
    print('[LOG][Flow_8_Parallel_ICA-Global_reconstruction]')
    print('[LOG][Flow_8a_Parallel_ICA-Global_Weight_update]') 




    np.savetxt( DATA_PATH_OUTPUT  + "GlobalW_unmixer_X_" + str(self.run) + ".csv", GlobalW_unmixer_X, delimiter=",")
    np.savetxt( DATA_PATH_OUTPUT  + "GlobalW_unmixer_Y_" + str(self.run) + ".csv", GlobalW_unmixer_Y, delimiter=",")

    print('[LOG][Flow_8a_Parallel_ICA-Global] Save all A, W, S cvs files.') 
    print('[LOG][Flow_8a_Parallel_ICA-Global] Finish run ', str(self.run), ".") 
    print('DATA_PATH_OUTPUT = ', DATA_PATH_OUTPUT)
    print('maxcorr = ', self.mymaxcorr_list)
    print('max_pair_X = ', self.mymaxcol_list)
    print('max_pair_Y = ', self.mymaxrow_list)       
    print('ENTROPY_X =  ' , self.myentropy_list_X)
    print('ENTROPY_Y =  ' , self.myentropy_list_Y)
    print('STEP_X =  ' , self.mySTEP_list_X)
    print('STEP_Y =  ' , self.mySTEP_list_Y)
    print('======================================')



    return (self,  GlobalW_unmixer_X, sphere_X, \
                   GlobalW_unmixer_Y, sphere_Y )


if __name__ == '__main__':
    print('main')
    unittest.main()
    print('End of main')
    print('===')