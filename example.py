import KitNET as kit
import numpy as np
import pandas as pd
import time
from utils import get_ds

##############################################################################
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates KitNET's ability to incrementally learn, and detect anomalies.
# The demo involves an m-by-n dataset with n=115 dimensions (features), and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of incremental damped statistics (see the NDSS paper for more details)

#The runtimes presented in the paper, are based on the C++ implimentation (roughly 100x faster than the python implimentation)
###################  Last Tested with Anaconda 2.7.14   #######################

# Load sample dataset (a recording of the Mirai botnet malware being activated)
# The first 70,000 observations are clean...
# print("Unzipping Sample Dataset...")
# import zipfile
# with zipfile.ZipFile("dataset.zip","r") as zip_ref:
#     zip_ref.extractall()

print("Reading Sample dataset...")
# X = pd.read_csv("mirai3.csv", "../DI_RePO/cic_dataset/thursday/part_00000.npy").to_numpy() #an m-by-n dataset with m observations
X, y, train_ds_size = get_ds("./data/")

# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = train_ds_size // 10 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = train_ds_size - FMgrace #the number of instances used to train the anomaly detector (ensemble itself)

# Build KitNET
K = kit.KitNET(X.shape[1],maxAE,FMgrace,ADgrace)
RMSEs = np.zeros(X.shape[0]) # a place to save the scores

print("Running KitNET:")
start = time.time()
# Here we process (train/execute) each individual observation.
# In this way, X is essentially a stream, and each observation is discarded after performing process() method.
for i in range(X.shape[0]):
    if i % 1000 == 0:
        print(i)
    RMSEs[i] = K.process(X[i,]) #will train during the grace periods, then execute on all the rest.
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))

real_labels = y!=-1
all_scores = RMSEs[FMgrace+ADgrace:][real_labels]
all_labels = y[real_labels]

fpr = 0.01
benign_scores_sorted = np.sort(all_scores[all_labels==0])
thr_ind = benign_scores_sorted.shape[0]*fpr
thr_ind = int(np.round(thr_ind))
thr = benign_scores_sorted[-thr_ind]
print("Threshold: ", thr)

label_names = ['Benign','FTP-Patator','SSH-Patator','Slowloris','Slowhttptest','Hulk','GoldenEye','Heartbleed', 'Web-Attack', 'Infiltration','Botnet','PortScan','DDoS']

for i in range(len(label_names)):
    #### Exclude web attacks from results
    if label_names[i]=='Web-Attack':
        continue
    scores = all_scores[all_labels==i]
    if i==0:
        fpr = "{0:0.4f}".format(np.sum(scores>=thr)/(0. + len(scores)))
        print('FPR:',fpr)
    else:
        tpr = "{0:0.4f}".format(np.sum(scores>=thr)/(0. + len(scores)))
        print(label_names[i]+':',tpr)
        print(len(scores))
        print(thr)
