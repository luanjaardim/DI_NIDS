import KitNET as kit
import numpy as np
import pandas as pd
import time
from utils import get_ds
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras import Model
from keras.optimizers import Adam
from utils import get_day_set, get_ds

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
X, y, _, _ = get_day_set("monday", root="./data/", num_files=np.inf)
train_ds_size = X.shape[0]

# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = train_ds_size // 10 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = train_ds_size - FMgrace #the number of instances used to train the anomaly detector (ensemble itself)

# Build KitNET
K = kit.KitNET(X.shape[1],maxAE,FMgrace,ADgrace)

print("Running KitNET:")
start = time.time()
# Here we process (train/execute) each individual observation.
# In this way, X is essentially a stream, and each observation is discarded after performing process() method.
for i in range(train_ds_size):
    if i % 1000 == 0:
        print(i)
    K.process(X[i,]) #will train during the grace periods, then execute on all the rest.
stop = time.time()
print("Training Complete. Time elapsed: "+ str(stop - start))

### At this threshold the FPR of the model we trained is 0.1
thr = 0.07062649726867676

attack_to_label={
    'FTP-Patator':1,'SSH-Patator':2, #Tuesday attacks
    'Slowloris':3,'Slowhttptest':4,'Hulk':5,'GoldenEye':6,'Heartbleed':7, #Wednesday attacks
    'Web-Attack':8, 'Infiltration':9, #Thursday attacks
    'Botnet':10,'PortScan':11,'DDoS':12 #Friday attacks
}
print("com 1000")
timesteps = 20
attack_types = ['SSH-Patator','Slowloris','Slowhttptest','Hulk','GoldenEye','Heartbleed', 'Web-Attack', 'Infiltration','Botnet','PortScan','DDoS']

for attack_type in attack_types:

    @tf.function
    def test_step(x):
        # NOTE: removing aplication of partial visualization
        #
        # def_mask = tf.random.uniform(shape=[1*100,timesteps,num_input])
        # def_mask = tf.cast((def_mask>0.75),tf.float32)
        # x_normalized =(x - train_min)/(train_max - train_min+0.000001)
        # partial_x = def_mask*x_normalized
        # rec_x = model(partial_x, training=False)

        # rec_x = K.process(x)
        # score = tf.reduce_mean(tf.square(rec_x - x),axis=[1,2])
        # score = tf.reduce_min(tf.reshape(score,[5,20]),axis=-1)
        # score = tf.reduce_sum(score)

        return K.process(x)

    # Crafiting adversarial Examples:
    @tf.function
    def get_delayed_splited(x,p_len):
        
        alpha_split_2 = tf.zeros((1,1,num_input)) + alpha_split
        alpha_split_2 = tf.minimum(alpha_split_2,0.)
        alpha_split_2 = tf.maximum(alpha_split_2,-p_len+np.float32(61))
        alpha_delay_2 = tf.maximum(alpha_delay,0.)
        alpha_delay_2 = tf.minimum(alpha_delay_2,15.)
        mask = np.ones((1,1,29))
        masked_alpha = alpha_delay_2*mask
        mask_split = np.zeros((1,1,29))
        mask_split[0,0,1] = mask_split[0,0,3] = 1
        mask_split = mask_split.astype(np.bool)
        alpha_final = tf.where(mask_split,alpha_split_2,masked_alpha)

        last_ts_modified = x[0,-1]+alpha_final
        adv_x = tf.concat((x[:,:19],last_ts_modified),axis=1)
        return adv_x

    @tf.function
    def delay_split_optim(x,p_len):

        with tf.GradientTape() as tape:
            adv_x = get_delayed_splited(x,p_len)
            # adv_x_normalized = (adv_x- train_min)/(train_max - train_min+0.000001)
            # rand_mask = tf.random.uniform(shape=[100,timesteps,num_input])
            # rand_mask = tf.cast((rand_mask>0.75),tf.float32)
            # partial_adv_x_n = adv_x_normalized*rand_mask
            # rec_adv_x_n = model(partial_adv_x_n,training=False)
            # score1_split = tf.reduce_mean(tf.square(rec_adv_x_n - adv_x_normalized),axis=[1,2])
            # score1_split = tf.reduce_sum(score1_split)
            # loss_split = score1_split
            loss_split = K.process(adv_x)

        gradients = tape.gradient(loss_split, [alpha_delay,alpha_split])
        optimizer.apply_gradients(zip(gradients, [alpha_delay,alpha_split]))


    @tf.function
    def get_injected(x,inject_mask):

        packet_mins = [0,60,20,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1]
        packet_maxs = [15,2**16,20,2**16,1,1,1,255,2**16,2**16,2**32,2**32,1,1,1,1,1,1,1,1,1,1,2**16,2**16,2**16,2**16,2**16,0,1]
        alpha_inject_2 = tf.minimum(alpha_inject,packet_maxs)
        alpha_inject_2 = tf.maximum(alpha_inject_2,packet_mins)
        alpha_inject_masked= alpha_inject_2*inject_mask
        adv_x1 = tf.concat((x[:,:19],alpha_inject_masked),axis=1)
        adv_x2 = tf.concat((x[:,1:19],alpha_inject_masked,x[:,19:]),axis=1)
        return adv_x1,adv_x2


    @tf.function
    def inject_optim(x,inject_mask):
        
        with tf.GradientTape() as tape:
            adv_x1,adv_x2 = get_injected(x,inject_mask)
            # adv_x1_normalized = (adv_x1- train_min)/(train_max - train_min+0.000001)
            # adv_x2_normalized = (adv_x2- train_min)/(train_max - train_min+0.000001)
            # rand_mask = tf.random.uniform(shape=[100,timesteps,num_input])
            # rand_mask = tf.cast((rand_mask>0.75),tf.float32)
            # partial_adv_x_n1 = adv_x1_normalized*rand_mask
            # rand_mask = tf.random.uniform(shape=[100,timesteps,num_input])
            # rand_mask = tf.cast((rand_mask>0.75),tf.float32)
            # partial_adv_x_n2 = adv_x2_normalized*rand_mask
            # rec_adv_x_n1 = model(partial_adv_x_n1,training=False)
            # rec_adv_x_n2 = model(partial_adv_x_n2,training=False)
            # score1_inject1 = tf.reduce_mean(tf.square(rec_adv_x_n1 - adv_x1_normalized),axis=[1,2])
            # score1_inject1 = tf.reduce_sum(score1_inject1)
            # score1_inject2 = tf.reduce_mean(tf.square(rec_adv_x_n2 - adv_x2_normalized),axis=[1,2])
            # score1_inject2 = tf.reduce_sum(score1_inject2)
            score1_inject1 = K.process(adv_x1)
            score1_inject2 = K.process(adv_x2)
            loss_inject = score1_inject1 + score1_inject2

        gradients = tape.gradient(loss_inject, [alpha_inject])
        optimizer.apply_gradients(zip(gradients, [alpha_inject]))


    def find_adv(x):

        sc = test_step(x)
        sc = sc.numpy()
        if sc<thr:
            return 'cons_as_ben'
        if x[0,-1,-1]==2: #packet is sent from victim
            return inject(np.copy(x))

        #packet is sent from attacker
        res = delay_and_split(np.copy(x))
        if res and len(res)>0:
            return ('split',res)
        i_res = inject(np.copy(x))
        return i_res

    def delay_and_split(x):

        alpha_delay.assign(np.zeros(alpha_delay.shape))
        alpha_split.assign(np.zeros(alpha_split.shape))
        len_last = x[0,-1,1]
        ip_len_last = x[0,-1,3]
        adv_x = get_delayed_splited(x,len_last)
        adv_x = adv_x.numpy()
        sc = test_step(adv_x)
        sc = sc.numpy()
        if sc<thr:
            return [adv_x]
        res = []
        for i in range(300):
            delay_split_optim(x,len_last)
            adv_x = get_delayed_splited(x,len_last)
            adv_x = adv_x.numpy()
            adv_x[0,-1,1:4] = np.round(adv_x[0,-1,1:4])
            sc = test_step(adv_x)
            sc = sc.numpy()
            if sc<thr:
                first_part = np.copy(adv_x)
                diff = len_last - adv_x[0,-1,1]
                if diff>0:
                    adv_x[0,-1,1] = diff + 60
                    adv_x[0,-1,0] = 0
                    adv_x[0,-1,3] = diff + 60 - 14 #14 is the frame header len.
                    second_part = delay_and_split(adv_x)
                    if second_part==None:
                        return None
                    res.append(first_part)
                    res.extend(second_part)
                else:
                    res.append(first_part)
                break
        if len(res)==0:
            return None
        return res


    tcp_mask = [1]*8 + [1]*16 + [0]*3 + [0]*1 + [1]*1
    udp_mask = [1]*8 + [0]*16 + [1]*3 + [0]*1 + [1]*16
    def inject(x,mask_type = 'tcp'):
        alpha_inject.assign(np.zeros(alpha_inject.shape))
        cur_mask = tcp_mask if mask_type=='tcp' else udp_mask
        for i in range(300):
            inject_optim(x,cur_mask)
            adv_x1,adv_x2 = get_injected(x,cur_mask)
            adv_x1,adv_x2  = adv_x1.numpy(),adv_x2.numpy()
            adv_x1[0,:,1:] = np.round(adv_x1[0,:,1:])
            sc = test_step(adv_x1)
            sc = sc.numpy()
            adv_x2[0,:,1:] = np.round(adv_x2[0,:,1:])
            sc2 = test_step(adv_x2)
            sc2 = sc2.numpy()
            if sc<thr and sc2<thr: #fooled
                fake_packets.append(adv_x1[0,-1])
                return ('inject',adv_x1[0,-1]) #<--- the packet which is inject should be returned
        res = None
        return res


    # TODO: kitsune model cannot be imported
    # model = tf.keras.models.load_model('../models/pkt_model/')

    if attack_type in ['FTP-Patator','SSH-Patator']:
        day = 'tuesday'
    elif attack_type in ['Slowloris','Slowhttptest','Hulk','GoldenEye','Heartbleed']:
        day = 'wednesday'
    elif attack_type in ['Web-Attack', 'Infiltration']:
        day = 'thursday'
    else:
        day = 'friday'

    x_test,y_test,train_min,train_max = get_day_set(day, num_files=np.inf, root="./data")
    num_input = x_test.shape[1]
    print(x_test.shape,y_test.shape)

    alpha_delay = tf.Variable(np.zeros((1, 1,num_input),dtype=np.float32),name='delay')
    alpha_split = tf.Variable(np.zeros((1),dtype=np.float32),name='split')
    alpha_inject = tf.Variable(np.zeros((1, 1,num_input), dtype=np.float32),name='modifier')

    optimizer = Adam(learning_rate=0.1)

    attack_label = attack_to_label[attack_type]
    x_test_mal = x_test[y_test==attack_label]
    x_test_mal = np.concatenate((np.zeros((timesteps-1,num_input)),x_test_mal),axis=0)
    print (x_test_mal.shape)
    x_test_mal = x_test_mal[:1000].astype(np.float32)
    score_np = np.zeros(len(x_test_mal))
    st = timesteps-1
    begin_time = time.time()
    for i in range(len(x_test_mal)-timesteps):
        sample = x_test_mal[i:i+timesteps][None]
        score_temp = test_step(sample)
        score_np[st+i] = score_temp.numpy()

    mal_scores = score_np[timesteps:]
    print ("TPR in normal setting for "+attack_type+" is {0:0.4f}".format(np.sum(mal_scores>=thr)/len(mal_scores)))


    stream = []
    stream_status = []
    fake_packets = []
    cons_as_mal = 0
    cons_as_ben = 0
    fooled = 0
    x = []
    begin_time = time.time()
    for i in range(timesteps-1):
        stream.append(x_test_mal[i])
        stream_status.append(None)
    for i in range(timesteps-1,len(x_test_mal)):
        x = np.zeros((1,20,29),dtype=np.float32)
        x[0,:19] = np.array(stream[-19:])
        x[0,19] = x_test_mal[i]
        temp = find_adv(np.copy(x))
        stream_status.append(temp)
        if isinstance(temp,type(None)):
            stream.append(x_test_mal[i])
            cons_as_mal+=1
        elif temp == 'cons_as_ben':
            stream.append(x_test_mal[i])
            cons_as_ben+=1
        elif temp[0]=='split':
            fooled+=1
            for pkt in temp[1]:
                p2 = pkt[0,-1]
                stream.append(p2)
        elif temp[0]=='inject':
            fooled+=1
            fake_pkt = temp[1]
            stream.append(fake_pkt)
            stream.append(x_test_mal[i])
    print ('duration:',time.time() - begin_time)


    print ("TPR in adversarial setting for "+attack_type+" is {0:0.4f}".format(cons_as_mal/len(x_test_mal)))
