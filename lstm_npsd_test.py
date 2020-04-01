########################################################################
#
# This is the test code for LSTM-based noise PSD estimation method
#
# The methods are described in the paper:
#
# - Xiaofei Li, Simon Leglaive, Laurent Girin and Radu Horaud. Audio-noise Power Spectral Density Estimation Using Long Short-term Memory. IEEE Signal Processing Letters, 2019.
#
# Author: Xiaofei Li, INRIA Grenoble Rhone-Alpes
# Copyright: Perception Team, INRIA Grenoble Rhone-Alpes
#
########################################################################


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import wave,struct,os

import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM,TimeDistributed,Bidirectional

# lstm network
flen = 3
lstm_output_size1 = 256 
lstm_output_size2 = 128
model = Sequential()
model.add(LSTM(lstm_output_size1,input_shape=(None,flen),return_sequences=True))
model.add(LSTM(lstm_output_size2,return_sequences=True))
model.add(TimeDistributed(Dense(1)))
#model.load_weights('model-lstm-9.hdf5')
model.load_weights('model-lstm-12.hdf5')
model.summary()

# stft parameters
fs = 16000
ftLen = 512
ftOlp = ftLen/2
fre_num = int(ftLen/2)+1

# smoothing factor for true NPSD calculation
psdhis = 20
lamb = float(psdhis-1)/float(psdhis+1)

# lstm prediction setting
num_steps = 128 
stp = int(num_steps*6/8)
enp = int(num_steps*8/8)
skp = enp-stp

noiseType = ['white', 'buccaneer1', 'f16', 'hfchannel', 'factory1', 'destroyerengine', 'destroyerops', 'm109', 'babble','pink','buccaneer2','factory2']
SNR = [0,5,10,15]

#speech signal, randomly selected from TIMIT test set, 120 s
sh=wave.open('speech.wav','r')
sl = sh.getnframes()
speech = sh.readframes(sl)
speech = struct.unpack('{n}h'.format(n=sl),speech)
sh.close()
speech = np.asarray(speech)/float(np.abs(speech).max())
speechPower = np.square(speech).sum()
f, t, speechstft = signal.stft(speech,nperseg=ftLen,noverlap=ftOlp)
fra_num = speechstft.shape[1]


LogErr = np.zeros((len(noiseType),len(SNR)))

for noiseindx in range(len(noiseType)):
  ntype = noiseType[noiseindx]
  print 'noise type: '+ntype

  # noise signal, the last 60 s of NOISEX92 signal, repeated twice
  nh = wave.open(ntype+'_test.wav','r')
  nl = nh.getnframes()
  noise = nh.readframes(nl)
  noise = struct.unpack('{n}h'.format(n=nl),noise)
  nh.close()
  noise = np.asarray(noise)/float(np.abs(noise).max())
  noisePower = np.square(noise).sum()
  f, t, noisestft = signal.stft(noise,nperseg=ftLen,noverlap=ftOlp)
  
  # ground truth noise psd
  noisestft2 = np.square(np.abs(noisestft))
  NPSD = np.zeros((fre_num,fra_num))
  npsd = np.zeros(fre_num)
  for fra in range(fra_num):
    npsd = lamb*npsd + (1-lamb)*noisestft2[:,fra]
    NPSD[:,fra] = npsd

  for snrindx in range(len(SNR)):
      snr = SNR[snrindx]
      print 'snr: '+str(snr)+'dB'
      
      # noisy signal
      scoe = np.sqrt(noisePower/(speechPower*10**(-float(snr)/10)))
      noisystft = scoe*speechstft+noisestft      
      
      ############# noise psd prediction ###############     
      data_x = np.abs(noisystft)
      pred_y = np.zeros(noisystft.shape)
 
      for fra in range(0,fra_num-num_steps,skp)+[fra_num-num_steps]:
         x_fra = np.zeros((fre_num,num_steps,flen))
         x_fra[:,:,0] = np.abs(np.concatenate((data_x[0,fra:fra+num_steps].reshape(1,num_steps),data_x[:-1,fra:fra+num_steps]),axis=0))
         x_fra[:,:,1] = np.abs(data_x[:,fra:fra+num_steps])
         x_fra[:,:,2] = np.abs(np.concatenate((data_x[1:,fra:fra+num_steps],data_x[-1,fra:fra+num_steps].reshape(1,num_steps)),axis=0))
         norml = np.mean(x_fra[:,:,1],axis=1).reshape(fre_num,1,1)
         xn_fra = x_fra/(norml)         
         pred_fra = model.predict_on_batch(xn_fra)
         pred_fra = (np.exp(pred_fra)*np.square(norml)).reshape(fre_num,num_steps)
   
         if fra==0:
           pred_y[:,:num_steps] = pred_fra
         elif fra==fra_num-num_steps:
           pred_y[:,fra+stp:] = pred_fra[:,stp:]      
         else: 
           pred_y[:,fra+stp:fra+enp] = pred_fra[:,stp:enp]
      ###################################################      

      LogErr[noiseindx,snrindx] = np.abs(10*np.log10(pred_y)-10*np.log10(NPSD)).mean()
  
  print LogErr[noiseindx,]     
    
     # freq = 50
     # plt.plot(10*np.log10(pred_y[freq,]))
     # plt.plot(10*np.log10(NPSD[freq,]))
     # plt.show()


