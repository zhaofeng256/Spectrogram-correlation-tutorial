# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:11:35 2021

@author: Administrator
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
import glob
import pandas as pd
import datetime as dt
from skimage.feature import match_template
from scipy.signal import find_peaks
from matplotlib.path import Path
from inspect import currentframe, getframeinfo

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

#%% load template shape
# kernel_csv=r"kernel_zcall_1.csv"
kernel_csv=r"rec_2022-12-17_14-08-35.csv"
df=pd.read_csv(kernel_csv,index_col=0)
shape_t=df['Timestamp'].values - df['Timestamp'].min()
print(df['Timestamp'].values)
shape_f=df['Frequency'].values

#%% load .wav and calc spectrogram

# audiopath=  r'aural_2016_04_02_12_25_00.wav'
# time= dt.datetime.strptime( audiopath.split('\\')[-1], 'aural_%Y_%m_%d_%H_%M_%S.wav' )
audiopath=  r'rec_2022-12-17_14-08-35.wav'
time= dt.datetime.strptime( audiopath.split('\\')[-1], 'rec_%Y-%m-%d_%H-%M-%S.wav' )

fs, x = wav.read(audiopath)
if x.shape[1] == 2:
    x=x[:,0]
x=x/32767    
dBFS=155
p =np.power(10,(dBFS/20))*x #convert data.signal to uPa    
# fft_size=2**17
fft_size=8192
f, t, Sxx = signal.spectrogram(p, fs, window='hamming',nperseg=fft_size,noverlap=0.8*fft_size)
# f, t, Sxx = signal.spectrogram(p, fs)
print(f.shape, t.shape, Sxx.shape)
#%%

def spectrogram_correlation(f, t, Sxx,shape_t,shape_f,offset_t,offset_f):
    print(shape_t, shape_f)
    f_lim=[ shape_f.min() - offset_f ,  shape_f.max() + offset_f ]

    # offset_f=50
    # offset_t=0.2
    k_length_seconds=shape_t.max()+offset_t*2
    shape_t=shape_t+offset_t
    
    # generate kernel  
    time_step=np.diff(t)[0]
    
    k_t=np.linspace(0,k_length_seconds,int(k_length_seconds/time_step) )
    ix_f=np.where((f>=f_lim[0]) & (f<=f_lim[1]))[0]
    k_f=f[ix_f[0]:ix_f[-1]]
    # k_f=np.linspace(f_lim[0],f_lim[1], int( (f_lim[1]-f_lim[0]) /f_step)  )
    print(k_t.shape,k_f.shape)
    kk_t,kk_f=np.meshgrid(k_t,k_f)   
    kernel_background_db=0
    kernel_signal_db=1
    kernel=np.ones( [ k_f.shape[0] ,k_t.shape[0] ] ) * kernel_background_db
    
    x, y = kk_t.flatten(), kk_f.flatten()
    points = np.vstack((x,y)).T 
    p = Path(list(zip(shape_t, shape_f))) # make a polygon
    grid = p.contains_points(points)
    print(points.shape, grid.shape, kk_t.shape)
    mask = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon  
    kernel[mask]=kernel_signal_db

    fig=plt.figure(num=2)
    plt.clf()
    plt.imshow(kernel,origin = 'lower',aspect='auto',extent=[k_t[0],k_t[-1],k_f[0],k_f[-1]])

    ix_f=np.where((f>=f_lim[0]) & (f<=f_lim[1]))[0]
    spectrog =10*np.log10( Sxx[ ix_f[0]:ix_f[-1],: ] )

    print(spectrog.shape, kernel.shape)
    result = match_template(spectrog, kernel)
    corr_score=result[0,:]
    print(result.shape,corr_score.shape)
    print(result)
    print(t)
    t_score=np.linspace( t[int(kernel.shape[1]/2)] , t[-int(kernel.shape[1]/2)], corr_score.shape[0] )

    fig=plt.figure(num=3)
    plt.clf()
    plt.subplot(211)
    plt.imshow(spectrog,aspect='auto',origin = 'lower')
    plt.colorbar()
    plt.subplot(212)
    plt.plot(t_score,corr_score)
    plt.grid()
    plt.colorbar()
    plt.xlim( [t_score[0],t_score[-1]] )
    # plt.savefig(audiopath[:-4]+'_speccorr_zcall.jpg')

    #plt.show()
    return t_score,corr_score

t_score,corr_score=spectrogram_correlation(f, t, Sxx,shape_t,shape_f,3,0)

peaks_indices = find_peaks(corr_score, height=0)[0]
print('-----',peaks_indices)
print(t_score[peaks_indices])
# detection_times=time +  pd.to_timedelta( t_score[peaks_indices]  , unit='s')
# print(detection_times)



