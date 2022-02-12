# IDEA: Add neck to the posture map?
# from IPython import get_ipython

import time, os, sys, shutil
# from utils.fitting_utils import *

# for math and plotting
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#%matplotlib notebook
#%matplotlib widget
import scipy.stats as stats

from itertools import compress # for list selection with logical
from tqdm import tqdm

# from multiprocessing import Process

# and pytorch
import torch

import sys, os, pickle
# import cv2
from colour import Color
import h5py
from tqdm import tqdm, tqdm_notebook
import glob
import itertools


# Make a function that plots a psth onto an axis
def ax_paint_raster(ax,spike_times, event_times, pre_time = 4, post_time =2, dotcolor = 'k',dotsize = .5, linecolor = 'peru'):
    aligned_spikes = []
    for i,et in enumerate(event_times):
        st = spike_times[( spike_times >= (et-pre_time) )*( spike_times < (et+post_time) ) ] - et
        aligned_spikes.append(st)

        ax.plot(st,np.ones_like(st) + i,'.',markersize = dotsize,color=dotcolor)

    ax.set_xlim([-pre_time,post_time])
    ax.set_ylim([0-1,i+1])
    ax.axvline(0,c = linecolor)     

def ax_paint_psth(ax,spike_times, event_times, pre_time = 4, post_time =2, facecolor = 'lightgray', linecolor = 'peru',line=True):
    aligned_spikes = []
    for i,et in enumerate(event_times):
        st = spike_times[( spike_times >= (et-pre_time) )*( spike_times < (et+post_time) ) ] - et
        aligned_spikes.append(st)

    # bin the counts
    edges = np.linspace(-pre_time,post_time,40)
    count,_ = np.histogram(np.hstack(aligned_spikes),edges )

    rate = count/len(event_times) / np.median(np.diff(edges))

    ax.bar(edges[:-1],rate,width = np.median(np.diff(edges)), align ='edge' ,facecolor=facecolor)
    ax.set_xlim([-pre_time,post_time])
    if line:
        ax.axvline(0,c = linecolor) 
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Firing rate [Hz]')
                
def ax_paint_common_yaxis(axs):
    current_max = [ax.get_ylim()[1] for ax in axs]
    max_all = np.max(current_max)
    for ax in axs:
        ax.set_ylim(top = max_all)            
        
# using phy's fast acorr loop"
from phylib.stats import correlograms, firing_rate
def ax_paint_acorr(ax,spike_times,facecolor = 'grey'):
    window_size = 0.05
    bin_size = 0.001
    acorr = correlograms(spike_times,np.zeros_like(spike_times,dtype='int'),cluster_ids=[0],bin_size = bin_size,
                         window_size=window_size,sample_rate = 30000.)
    n_bins = len(acorr.squeeze())
    # acorr_times = np.arange(-window_size,window_size,bin_size)
    ax.bar( np.arange(0,n_bins), acorr.squeeze(),width = 1 ,facecolor = facecolor)
    ax.set_yticks([])
    ax.set_xlabel("Time [ms]")
    ax.set_xticks([0,window_size/bin_size])
    ax.set_xticklabels([-window_size*1e3,window_size*1e3])
    ax.set_ylabel('acorr')
    

    
    
# copied from phylab, because they keep it hidden

def _extract_waveform(traces, sample, channel_ids=None, n_samples_waveforms=None):
    """Extract a single spike waveform."""
    nsw = n_samples_waveforms
    assert traces.ndim == 2
    dur = traces.shape[0]
    a = nsw // 2
    b = nsw - a
    assert nsw > 0
    assert a + b == nsw
    if channel_ids is None:  # pragma: no cover
        channel_ids = slice(None, None, None)
        n_channels = traces.shape[1]
    else:
        n_channels = len(channel_ids)
    t0, t1 = int(sample - a), int(sample + b)
    # Extract the waveforms.
    w = traces[max(0, t0):t1][:, channel_ids]
    if not isinstance(channel_ids, slice):
        w[:, channel_ids == -1] = 0
    # Deal with side effects.
    if t0 < 0:
        w = np.vstack((np.zeros((nsw - w.shape[0], n_channels), dtype=w.dtype), w))
    if t1 > dur:
        w = np.vstack((w, np.zeros((nsw - w.shape[0], n_channels), dtype=w.dtype)))
    assert w.shape == (nsw, n_channels)
    return w

def extract_waveforms(traces, spike_samples, channel_ids, n_samples_waveforms=None):
    """Extract waveforms for a given set of spikes, on certain channels."""
    # Create the output array.
    ns = len(spike_samples)
    nsw = n_samples_waveforms
    assert nsw > 0, "Please specify n_samples_waveforms > 0"
    nc = len(channel_ids)
    # Extract the spike waveforms.
    out = np.zeros((ns, nsw, nc), dtype=traces.dtype)
    for i, ts in enumerate(spike_samples):
        out[i] = _extract_waveform(
            traces, ts, channel_ids=channel_ids, n_samples_waveforms=nsw)[np.newaxis, ...]
    return out


# cluster_id = 12

def ax_paint_waves(ax,cluster_id,phy_params_path,spike_index_phy,clusters_phy,linecolor= 'peru',dotcolor ='peru',dotsize=1,all_channels = False):
    # load the spike templates using the phy interface
    from phylib.io.model import load_model

    # Load the TemplateModel instance.
    model = load_model(phy_params_path)

    # get out the channel_ids and waves
    if all_channels:
        wave_channels = np.arange(32)
    else:
        wave_channels = bunch.channel_ids

    n_samples_waveforms = 100
    n_waveforms = 100
    model.n_samples_waveforms = n_samples_waveforms
    samples_to_load = np.random.choice(spike_index_phy[clusters_phy == cluster_id],size=n_waveforms,replace=False)
    waveforms = extract_waveforms(model.traces,samples_to_load,wave_channels,n_samples_waveforms)

    # load the probe
    spyking_circus_path = '/media/chrelli/SSD4TB/EPHYS_COPY/Cambridge_P2_circus_cleaner.prb'
    probe = {}
    with open(spyking_circus_path, 'r') as f:
        probetext = f.read()
        exec(probetext, probe)
        del probe['__builtins__']

    # plot the probe locations
    for ii in range(32):
        probe_xy = probe['channel_groups'][0]['geometry'][ii]

        # shift the shanks closer together
        if probe_xy[0] > 200:
            probe_xy[0] -= 180 

#         ax.plot(probe_xy[0],probe_xy[1],'o',color = dotcolor)

    # plot the waveforms    
    for i_ch,ch in enumerate(wave_channels):
        waves = waveforms[:,:,i_ch]

        probe_xy = probe['channel_groups'][0]['geometry'][ch]

        wave_width = 20
        wave_scaling = 0.01*2        
        wave_x = np.linspace(0,1,waves.shape[1]) * wave_width
        
        for j_wave in range(n_samples_waveforms):
            
    #         ax.plot(wave_x + probe_xy[0], wave_scaling*wave + probe_xy[1],color = linecolor)
            ax.plot(wave_x + probe_xy[0], wave_scaling*waves[j_wave,:] + probe_xy[1], c = linecolor, alpha = max(1/n_waveforms,0.01))
        
    ax.set_ylim(-15,187.5+15)    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
# plt.figure()
# ax = plt.gca()
# ax_paint_waves(ax,cluster_id,phy_params_path,linecolor= 'k',all_channels = True)
# plt.show()    
            