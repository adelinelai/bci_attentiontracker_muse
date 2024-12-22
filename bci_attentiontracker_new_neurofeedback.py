
# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers

This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it shows how ratios of the band powers can be used to estimate
mental state for neurofeedback.
"""

import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
import pyautogui
from pynput.keyboard import Key, Controller
from datetime import datetime
from win11toast import toast
import time

# Handy little enum to make code more readable
class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing
BUFFER_LENGTH = 5  # EEG data buffer length in seconds
EPOCH_LENGTH = 1  # Epoch length used for FFT in seconds
OVERLAP_LENGTH = 0  # Amount of overlap between two consecutive epochs in seconds
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH  # How much to 'shift' between consecutive epochs
INDEX_CHANNEL = [0]  # Index of the channel(s) to be used

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    info = inlet.info()
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))
    band_buffer = np.zeros((n_win_test, 4))

    """ 3. GET DATA """
    print('Press Ctrl-C in the console to break the while loop.')
   
    t = datetime.now()
    focus_start_time = None  # To track when focus starts
    focus_time = 0  # Track how long focus has lasted in seconds
    try:
        keyboard = Controller()
        while True:

            """ 3.1 ACQUIRE DATA """
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))
            
            if len(eeg_data) == 0:
                continue  # Skip this iteration if no data was received

            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

            # Update EEG buffer with the new data
            eeg_buffer, filter_state = utils.update_buffer(
                eeg_buffer, ch_data, notch=True, filter_state=filter_state)

            """ 3.2 COMPUTE BAND POWERS """
            data_epoch = utils.get_last_data(eeg_buffer, EPOCH_LENGTH * fs)
            band_powers = utils.compute_band_powers(data_epoch, fs)
            band_buffer, _ = utils.update_buffer(band_buffer, np.asarray([band_powers]))
            smooth_band_powers = np.mean(band_buffer, axis=0)

            #Assigning the varible to the theta/beta
            theta_beta_ratio = smooth_band_powers[Band.Theta] / smooth_band_powers[Band.Beta]
            #Printing it's value
            print('Theta/Beta Ratio: ', theta_beta_ratio)
            
            #Track focus time when beta metric < 0.9
            if theta_beta_ratio < 0.9:
                if focus_start_time is None:
                    focus_start_time = time.time()  # Start tracking time when focus begins
                focus_time = time.time() - focus_start_time  # Calculate time spent focused
            
            if theta_beta_ratio > 0.95 and (time.time() - t.timestamp()) > 10:
                print("----------- CONCENTRATION CHANGE -----------")
                toast(f"You're distracted. You've been focused for {focus_time:.2f} seconds. This is a friendly reminder to FOCUS.")
                focus_start_time = None  # Reset focus time after notification
                focus_time = 0  # Reset focus time after concentration change
                t = datetime.now()  # Reset time after distraction

    except KeyboardInterrupt:
        print('Closing!')