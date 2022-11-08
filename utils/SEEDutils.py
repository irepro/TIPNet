# Import APIs
import scipy.io # To load .mat files
import numpy as np

from mne.filter import filter_data, notch_filter
from scipy import signal

# Define load imagined speech EEG dataset class
class load_dataset():
    def __init__(self, sbj_idx):
        self.sbj_idx = sbj_idx
        #self.path = '/DataCommon/jhjeon/Track3/'
        self.path = '../TIPNetPrac/data/SEED/'  # Define the data path

    def load_data(self):
        X = []
        Y = []
        for sbj_idx in self.sbj_idx:
            data_tr = scipy.io.loadmat(self.path + f'Trainingset/Data_Sample{sbj_idx:02d}.mat')

            y = np.array(scipy.io.loadmat(self.path + f'label.mat')['label'])
            for i in range(1,16):
                X.append(np.array(data_tr['djc_eeg'+str(i)]))

            Y_tr = np.repeat(y, repeats=15, axis=0)
            Y.extend(Y_tr)

            del data_tr
        return  X, np.array(Y)

    def preprocessing(self, data, sfreq, rm_front = False):
        # # BPF (30~125Hz, gamma range) using a 4th order Butterworth filter
        # data = filter_data(data, sfreq=256, l_freq=30, h_freq=125, verbose=False)

        # Remove 60Hz line noise with the 120Hz harmonic
        #data = notch_filter(data, Fs=256, freqs=np.arange(60, 121, 60), verbose=False)
        temp = []
        for dt in data:
            if rm_front:
                temp_data = dt[:,:30000]
                temp.append(temp_data)
            else:
                temp_data = dt[:, -30000:]
                temp.append(temp_data)
        data = np.array(temp)

        del temp, temp_data, dt

        data = np.moveaxis(data, -1, 0)
        data = signal.resample(data, 30*sfreq)
        data = np.moveaxis(data, 0, -1)

        return data

    def call(self, fold, ssl = False, sfreq = None):
        X, Y = self.load_data()
        if ssl:
            X = self.preprocessing(X, sfreq)

        num_samples = int(len(X)/fold) # Samples per fold

        # Set training/validation/testing data indices
        rand_idx = np.random.RandomState(seed=981220).permutation(len(X))
        test_idx = rand_idx[(fold - 1) * num_samples:fold * num_samples]
        train_idx = np.setdiff1d(rand_idx, test_idx)
        valid_idx = np.random.RandomState(seed=3940).permutation(train_idx.shape[0])[:num_samples]
        valid_idx = train_idx[valid_idx]
        train_idx = np.setdiff1d(train_idx, valid_idx)

        #X = np.expand_dims(X, axis=-1) # (5250, 64, 256, 1)

        X_tr, X_vl, X_ts = X[train_idx, ...], X[valid_idx, ...], X[test_idx, ...]
        Y_tr, Y_vl, Y_ts = Y[train_idx, ...], Y[valid_idx, ...], Y[test_idx, ...]

        return (X_tr, Y_tr), (X_vl, Y_vl), (X_ts, Y_ts)