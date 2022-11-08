import pickle

from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
import torch
from scipy import signal
import scipy.io # To load .mat files
import mne
import math

def getAvailbatchsize(idxs, lens):
    maxbatch = lens
    for idx, values in lens.items():
        temp = []
        for name, value in values.items():
            temp.append(value)
        if len(temp) != 1:
            gcd = np.gcd.reduce(temp)
        else:
            gcd = temp[0]

        if gcd%idxs[idx]["batch"] != 0:
            maxbatch[idx]["batch"] = gcd
        else:
            maxbatch[idx]["batch"] = idxs[idx]["batch"]

    return maxbatch

def printDatainfo(lens):
    for idx, values in lens.items():
        print(idx + " :")
        for name, value in values.items():
            if name == "batch":
                print(" " + idx + " batch size is " + str(value))
            else:
                print(" - " + name + " : " + str(value))

class DEAPDataset(Dataset):
    def __init__(self, idxs, ssl = True, slength=1000, ratio = None):
        self.path = '../TIPNetPrac/data/DEAP/'
        data = []
        labels = []
        for idx in idxs:
            tmp = pickle.load(open(self.path + f's{idx:02d}.dat', 'rb'), encoding='latin1')
            if ratio != None:
                length = int(tmp['data'].shape[0] * ratio)
                data.append(tmp['data'][:length, :32, -30*128:])
                labels.append(tmp['labels'][:length, :])
            else:
                data.append(tmp['data'][:, :32, -30*128:])
                labels.append(tmp['labels'])

        del tmp

        data = np.moveaxis(np.concatenate(data, axis=0), -1, 0)
        data = signal.resample(data, slength)
        data = np.moveaxis(data, 0, -1)

        self.labels = np.concatenate(labels, axis=0)
        self.data = data
        self.ssl = ssl
        self.slength = slength

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

class SEEDDataset(Dataset):
    def __init__(self, idxs, ssl = True, slength=1000, rm_front = False, ratio = None):
        self.path = '../TIPNetPrac/data/SEED/'
        data = []
        labels = []

        y = scipy.io.loadmat(self.path + f'label.mat')['label']
        for idx in idxs:
            data_tr = scipy.io.loadmat(self.path + f'Trainingset/Data_Sample{idx:02d}.mat')

            if ratio != None:
                length = int(15 * ratio)+1
                label_temp = []
                for i in range(1, length):
                    data.append(np.array(data_tr['djc_eeg' + str(i)]))
                    label_temp.append(y.item((0,i)))

                label_temp = np.array(label_temp).reshape([1, -1])
                labels.append(np.array(label_temp))

                del label_temp
            else:
                for i in range(1, 16):
                    data.append(np.array(data_tr['djc_eeg' + str(i)]))

                # Y_tr = np.repeat(y, repeats=15, axis=0)
                labels.append(y)

        del data_tr

        temp = []
        for dt in data:
            if rm_front:
                temp_data = dt[:, :30*1000]
                temp.append(temp_data)
            else:
                temp_data = dt[:, -30*1000:]
                temp.append(temp_data)
        data = np.array(temp)

        del temp, temp_data, dt

        data = np.moveaxis(data, -1, 0)
        data = signal.resample(data, slength)

        self.data = np.moveaxis(data, 0, -1)
        self.labels = np.array(labels).flatten()
        self.ssl = ssl
        self.slength = slength

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

class SEEDIVDataset(Dataset):
    def __init__(self, idxs, ssl = True, slength=1000, rm_front = True, ratio = None):
        self.path = '../TIPNetPrac/data/SEED-IV/'
        data = []
        labels = []

        y = scipy.io.loadmat(self.path + f'label.mat')['label']
        for idx in idxs:
            data_tr = scipy.io.loadmat(self.path + f'Trainingset/Data_Sample{idx:02d}.mat')

            if ratio != None:
                length = int(15 * ratio) + 1
                label_temp = []
                for i in range(1, length):
                    data.append(np.array(data_tr['djc_eeg' + str(i)]))
                    label_temp.append(y.item((0, i)))

                label_temp = np.array(label_temp).reshape([1, -1])
                labels.append(np.array(label_temp))

                del label_temp
            else:
                for i in range(1, 16):
                    data.append(np.array(data_tr['djc_eeg' + str(i)]))

                # Y_tr = np.repeat(y, repeats=15, axis=0)
                labels.append(y)

        del data_tr

        temp = []
        for dt in data:
            if rm_front:
                temp_data = dt[:, :30*1000]
                temp.append(temp_data)
            else:
                temp_data = dt[:, -30*1000:]
                temp.append(temp_data)
        data = np.array(temp)

        del temp, temp_data, dt

        data = np.moveaxis(data, -1, 0)
        data = signal.resample(data, slength)

        self.data = np.moveaxis(data, 0, -1)
        self.labels = np.array(labels).flatten()
        self.ssl = ssl
        self.slength = slength

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

class CHB_MIT_Dataset(Dataset):
    def __init__(self, idxs, ssl=True, slength=1000, ratio = None):
        #self.path = os.getcwd()
        self.path = '../TIPNetPrac/data/CHB-MIT/'

        data = []
        labels = []

        for idx in idxs:
            tmp = np.load(self.path + f'Data_Sample{idx:02d}.npy')
            #y = np.load(self.path + f'Data_Label{idx:02d}.npy')
            y = np.load(self.path + f'Data_Label{idx:02d}.npy')

            if ratio != None:
                length = int(tmp.shape[0] * ratio)
                data.append(tmp[:length, :, :])
                labels.append(y[:length])
            else:
                data.append(tmp)
                labels.append(y)

        del tmp, y

        data = np.concatenate(data, axis=0)

        data = np.moveaxis(data, -1, 0)
        data = signal.resample(data, slength)

        self.data = np.moveaxis(data, 0, -1)
        self.labels = np.array(np.concatenate(labels, axis=0))
        self.ssl = ssl
        self.slength = slength

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


class MASSDataset(Dataset):
    def __init__(self, idx, ssl=True, slength=1000, ratio=None):
        self.idx = idx
        self.sequence_length = 100000  # 바꿔가면서,
        self.path = '../TIPNetPrac/data/MASS/SS1_'
        X = []

        for i in self.idx:
            a = self.path + f'{i}.pickle'  # path
            with open(file=a, mode='rb') as f:
                data = pickle.load(f)  # data = (c,t)
            if ratio != None:
                _, length = data.shape
                length = int(length * ratio)
                data = data[:, :length]

            data = mne.filter.resample(data, down=1.28)  # downsampling to 200Hz
            X.extend(data)

        self.X = np.array(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item]


class SleepdefDataset(Dataset):
    def __init__(self, idx, ssl=True, sfreq=200, ratio=None):
        self.idx = idx
        self.sequence_length = 200000  # 바꿔가면서,
        self.path = '/content/drive/MyDrive/sleep_edfx/sleep_edfx_CT+SC/'
        X = []

        for i in self.idx:
            a = self.path + f'{i}.pickle'  # path
            with open(file=a, mode='rb') as f:
                data = pickle.load(f)  # data = (c,t)
            if ratio != None:
                _, length = data.shape
                length = int(length * ratio)
                data = data[:, :length]

            data = mne.filter.resample(data, up=2.0)  # upsampling to 200Hz
            X.extend(data)

        self.X = np.array(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item]


class ConcatDataInit():
    def __init__(self, idxs, ssl = True, slength=1000):
        dataset = {"train" : [], "validation" : [], "test" : []}
        self.len = {
            "train":{
            }, "validation" : {
            }, "test" : {
            }
        }
        for idx, values in idxs.items():
            for name, value in values.items():
                if name == "batch":
                    continue
                if len(value) == 1:
                    if name == "SEED":
                        temp_dataset = SEEDDataset(value, ssl, slength)
                    elif name == "SEEDIV":
                        temp_dataset = SEEDIVDataset(value, ssl, slength)
                    elif name == "DEAP":
                        temp_dataset = DEAPDataset(value, ssl, slength)
                    elif name == "CHB-MIT":
                        temp_dataset = CHB_MIT_Dataset(value, ssl, slength)
                    elif name == "Sleepdef":
                        temp_dataset = SleepdefDataset(value, ssl, slength)
                    elif name == "MASS":
                        temp_dataset = MASSDataset(value, ssl, slength)
                else:
                    dem, ratio = value[0], value[1]
                    if name == "SEED":
                        temp_dataset = SEEDDataset(dem, ssl, slength, ratio = ratio)
                    elif name == "SEEDIV":
                        temp_dataset = SEEDIVDataset(dem, ssl, slength, ratio = ratio)
                    elif name == "DEAP":
                        temp_dataset = DEAPDataset(dem, ssl, slength, ratio = ratio)
                    elif name == "CHB-MIT":
                        temp_dataset = CHB_MIT_Dataset(dem, ssl, slength, ratio = ratio)
                    elif name == "Sleepdef":
                        temp_dataset = SleepdefDataset(value, ssl, slength)
                    elif name == "MASS":
                        temp_dataset = MASSDataset(value, ssl, slength)

                self.len[idx][name] = temp_dataset.__len__()
                dataset[idx].append(temp_dataset)

        self.len = getAvailbatchsize(idxs, self.len)

        printDatainfo(self.len)

        self.tr_dataset = DataLoader(dataset=torch.utils.data.ConcatDataset(dataset["train"]),
                                                    batch_size=self.len["train"]["batch"])
        self.val_dataset = DataLoader(dataset=torch.utils.data.ConcatDataset(dataset["validation"]),
                                                    batch_size=self.len["validation"]["batch"])
        self.te_dataset = DataLoader(dataset=torch.utils.data.ConcatDataset(dataset["test"]),
                                                    batch_size=self.len["test"]["batch"])

        del dataset

    def getTrain(self):
        return self.tr_dataset

    def getVal(self):
        return self.val_dataset

    def getTest(self):
        return self.te_dataset





