import torch
import numpy as np

#  stationary: moving averaging filter
def MAF(x, window):
    new_signals = np.zeros(x.shape) 
    C, T = x.shape # C : batch_size*channels
    #각 sample별로 moving averaging filter(channel 1개의 신호를 하나의 sample로 봄)
    for i in range(C): 
        noise = np.convolve(x[i], np.ones(window), 'same') / window
        new_signals[i] = x[i] - noise
    return new_signals

# trendstationary
def asln(x, a, b):
    C, T = x.shape # C :batch_size*channels
    noise = a * np.linspace(0, 1, T) - b * np.ones(T)
    noise = noise * np.ones((C, 1))
    
    #randomly add or subtract noise
    n = np.random.randint(2)
    if n == 0:
        new_signals = x + noise
    else:
        new_signals = x - noise
    return new_signals

#cyclostationary
def apn(x, c, d):
    C, T = x.shape # # C :batch_size*channels
    noise = np.linspace(-np.pi + d, np.pi + d, T) - d * np.ones(T)
    noise = c * np.sin(noise) * np.ones((C, 1))
    new_signals = x + noise
    return new_signals


def augmented_data(x, window, a, b, c, d):
    sequence_length = 200000

    # x가서(B*C,1,T)인 tensor로 들어왔던 것으로 기억,,,
    # tensor로는 위에서 정의한 augmentation들이 적용 안되어서 ndarray로 변
    channels, length = x.shape
    a = sequence_length * int(length / sequence_length)
    if length == a:
        x = np.reshape(x, (int(length / sequence_length * channels), 1, sequence_length))
    else:
        x = x[:, :a]
        x = np.reshape(x, (int(a / sequence_length * channels), 1, sequence_length))

    x = np.squeeze(x, 1)

    x = np.squeeze(x)
    x = x.numpy()

    samples, sequence_length = x.shape형
    # augmented signals을 빈 list에 담아
    X = []
    Y = []
    x_maf = MAF(x, window)
    x_asln = asln(x, a, b)
    x_apn = apn(x, c, d)
    X.append(x)
    X.append(x_maf)
    X.append(x_asln)
    X.append(x_apn)

    # one-hot encoding
    original = [1, 0, 0, 0]
    s_maf = [0, 1, 0, 0]
    s_asln = [0, 0, 1, 0]
    s_apn = [0, 0, 0, 1]

    Y.append(original * np.ones((samples, 1))) # (1,0,0,0)이 smaples수 만큼 존재. (4,samples)
    Y.append(s_maf * np.ones((samples, 1)))
    Y.append(s_asln * np.ones((samples, 1)))
    Y.append(s_apn * np.ones((samples, 1)))

    X = np.array(X)  # X : (4,samples,sequence_length) 
    X = X.reshape(4 * samples, sequence_length) 

    Y = np.array(Y)  # X : (4,sample,4) 
    Y = Y.reshape(4 * samples, 4)

    return X, Y



class Temporal_Trend_Identification_Task_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

    def forward(self, batch, encoder, train):
        acc = 0
        x, y = augmented_data(batch, window=50, a=50, b=0, c=50, d=1.5) 
        c, t = x.shape #(4*sample,sequence_length), sample = batch_size * channels
        x = np.reshape(x, (c, 1, t)) # separableconv1d이 요구하는 인풋 형태가 (batch,1,t),, # 현재 코드는 batch <-- batch*c,개선필요
        CrossEL = torch.nn.CrossEntropyLoss()
        pred = encoder.forward(torch.Tensor(x).to(device))
        y = torch.Tensor(y).to(device)
        loss = CrossEL(pred, torch.Tensor(y).to(device))

        if train:
            loss.backward(retain_graph=True)

        _, y = torch.max(y, 1)# 행에서 y의 max인 자리의 위치를 리턴,
        _, predicted = torch.max(pred, 1)  #행에서 pred의 max인 자리의 위치를 리턴
    
        acc = (predicted == y).sum().item() # max인 위치가 같은 갯수 return
        acc = acc / c  # acc/(batch*channels*4(augmented)) # 맞은 갯수 / 전체 갯
        loss = loss
        del x
        del y
        return loss, acc
