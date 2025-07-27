import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
import models

def loadData():

    data = sio.loadmat('../../data/houstonU2018.mat')['houstonU']
    labels = sio.loadmat('../../data/houstonU2018.mat')['houstonU_gt']

    return data, labels

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData_removed = patchesData[patchesLabels>0,:,:,:]
        patchesLabels_removed = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData_removed, patchesLabels_removed, patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=testRatio,random_state=randomState,stratify=y)
    return X_train, X_test, y_train, y_test
BATCH_SIZE_TRAIN = 64

def create_data_loader():
    X, y = loadData()
    y_out = y
    index = np.nonzero(y.reshape(y.shape[0]*y.shape[1]))
    index = index[0]
    test_ratio = 0.95
    patch_size = 5
    pca_components = 30

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)
    print('\n... ... create data cubes ... ...')
    X_pca, y, X_pca_whole, y_whole = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
    Xtest = X_pca
    ytest = y
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    X_whole = X_pca_whole.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    X_whole = X_whole.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    X = TestDS(X, y)
    X_whole = TestDS(X_whole, y_whole)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               drop_last=True
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=False
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False
                                              )
    all_data_loader_whole = torch.utils.data.DataLoader(dataset=X_whole,
                                                        batch_size=BATCH_SIZE_TRAIN,
                                                        shuffle=False,
                                                        num_workers=0,
                                                        drop_last=False
                                                        )

    return train_loader, test_loader, y, index, all_data_loader, all_data_loader_whole, y_out

""" Training dataset"""
class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):
    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len

def train(train_loader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = models.CNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    total_loss = 0
    for epoch in range(100):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))
    print('Finished Training')
    return net, device

def mytest(device, net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))
    return y_pred_test, y_test
def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    return oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':
    oa = []
    acc = []
    aa = []
    kappa = []
    for num in range(1):
        train_loader, test_loader, y_all, index, all_data_loader, all_data_loader_whole, y = create_data_loader()
        tic1 = time.perf_counter()
        net, device = train(train_loader)
        toc1 = time.perf_counter()
        tic2 = time.perf_counter()
        y_pred_test, y_test = mytest(device,net, test_loader)
        toc2 = time.perf_counter()
        each_oa, confusion, each_acc, each_aa, each_kappa = acc_reports(y_test, y_pred_test)
        oa.append(each_oa)
        acc.append(each_acc)
        aa.append(each_aa)
        kappa.append(each_kappa)
        print('\n Training time: %.2f seconds' % (toc1 - tic1))
        print('\n Testing time: %.2f seconds' % (toc2 - tic2))
    print(np.mean(acc, axis=0), np.std(acc, axis=0))
    print('\n OA: %.2f, std: %.2f ' % (np.mean(oa), np.std(oa)))
    print('\n AA: %.2f, std: %.2f ' % (np.mean(aa), np.std(aa)))
    print('\n Kappa: %.2f, std: %.2f ' % (np.mean(kappa), np.std(kappa)))