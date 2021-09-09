# import module
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data  # Data.Dataset , Data.DataLoader
import cv2
from torchvision import transforms

torch.manual_seed(1)  # Reproducible

# constant
imgSize = 64  # Images are [64 x 64]


# Hyper-Parameters
EPOCH = 300  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001  # learning rate
kernel_size = 3
padding = int((kernel_size-1)/2)
filterNum = 10

# Data Path
TestPath = "./TestFace"
TrainPath = "./TrainFace"


# ----- Create Dataset ----- #
class TorchDataset(Data.Dataset):
    def __init__(self, filePath, repeat=1):
        """
        :param filePath: the director where store the Images(test or train)
        :param res_ImgSize: default = 64
        :param repeat: the repeat times of all sample, default is 1 time
        """
        self.filePath = filePath
        self.image_label_list = self.read_file(filePath)
        self.len = len(self.image_label_list)
        self.repeat = repeat

        '''class torchvision.transforms.ToTensor'''
        self.toTensor = transforms.ToTensor()


    def __getitem__(self, i):
        index = i % self.len
        label = np.array(self.image_label_list[index])
        ImgContain = cv2.imread(str(self.filePath + '/' + str(index) + '.jpg'))
        ImgContain = cv2.resize(ImgContain, (64, 64), interpolation=cv2.INTER_CUBIC)
        # ImgContain.transpose(2, 0, 1)
        Imgdata = ImgContain.transpose(2, 0, 1)
        Imgdata = torch.from_numpy(np.asarray(Imgdata))
        Imgdata = Imgdata.type(torch.FloatTensor) / 255.
        return Imgdata, label

    def __len__(self):
        data_len = len(self.image_label_list) * self.repeat
        return data_len

    def read_file(self, filePath):
        # load Data label
        print(filePath + "!!!")
        image_label = np.load(filePath + '/SubimgLabel.npy')
        return image_label

    def data_preproccess(self, data):
        data = self.toTensor(data)
        return data
#-----------------------------#

# load Train label and Test label
Test_label = np.load(TestPath + '/SubimgLabel.npy')
Train_label = np.load(TrainPath + '/SubimgLabel.npy')


# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 3, 64, 64) # 3 color -> (R,G,B)
train_data = TorchDataset(filePath=TrainPath, repeat=1)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = TorchDataset(filePath=TestPath, repeat=1)
test_loader = Data.DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)



# ----- Construct CNN ----- #
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28) [3 x 64 x 64]
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=filterNum*3,             # filters:10, out_channel=3*3=9
                kernel_size=kernel_size,              # filter size [9 x 9]
                stride=1,                   # filter movement/step
                padding=padding,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape [9 x 64 x 64]
            #nn.Dropout(0.5),                # drop 50% of the neuron
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape [9 x 32 x 32]
        )
        self.conv2 = nn.Sequential(         # input shape [9 x 32 x 32]
            nn.Conv2d(filterNum*3, filterNum*3*2, kernel_size, 1, padding),     # output shape [18 x 32 x 32]
            #nn.Dropout(0.5),                # drop 50% of the neuron
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape [18 x 16 x 16]
        )
        self.conv3 = nn.Sequential(         # input shape  [18 x 16 x 16]
            nn.Conv2d(filterNum*3*2, filterNum*3*2*2, kernel_size, 1, padding),     # output shape [36 x 16 x 16]
            #nn.Dropout(0.5),                # drop 50% of the neuron
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape  [36 x 8 x 8]
        )
        self.out = nn.Linear((filterNum*3*2*2) * 8 * 8, 3)   # fully connected layer, output 3 classes

    def forward(self, x):
        #with torch.no_grad():
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to [batch_size, 120 x 8 x 8]
        output = self.out(x)
        return output, x    # return x for visualization

# ---------------------- #

cnn = CNN()
print(cnn)  # net architecture

# optimize all cnn parameters
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.01)

device = torch.device("cuda")
total_size = Train_label.sum()
weights = total_size/sum(Train_label)
class_weights = torch.FloatTensor(weights).to(device)
loss_func = nn.CrossEntropyLoss(weight=class_weights)


# training and testing
TrainAcc = []
TestAcc = []
TrainLoss = []
TestLoss = []
TrainPredict = []
isFirst = True

cnn = cnn.to(device)
for epoch in range(EPOCH):
    train_acc = 0.
    train_loss = 0.
    i = 0
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        cnn.train()
        output = cnn(b_x)[0]               # cnn output
        B_y = torch.max(b_y, 1)[1]
        loss = loss_func(output, B_y)   # cross entropy loss
        train_loss += loss.data
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        
        # training error rate
        pred = torch.max(output, 1)[1]
        num_correct = (pred == B_y).sum()
        train_acc += float(num_correct.data)
        if epoch == 2:
            if isFirst == True:
                TrainPredict = (output.float().cpu()).detach().numpy()
                isFirst = False
            else:
                TrainPredict = np.append(TrainPredict,np.array((output.float().cpu()).detach()), axis=0)
                #TrainPredict.append(np.array((output.float().cpu()).detach()))
        


    cnn.eval()  
    eval_loss2 = 0.
    eval_acc = 0.

    for i, (t_x, t_y) in enumerate(test_loader):
        t_x = t_x.to(device)
        t_y = t_y.to(device)
        output2 = cnn(t_x)[0]
        T_y = torch.max(t_y, 1)[1]
        loss2 = loss_func(output2, T_y)
        eval_loss2 += loss2.data
        #print(T_y)        
        pred2 = torch.max(output2, 1)[1]
        #print(pred2) 
        num_correct2 = (pred2 == T_y).sum()
        eval_acc += float(num_correct2.data)
        
    test_rate = eval_acc / float(len(test_data))
    train_rate = train_acc / float(len(train_data))
    print("-----Epoch"+str(epoch)+"-----")
    print('Test Acc: {:.6f}'.format(test_rate))
    print('Train Acc: {:.6f}'.format(train_rate))
    # testloss: tensoe type -> cpu.float
    TestAcc.append(test_rate)
    TestLoss.append((eval_loss2.cpu().item()) / float(len(test_data)))
    TrainAcc.append(train_rate)
    TrainLoss.append((train_loss.cpu().item()) / float(len(train_data)))


np.save("TrainPredict.npy",np.array(TrainPredict))
np.save("TestPredict.npy",(output2.float().cpu()).detach().numpy())
np.save("TestAcc.npy",np.array(TestAcc))
np.save("TestLoss.npy",np.array(TestLoss))
np.save("TrainAcc.npy",np.array(TrainAcc))
np.save("TrainLoss.npy",np.array(TrainLoss))

