import torch
import torch.nn as nn

class C3D(nn.Module):
    '''
    The C3D network
    '''
    
    def __init__(self,seq_len,feature_dim):  #feature_dim = 768
        super(C3D,self).__init__()
        
        # x = [batch_size,3,32,112,112]
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))    # x = [batch_size,64,32,112,112]
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))         # x = [batch_size,64,32,56,56]

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # x = [batch_size,128,32,56,56]
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))         # x = [batch_size,128,16,28,28]

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))# x = [batch_size,256,16,28,28]
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))# x = [batch_size,256,16,28,28]
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))         # x = [batch_size,256,8,14,14]

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))# x = [batch_size,512,8,14,14]
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))# x = [batch_size,512,8,14,14]
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))         # x = [batch_size,512,4,7,7]

        self.conv5a = nn.Conv3d(512,512, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # x = [batch_size,512,4,7,7]
        self.conv5b = nn.Conv3d(512,512, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # x = [batch_size,512,4,7,7]
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),padding=(0, 1, 1))         # x = [batch_size,512,2,4,4]

        self.conv6 = nn.Conv3d(512,512, kernel_size=(3, 3, 3), padding=(1, 1, 1))   # x = [batch_size,512,2,4,4]
        self.pool6 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))          # x = [batch_size,512,2,2,2]

        self.fc5 = nn.Linear(4096,seq_len*feature_dim)

        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(p=0.5)

        self.relu =  nn.ReLU()
        

    def forward(self,x):  #x = [batch_size,3,32,56,56]
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = self.relu(self.conv6(x))
        x = self.pool6(x)

        x = x.view(-1, 4096)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)

        #Video_feature_Out = self.fc7(x)
        Video_feature_Out = x.view(-1,self.seq_len,self.feature_dim)
        return Video_feature_Out   #output_size = [batch_size,feature_dim=768]


#This is a test
# x = torch.rand(32,3,32,112,112)
# net = C3D(8,768)
# out_put = net(x)
# print(out_put.size())
