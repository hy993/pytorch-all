import torch
import torch.nn as nn
import torch.nn.functional as F

class model_test(nn.Module):
    # test
    def __init__(self):
        super(model_test, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size = 1, stride = 1)
        self.conv2_1 = nn.Conv2d(64, 32, kernel_size = 3, stride = 1)
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size = 3, stride = 2)
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size = 1, stride = 1)
        self.fc1 = nn.Linear(12*12, 96)
        self.fc2 = nn.Linear(96, 10)
        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        
        x = self.conv2_1(x)
        x = F.relu(x)
        
        x = self.conv3_1(x)
        x = F.relu(x)

        x = self.conv4_1(x)
        x = F.relu(x)
        
        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        #x = F.log_softmax(x)
        
        return F.log_softmax(x)

class model_test_1(nn.Module):
    def __init__(self):
        super(model_test_1, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 10, kernel_size = 3, stride = 1)
        self.conv2_1 = nn.Conv2d(10, 32, kernel_size = 5, stride = 2)
        self.fc1 = nn.Linear(32*11*11, 96)
        self.fc2 = nn.Linear(96, 10)
        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        
        x = self.conv2_1(x)
        x = F.relu(x)

        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        #x = F.log_softmax(x)
        
        return F.log_softmax(x)        

class model_test_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 10, kernel_size = 5, stride = 1)
        self.conv2_1 = nn.Conv2d(10, 10, kernel_size = 5, stride = 2)
        self.fc1 = nn.Linear(10*10*10, 96)
        self.fc2 = nn.Linear(96, 10)
        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        
        x = self.conv2_1(x)
        x = F.relu(x)

        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        #x = F.log_softmax(x)
        
        return F.log_softmax(x)  

class model_test_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 10, kernel_size = 5, stride = 1)
        self.conv2_1 = nn.Conv2d(10, 10, kernel_size = 5, stride = 2)
        self.fc1 = nn.Linear(10*10*10, 10)

        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        
        x = self.conv2_1(x)
        x = F.relu(x)

        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        
        return F.log_softmax(x) 

class model_test_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 10, kernel_size = 7, stride = 2)
        self.conv2_1 = nn.Conv2d(10, 10, kernel_size = 7, stride = 1)
        self.fc1 = nn.Linear(10*5*5, 10)

        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        
        x = self.conv2_1(x)
        x = F.relu(x)

        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        
        return F.log_softmax(x) 

class model_test_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 5, kernel_size = 7, stride = 2)
        self.conv2_1 = nn.Conv2d(5, 10, kernel_size = 7, stride = 1)
        self.fc1 = nn.Linear(10*5*5, 10)

        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        
        x = self.conv2_1(x)
        x = F.relu(x)

        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        
        return F.log_softmax(x) 

class model_test_base(nn.Module):
    def __init__(self):
        super().__init__()
        # 128x28
        self.conv1=nn.Conv2d(1,10,5)         # 10, 24x24
        self.conv2=nn.Conv2d(10, 20,3)       #128, 10x10
        self.fc1=nn.Linear(20*10*10, 500)
        self.fc2=nn.Linear(500, 10)
    def forward(self, x):
        in_size=x.size(0)       # in_size 为 batch_size（一个batch中的Sample数）
        # 卷积层 -> relu -> 最大池化
        out = self.conv1(x)     # 24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 12
        #卷积层 -> relu -> 多行变一行 -> 全连接层 -> relu -> 全连接层 -> sigmoid
        out = self.conv2(out)  # 10
        out = F.relu(out)
        out = out.view(in_size, -1)     # view()函数作用是将一个多行的Tensor,拼接成一行。
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        # softmax
        out = F.log_softmax(out, dim=1)
        # 返回值 out
        return out

class model_test_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 10, kernel_size = 3, stride = 1, padding=1)
        self.conv2_1 = nn.Conv2d(10, 10, kernel_size = 3, stride = 2, padding=1)

        self.fc1 = nn.Linear(10*14*14, 10)

        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        x_1_o = x
        
        x = self.conv2_1(x)
        x = F.relu(x)
        x_2_o = x

        x = F.avg_pool2d(x_1_o, kernel_size=2) + x_2_o
        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        
        return F.log_softmax(x) 
    
class model_test_resnet_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 10, kernel_size = 3, stride = 1, padding=1)
        self.conv2_1 = nn.Conv2d(10, 10, kernel_size = 3, stride = 2, padding=1)

        self.fc1 = nn.Linear(10*14*14, 512)
        self.fc2 = nn.Linear(512, 10)

        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        x_1_o = x
        
        x = self.conv2_1(x)
        x = F.relu(x)
        x_2_o = x

        x = F.avg_pool2d(x_1_o, kernel_size=2) + x_2_o
        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        return F.log_softmax(x) 
    
class model_test_resnet_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 10, kernel_size = 3, stride = 1, padding=1)
        self.conv2_1 = nn.Conv2d(10, 10, kernel_size = 3, stride = 2, padding=1)
        self.conv3_1 = nn.Conv2d(10, 1, kernel_size = 5, stride = 2)

        self.fc1 = nn.Linear(1*5*5, 10)
        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        x_1_o = x
        
        x = self.conv2_1(x)
        x = F.relu(x)
        x_2_o = x

        x = F.avg_pool2d(x_1_o, kernel_size=2) + x_2_o
        x = self.conv3_1(x)
        x = F.relu(x)
        
        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        
        return F.log_softmax(x) 
        
class model_test_resnet_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 10, kernel_size = 3, stride = 1, padding=1)
        self.conv2_1 = nn.Conv2d(10, 10, kernel_size = 3, stride = 2, padding=1)
        self.conv3_1 = nn.Conv2d(10, 10, kernel_size = 5, stride = 2)

        self.fc1 = nn.Linear(10*5*5, 10)
        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        x_1_o = x
        
        x = self.conv2_1(x)
        x = F.relu(x)
        x_2_o = x

        x = F.avg_pool2d(x_1_o, kernel_size=2) + x_2_o
        x = self.conv3_1(x)
        x = F.relu(x)
        
        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        
        return F.log_softmax(x) 
    
class model_test_resnet_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 10, kernel_size = 3, stride = 1, padding=1)
        self.conv2_1 = nn.Conv2d(10, 10, kernel_size = 3, stride = 2, padding=1)
        self.conv3_1 = nn.Conv2d(10, 10, kernel_size = 5, stride = 5)

        self.fc1 = nn.Linear(10*2*2, 10)
        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        x_1_o = x
        
        x = self.conv2_1(x)
        x = F.relu(x)
        x_2_o = x

        x = F.avg_pool2d(x_1_o, kernel_size=2) + x_2_o
        x = self.conv3_1(x)
        x = F.relu(x)
        
        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        
        return F.log_softmax(x)

class model_test_resnet_5(nn.Module):
#测试网络结构
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 10, kernel_size = 3, stride = 1, padding=1)
        self.conv2_1 = nn.Conv2d(10, 10, kernel_size = 3, stride = 2, padding=1)
        self.conv3_1 = nn.Conv2d(10, 10, kernel_size = 5, stride = 5)

        self.fc1 = nn.Linear(10*2*2, 10)
        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = F.relu(x)
        x = self.conv1_1(x)
        #x = F.relu(x)
        x_1_o = x
        
        x = F.relu(x)
        x = self.conv2_1(x)
        #x = F.relu(x)
        x_2_o = x

        x = F.avg_pool2d(x_1_o, kernel_size=2) + x_2_o
        
        x = F.relu(x)
        x = self.conv3_1(x)
        #x = F.relu(x)
        
        x = x.view(in_size, -1)
        
        x = F.relu(x)
        x = self.fc1(x)
        
        return F.log_softmax(x) 
        
class model_test_resnet_6(nn.Module):
#测试网络结构
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 10, kernel_size = 3, stride = 1, padding=1)
        self.conv2_1 = nn.Conv2d(10, 10, kernel_size = 3, stride = 2, padding=1)

        self.fc1 = nn.Linear(10*14*14, 10)
        #self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        x_1_o = x
        
        x = self.conv2_1(x)
        x = F.relu(x)
        x_2_o = x

        x = F.avg_pool2d(x_1_o, kernel_size=2) + x_2_o
        
        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        #x = F.relu(x)
        #x = self.fc2(x)
        
        return F.log_softmax(x)

class model_test_resnet_7(nn.Module):
# 搜索最高准确率 99.5%
    def __init__(self):
        super().__init__()
        # 输入 28*28
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding=1)
        # 输出 28*28
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding=1)
        # 输出 14*14
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding=1)
        # 输出 14*14
        self.fc1 = nn.Linear(64*14*14, 512*2)
        self.fc2 = nn.Linear(512*2, 10)
        
        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        x_1_o = x
        
        x = self.conv2_1(x)
        x = F.relu(x)
        x_2_o = x

        x = F.avg_pool2d(x_1_o, kernel_size=2) + x_2_o
        
        x = self.conv3_1(x)
        x = F.relu(x)
        x_3_o = x
        
        x = x_3_o + x_2_o
        #x = F.avg_pool2d(x_1_o, kernel_size=2) + x_3_o
        
        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return F.log_softmax(x)

class model_test_resnet_7_cat_dog_cls(nn.Module):
# 搜索最高准确率 99.5%
    def __init__(self):
        super().__init__()
        # 输入 28*28
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding=1)
        # 输出 28*28
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding=1)
        # 输出 14*14
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding=1)
        # 输出 14*14
        self.fc1 = nn.Linear(64*14*14, 512*2)
        self.fc2 = nn.Linear(512*2, 2)
        
        
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1_1(x)
        x = F.relu(x)
        x_1_o = x
        
        x = self.conv2_1(x)
        x = F.relu(x)
        x_2_o = x

        x = F.avg_pool2d(x_1_o, kernel_size=2) + x_2_o
        
        x = self.conv3_1(x)
        x = F.relu(x)
        x_3_o = x
        
        x = x_3_o + x_2_o
        #x = F.avg_pool2d(x_1_o, kernel_size=2) + x_3_o
        
        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return F.log_softmax(x)
        
        
class VGG10(nn.Module):

    def __init__(self):
        super().__init__()
        # 28*28
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        # 28*28
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 14*14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # 14*14
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 7*7
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # 7*7
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # 7*7
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.mp7 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # 4*4
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # 4*4
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # 4*4
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.mp10 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2*2
        self.conv11 = nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=1)
        # 2*2
        self.conv12 = nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=1)
        # 2*2
        self.conv13 = nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=1)
        self.mp13 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 1*1
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
    
        in_size = x.size(0)
        
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.mp2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.mp4(x)
        
        x = self.conv5(x)
        x = F.relu(x)
        
        x = self.conv6(x)
        x = F.relu(x)
        
        x = self.conv7(x)
        x = F.relu(x)
        x = self.mp7(x)
        
        x = self.conv8(x)
        x = F.relu(x)
        
        x = self.conv9(x)
        x = F.relu(x)
        
        x = self.conv10(x)
        x = F.relu(x)
        x = self.mp10(x)
        
        x = self.conv11(x)
        x = F.relu(x)
        
        x = self.conv12(x)
        x = F.relu(x)
        
        x = self.conv13(x)
        x = F.relu(x)
        x = self.mp13(x)
        
        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.log_softmax(x)
