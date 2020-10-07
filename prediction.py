import numpy as np
import vrep
import cv2
import time
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn
from skimage.measure import compare_ssim as ssim
#import sim
from tensorboardX import SummaryWriter
import torchvision

## globals
SRV_PORT = 19999
CAMERA = "Vision_sensor"
IMAGE_PLANE = "Plane0"
DIR_LIGHT0="light"
N_BASE_IMGS=50
CAPTURED_IMGS_PATH="./validation/"
testTarget1="testTarget1"
objects_names = [CAMERA, IMAGE_PLANE, testTarget1]

batchsize = 1
torch.set_printoptions(precision=8)





#root =os.getcwd()+ '/capture/'#数据集的地址
#-----------ready the dataset--------------
def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
###----construct function with defualt parameters
    def __init__(self,image_root,label_root,transform=None,target_transform=None,loader=default_loader):

        super(MyDataset,self).__init__()
        all_img_name= []
        all_label = []
        fi = open(image_root, 'r')
        for name_img_line in fi:
            name_img_line = name_img_line.strip('\n')
            name_img_line = name_img_line.rstrip('\n')
            all_img_name.append(name_img_line)

        fl = open(label_root, 'r')
        for label_line in fl:
            label_line = label_line.strip('\n')
            label_line = label_line.rstrip('\n')
            label_line = label_line.split()
            all_label.append(label_line)

        self.all_img_name=all_img_name
        self.all_label = all_label

        self.transform = transform
        self.target_transform = target_transform
        #self.label = []
        #self.data = []
        self.loader = loader

    def __getitem__(self, index):
        label = self.all_label[index]
        #print('index is:',index)
        label = np.array([i for i in label], dtype=np.float16)

        label = torch.Tensor(label)
        #label = transforms.Normalize([],[])

        #print('label is :',label)

        fn =self.all_img_name[index]
        image = self.loader(fn)
        if self.transform is not None:
            image = self.transform(image)

        #if self.target_transform is not None:
            #label = self.target_transform(label)

        #label = Variable(label)
        #label = array.array(label)
        #label=torch.Tensor(label)

        return image,label

    def __len__(self):
        return len(self.all_img_name)

class sp_noise(object):
    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        Nd = self.density

        Sd = 1 - Nd

        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])
        mask = np.repeat(mask, c, axis=2)
        img[mask == 0] = 0
        img[mask == 1] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')

        return img

label_validation_root = 'lable_validation.txt'
image_validation_root = 'imgs_name_validation.txt'#imgs_name_validation.txt'#
test_data = MyDataset(image_root=image_validation_root, label_root=label_validation_root, transform=transforms.Compose([
                                                                                                    transforms.Resize(size=256,interpolation=2),
                                                                                                    #sp_noise(0.01),
                                                                                                    transforms.ToTensor(),
                                                                                                    #transforms.Normalize(mean =(0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                                                                                                    transforms.RandomErasing(p=0.5,scale=(0.015,0.02),ratio=(0.2,0.6),value=(60,0,0))
                                                                                                    ]))
test_loader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=False, num_workers=8,pin_memory=True)

def connect(port, message):
  # connect to server
  vrep.simxFinish(-1)  # just in case, close all opened connections
  clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # start a connection
  if clientID != -1:
    print("Connected to remote API server")
    print(message)
  else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")
  return clientID
def getObjectsHandles(clientID, objects):
  handles = {}
  for obj_idx in range(len(objects)):
    err_code, handles[objects[obj_idx]] = vrep.simxGetObjectHandle(clientID, objects[obj_idx], vrep.simx_opmode_blocking)
    print('err_code is :',err_code)
    if err_code:
      print("Failed to get a handle for object: {}, got error code: {}".format( objects[obj_idx], err_code))
      break;
  return handles
def setCameraRandomPose(clientID, obj, newPose):

    errPos= vrep.simxSetObjectPosition(clientID, obj, -1, newPose[0,:], vrep.simx_opmode_oneshot_wait)

    errOrient= vrep.simxSetObjectOrientation(clientID, obj, -1, newPose[1,:], vrep.simx_opmode_oneshot_wait)

    if errPos :
        print("Failed to set position for object: {}, got error code: {}".format(obj, errPos))
    elif errOrient:
        print("Failed to set orientation for object: {}, got error code: {}".format(obj, errOrient))
    else:pass
def renderSensorImage(clientID, camera,sleep_time):
  #errRender, resolution, image = vrep.simxGetVisionSensorImage(clientID, camera, 0, vrep.simx_opmode_streaming)
  errRender, resolution, image = vrep.simxGetVisionSensorImage(clientID, camera, 0, vrep.simx_opmode_blocking)
  #print('errRender1:', errRender)
  time.sleep(sleep_time)
  #errRender, resolution, image = vrep.simxGetVisionSensorImage(clientID, camera, 0, vrep.simx_opmode_buffer)
  errRender, resolution, image = vrep.simxGetVisionSensorImage(clientID, camera, 0, vrep.simx_opmode_blocking)
  #print('errRender:',errRender)
  #print('vrep.simx_return_ok is :',vrep.simx_return_ok)
  if errRender == vrep.simx_return_ok:
      img = np.array(image, dtype=np.uint8)
      img.resize([resolution[0], resolution[1], 3])
      img = cv2.flip(img, 0)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      img = cv2.resize(img,(256,256))

  return img

def image_switch(images):
    np_images = images.cuda().data.cpu().numpy()

    np_images *= 255
    np_images = np_images.astype(np.uint8)#3x512x512

    # cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE )

    np_images1 = np.swapaxes(np_images, 0, 1)  # 512x3x512
    np_images2 = np.swapaxes(np_images1, 1, 2) #512x512x3

    #np_images2 = np.swapaxes(np_images2, 0, 1)

    b, g, r = cv2.split(np_images2)
    img_switch = cv2.merge([r, g, b])

    return img_switch
def img_get(pose,i):
    newPose = pose.cuda().data.cpu().numpy().reshape(2, 3)
    #print('newpose is:', object_handles[testTarget1])
    setCameraRandomPose(clientID, object_handles[testTarget1], newPose)
    sleep_time = 0.05
    #print('sleep time is:',sleep_time)
    render_image = renderSensorImage(clientID, object_handles[CAMERA], sleep_time)

    return render_image



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('torch.cuda.is_available() is:', torch.cuda.is_available())
###-----------model define-------resnet
'''
def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()
'''
#model_vrep0730.kpl
model = torch.load('model_vrep0730.kpl')#model_sim_0809.pkl
#model.apply(fix_bn)
model = model.to(device)
model.eval()

print(model)
###-------define LOSS and optimizer
from torch.optim import lr_scheduler
learning_rate = 0.01
criterion = nn.L1Loss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.99)



total_validation_step = len(test_loader)

import torch.nn.functional as F
image_size = 196608#256x256x3

from math import exp
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])

    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_pytorch(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:

        return ret, cs
    return ret
def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=True):
    #device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim_pytorch(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        #print('nan sim', torch.isnan(sim))
        #print('nan cs', torch.isnan(cs))


        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)
    #print('nan ssims', torch.isnan(ssims))
    #print('nan mcs', torch.isnan(mcs))
    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights
    #print('nan pow1', torch.isnan(pow1))
    #print('nan pow2', torch.isnan(pow2))
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    #print('nan output', torch.isnan(output))
    return output

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



if __name__ == '__main__':
    clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    object_handles = getObjectsHandles(clientID, objects_names)
    print('object_handles is:', object_handles)
    with torch.no_grad():

        ssim = 0
        #loss = []
        predict_label=[]
        all_label = []
        for i, (images, label) in enumerate(test_loader):
            images = images.to(device)
            label = label.to(device)

            out_put = model(images)
            '''
            for j in range(batchsize):
                ###获得当前batchsize的图像img
                img_switch = image_switch(images[j])
                #img_switch = torch.tensor(img_switch).cuda().data.cpu().numpy().view(-1,image_size)#2
                ##获得预测位姿下的图像
                img_rendered = img_get(out_put[j],j)
                #img_rendered = torch.tensor(img_rendered).cuda().data.cpu().numpy().view(-1,image_size)


                hmerge = np.hstack((img_switch, img_rendered))
                cv2.imwrite('./validation_model_0814ssim/' + str(i)+str(j) + '.jpg', hmerge)

                img_rendered = to_tensor(img_rendered).unsqueeze(0).type(torch.FloatTensor)
                img_switch = to_tensor(img_switch).unsqueeze(0).type(torch.FloatTensor)

                img_rendered.requires_grad = True
                img_switch.requires_grad = True

                img_rendered = img_rendered.to(device)
                img_switch = img_switch.to(device)


                test = MSSSIM().cuda()
                loss_msssim = 1- test(img_switch, img_rendered)

                ssim = ssim + loss_msssim

                print('{}/{} ssim is:{}'.format(i,j,ssim.item()))
            '''
            predict_label.append(out_put.cuda().data.cpu().numpy().reshape(6,))
            #all_label.append(label.cuda().data.cpu().numpy().reshape(6,))
            #loss.append(loss_msssim.item())
    np.savetxt('predicte_sim_0820_0.015.txt',predict_label,fmt='%s')
    #np.savetxt('SSIM_L1_loss.txt',ssim, fmt='%.06f')


    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax = Axes3D(fig)

    predict_label = np.array(predict_label)
    ax.scatter(predict_label[:,0], predict_label[:,1], predict_label[:,2], c='r', marker='o',s=30)

    all_label = np.array(all_label)
    ax.scatter(all_label[:, 0], all_label[:, 1], all_label[:, 2], c='b', marker='^',s=10)
    plt.show()
    '''




