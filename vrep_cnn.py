import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from PIL import Image
import torch.nn as nn
from torchvision.transforms import transforms
from vrep_test import *

import math
import cv2
import vrep
from creat_data import *



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
    if err_code:
      print("Failed to get a handle for object: {}, got error code: {}".format( objects[obj_idx], err_code))
      break;
  return handles

def getLightHandles(clientID, lights):
  handles = {}
  for obj_idx in range(len(lights)):
    err_code, handles[lights[obj_idx]] = vrep.simxGetObjectHandle(clientID, lights[obj_idx], vrep.simx_opmode_blocking)
    if err_code:
      print("Failed to get a handle for object: {}, got error code: {}".format(lights[obj_idx], err_code))
      break;
  return handles

def setCameraInitialPose(clientID, obj):
  # print(obj)
  errPos, position = vrep.simxGetObjectPosition(clientID, obj, -1, vrep.simx_opmode_oneshot_wait)
  # print("1 error", err_code)
  # print("Position", position)

  errOrient, orientation = vrep.simxGetObjectOrientation(clientID, obj, -1, vrep.simx_opmode_oneshot_wait)
  # print("2 error", err_code)
  #print("Orientation", orientation)

  if errPos :
    print("Failed to get position for object: {}, got error code: {}".format(obj, errPos))
  elif errOrient:
    print("Failed to get orientation for object: {}, got error code: {}".format(obj, errOrient))
  else:
    return np.array([position, orientation])

def generateCameraRandomPose(clientID, obj, oldPose):
  # import matplotlib.pyplot as mlp
  print("old pose is :",oldPose)

  randPose = np.asarray(np.random.random([2, 3]))
  print("randPose is :",randPose)
  # print(np.shape(randPose))

  center = np.array([[0.01, 0.01, 0.01], np.deg2rad([-5, -5, -10])])
  variance = np.array([[0.01, 0.01, 0.01], np.deg2rad([5, 5, 10])])
  print("variance",variance)
  std = np.sqrt(variance)
  print("std is :",std)

  newPose = np.multiply(randPose, std) - std/2 + oldPose
  #print(np.shape(std))
  #print(oldPose)
  print("newpose shape is :",newPose)

  return newPose

def setCameraRandomPose(clientID, obj, newPose):
  # print(obj)

  errPos= vrep.simxSetObjectPosition(clientID, obj, -1, newPose[0,:], vrep.simx_opmode_oneshot_wait)
  # print("1 error", err_code)
  # print("Position", position)

  errOrient= vrep.simxSetObjectOrientation(clientID, obj, -1, newPose[1,:], vrep.simx_opmode_oneshot_wait)
  # print("2 error", err_code)
  # print("Orientation", orientation)

  if errPos :
    print("Failed to set position for object: {}, got error code: {}".format(obj, errPos))
  elif errOrient:
    print("Failed to set orientation for object: {}, got error code: {}".format(obj, errOrient))
  else:
    return newPose

transform = transforms.Compose([transforms.Resize(size=256,interpolation=2),transforms.ToTensor()])

baseName = 'UR5'
jointName = 'UR5_joint'
RAD2DEG =180 / math.pi

jointNum = 6

## globals
SRV_PORT = 19999
CAMERA = "Vision_sensor"
IMAGE_PLANE = "Plane0"

N_BASE_IMGS=50
CAPTURED_IMGS_PATH= 'vrep_cnn\\processing\\'

testTarget1="testTarget1"

time = 0
df = 2
dt =0.01
i = 0
err = []
v_all = []
w_all = []
time_intervals = []
traces = []




if __name__ == '__main__':
    clientID = connect(SRV_PORT, "Data generation started")
    objects_names = [CAMERA, IMAGE_PLANE, testTarget1]
    object_handles = getObjectsHandles(clientID, objects_names)

    ##init_pose初始位姿获取,应该编写欧拉角和对偶四元数、旋转矩阵之间的转换
    initPose = setCameraInitialPose(clientID,object_handles[testTarget1])#欧拉角
    print('initpose is:',initPose.reshape(-1))
    dq_AR = eulerTR2dualpq(initPose.reshape(-1))

    u_AR, theta_AR, R_AR, t_AR = dualq2uthetaRt(dq_AR)
    X_AR = np.vstack((np.hstack([R_AR, t_AR.reshape(t_AR.shape[0], 1)]), np.array([0, 0, 0, 1])))  ##
    #euler_AR = dualpq2eulerTR(dq_AR)
    euler_AR = initPose.reshape(2, 3)
    pointA = euler_AR[0].reshape(-1)
    print('euler_AR is:',euler_AR)


    ##desired pose，从真实标签中抽取一个。
    all_desired_pose = np.loadtxt('./lable.txt')
    index = np.random.randint(0, 4999)
    print('all_desired_pose[index] is',all_desired_pose[index])
    dq_BR = eulerTR2dualpq(all_desired_pose[index].reshape(-1))##欧拉角换对偶四元数

    u_BR, theta_BR, R_BR, t_BR = dualq2uthetaRt(dq_BR)
    X_BR = np.vstack((np.hstack([R_BR, t_BR.reshape(t_BR.shape[0], 1)]), np.array([0, 0, 0, 1])))
    euler_BR = dualpq2eulerTR(dq_BR)
    #euler_BR = all_desired_pose[index].reshape(2,3)
    pointB = euler_BR[0].reshape(-1)
    #euler_BR = all_desired_pose[index].reshape(2,3)
    print('euler_BR is:', euler_BR,index)

    # load model
    model = torch.load('model.kpl')
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  ##模型含有dropout，使用eval固定
    torch.no_grad()



    while time < df:

        error_dq_AB = muldualpq(conjdualqsimple(dq_BR), dq_AR)
        u_AB, theta_AB, R_AB, t_AB = dualq2uthetaRt(error_dq_AB)

        err.append(error_dq_AB)
        print('error_dq_AB is:',error_dq_AB)

        setCameraRandomPose(clientID, object_handles[testTarget1], euler_AR)

        ## Control Law
        lambdax = 5
        v = -lambdax * np.dot(R_AB.T, t_AB)
        w = -lambdax * theta_AB * u_AB
        control_law_AB = np.hstack([v, w])
        ####Convert Control Law
        T_BR = skew(t_BR)
        A = np.vstack([np.hstack([R_BR, np.dot(T_BR, R_BR)]), np.hstack([np.zeros((3, 3)), R_BR])])
        control_law_AR = np.dot(A, control_law_AB.T)

        v = control_law_AR[0:3]
        w = control_law_AR[3:6]
        v_all.append(v)
        w_all.append(w)

        theta = np.linalg.norm(w)

        if theta == 0:
            u = np.array([0, 0, 1])
        else:
            u = w / np.linalg.norm(w)

        ##当前图像获取与位姿预测
        fname = CAPTURED_IMGS_PATH + "img" + '{0:06d}'.format(i) + ".jpg"
        if i == 0:
            sleeptime1 = 0.05
            img = renderSensorImage(clientID,object_handles[CAMERA],fname,sleeptime1)
        else:
            sleeptime1 = 0
            img = renderSensorImage(clientID, object_handles[CAMERA], fname,sleeptime1)
        i = i + 1

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = transform(img).unsqueeze(0)
        img_ = img.to(device)

        #output = model(img_)

        #output = output.cuda().data.cpu().numpy().reshape(6,)
        #print('output',output)
        ####
        update_dq_AR = uthetat2dq(u, dt * theta, dt * v)
        dq_AR = muldualpq(update_dq_AR, dq_AR)

        euler_AR = dualpq2eulerTR(dq_AR)
        print('time is:',time)
        time = time + dt
        time_intervals.append(time)
        traces.append(euler_AR[0].reshape(-1))



    err = np.asarray(err)
    v_all = np.asarray(v_all)
    w_all = np.asarray(w_all)
    time_intervals = np.asarray(time_intervals)

###plot 3D the trajectory of the camera frame
traces1 = np.asarray(traces)


x = traces1[:,0]
y = traces1[:,1]
z = traces1[:,2]

fig = plt.figure()
ax = fig.gca(projection='3d')

# set figure information
ax.set_title("3D_Curve")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")



ax.scatter(pointA[0], pointA[1], pointA[2], c='g', marker='+')
ax.scatter(pointB[0], pointB[1], pointB[2], c='r', marker='^')




figure = ax.plot(x, y, z, c='r')


##velocity
fig = plt.figure()
plt.plot(time_intervals, v_all[:,0],c='r',label='v1')
plt.plot(time_intervals, v_all[:,1],c='b',label='v2')
plt.plot(time_intervals, v_all[:,2],c='g',label='v3')

##orientation

plt.plot(time_intervals, w_all[:,0],c='y',ls='-.',label='w1')
plt.plot(time_intervals, w_all[:,1],c='k',ls='-.',label='w2')
plt.plot(time_intervals, w_all[:,2],c='m',ls='-.',marker='o',ms=1,label='w3')
plt.legend() # 显示图例



plt.xlabel('time_intervals')

#plt.show()

###err
fig = plt.figure()
plt.plot(time_intervals, err[:,0],c='r',label='r1')
plt.plot(time_intervals, err[:,1],c='g',label='r2')
plt.plot(time_intervals, err[:,2],c='k',label='r3')
plt.plot(time_intervals, err[:,3],c='m',label='r4')

plt.plot(time_intervals, err[:,4],c='b',label='d1')
plt.plot(time_intervals, err[:,5],c='g',label='d2')
plt.plot(time_intervals, err[:,6],c='y',label='d3')
plt.plot(time_intervals, err[:,7],c='m',label='d4')

plt.legend() # 显示图例
plt.xlabel('time_intervals')



plt.show()

























