import sys
import time
from os import listdir
from os.path import isfile, join
import vrep
import test_position
from PIL import Image
import array
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from test_position import NUMBER

import math

jointNum = 6

## globals
SRV_PORT = 19999
CAMERA = "Vision_sensor"
IMAGE_PLANE = "Plane0"
DIR_LIGHT0="light"
N_BASE_IMGS=50
CAPTURED_IMGS_PATH="./validation/"
testTarget1="testTarget1"


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
  errPos, position = vrep.simxGetObjectPosition(clientID, obj, -1, vrep.simx_opmode_oneshot_wait)
  errOrient, orientation = vrep.simxGetObjectOrientation(clientID, obj, -1, vrep.simx_opmode_oneshot_wait)

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
  newPose = newPose.reshape(2,3)

  errPos= vrep.simxSetObjectPosition(clientID, obj, -1, newPose[0,:], vrep.simx_opmode_oneshot_wait)
  errOrient= vrep.simxSetObjectOrientation(clientID, obj, -1, newPose[1,:], vrep.simx_opmode_oneshot_wait)


  if errPos :
    print("Failed to set position for object: {}, got error code: {}".format(obj, errPos))
  elif errOrient:
    print("Failed to set orientation for object: {}, got error code: {}".format(obj, errOrient))
  else:
    return newPose

def renderSensorImage(clientID, camera,fname,sleep_time):
  errRender, resolution, image = vrep.simxGetVisionSensorImage(clientID, camera, 0, vrep.simx_opmode_blocking)
  time.sleep(sleep_time)
  #errRender, resolution, image = vrep.simxGetVisionSensorImage(clientID, camera, 0, vrep.simx_opmode_blocking)
  errRender, resolution, image = vrep.simxGetVisionSensorImage(clientID, camera, 0, vrep.simx_opmode_blocking)

  if errRender == vrep.simx_return_ok:
      # image_byte_array = array.array('b', image)
      # image_buffer = Image.frombuffer("RGB", (resolution[0], resolution[1]), image_byte_array, "raw", "RGB", 0, 1)
      # img = np.asarray(image_buffer)
      img = np.array(image, dtype=np.uint8)
      img.resize([resolution[0], resolution[1], 3])
      img = cv2.flip(img, 0)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      #cv2.imwrite(fname, img)

      #cv2.imshow("Vision  Sensor", img)
  return img


RAD2DEG =180 / math.pi

if __name__=="__main__":
    clientID = connect(SRV_PORT, "Data generation started")
    ##print("clientID is: ",clientID)
    objects_names = [CAMERA, IMAGE_PLANE,testTarget1]
    #lights_names = [DIR_LIGHT0]

    newPose=test_position.getPositinAndOrientation()

    object_handles = getObjectsHandles(clientID, objects_names)
    print("object_handles CAMERA is: ",object_handles[CAMERA])

    lable=[]
    name = []

    for i in range(NUMBER):
        ##最后将test_target1的姿态设置为何link6_visible一样
        #newPose = generateCameraRandomPose(clientID, object_handles[testTarget1], initPose_target),,,
        setCameraRandomPose(clientID, object_handles[testTarget1], newPose[i])

        fname = CAPTURED_IMGS_PATH +"img" + '{0:06d}'.format(i) + ".jpg"
        print(fname)
        if i == 0:
            sleep_time = 0.05
            img=renderSensorImage(clientID,object_handles[CAMERA],fname,sleep_time)
        else:
            sleep_time = 0
            img=renderSensorImage(clientID, object_handles[CAMERA],fname,sleep_time)
        #cv2.imwrite(fname, img)
        pose =newPose[i].reshape(-1)
        print(pose)
        lable.append(pose)
        name.append(fname)
        time.sleep(0.001)
    #np.savetxt('imgs_name_validation.txt', name,fmt='%s')
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=8)
    #np.savetxt('lable_validation.txt', lable, fmt='%.06f')
