
import matplotlib.pyplot as plt
from datagen import *
from mpl_toolkits.mplot3d import Axes3D

from vrep_test import *
DEG2RAD = math.pi / 180.0

##初始位姿Initial Pose
##轴、角度、位置
initial_axis = np.array([0,0,1])
initial_theta = 2.568585
initial_translate = np.array([0.7244065,	0.1302509,	1.6483233])
##计算初始initial对应初始的对偶四元数位姿dual quaternion
dq_AR= uthetat2dq(initial_axis, initial_theta, initial_translate)
euler_AR = dualpq2eulerTR(dq_AR)
print('euler_AR is:',euler_AR)
#euler_AR = np.array([[ 5.14829874e-01 ,7.15255737e-07  ,1.76500010e+00],[ 1.62920680e-07 ,-1.57079633e+00 , 1.57079628e+00]])
#print('euler_AR is:',euler_AR)
pointA = euler_AR[0].reshape(-1)
#dq_AR = eulerTR2dualpq(euler_AR.reshape(-1))
#% C-space % Cartesian space pose旋转矩阵+平移矩阵的齐次
u_AR, theta_AR, R_AR, t_AR = dualq2uthetaRt(dq_AR)
X_AR = np.vstack((np.hstack([R_AR,t_AR.reshape(t_AR.shape[0],1)]),np.array([0,0,0,1])))##


##期望位姿Desired Pose
desired_axis = np.array([0,0,1])
desired_theta = 1.0459538
desired_translate = np.array([0.39594293,	-0.2740591,	1.9317055])
##计算期望位姿Desired Pose对应初始的四元数位姿dual quaternion
dq_BR= uthetat2dq(desired_axis,desired_theta,desired_translate)
euler_BR = dualpq2eulerTR(dq_BR)#
print('euler_BR is:',euler_BR)
#euler_BR = np.array([[ 0.56454675 ,-0.22282828 , 2.03164851],[ 0.04260418 , 1.47954654 , 1.41468078]])
#print('euler_AR is:',euler_BR)
pointB = euler_BR[0].reshape(-1)
dq_BR = eulerTR2dualpq(euler_BR.reshape(-1))
### C-space % Cartesian space pose
u_BR, theta_BR, R_BR, t_BR = dualq2uthetaRt(dq_BR)
X_BR = np.vstack((np.hstack([R_BR,t_BR.reshape(t_BR.shape[0],1)]),np.array([0,0,0,1])))

### define something
time = 0  #% current time
tf = 2 #% final time
dt = 0.01 #% control sampling time



err = []
v_all = []
w_all = []
time_intervals = []
traces = []
#print('muldualpq( conjdualqsimple( dq_BR ),  dq_AR )',muldualpq( conjdualqsimple( dq_BR ),  dq_AR ))


clientID = connect(SRV_PORT, "Data generation started")
objects_names = [CAMERA, IMAGE_PLANE,testTarget1]
object_handles = getObjectsHandles(clientID, objects_names)
print("object_handles CAMERA is: ",object_handles[CAMERA])

i = 0



from math import exp
import torch
import torch.nn.functional as F



from skimage.measure import compare_ssim

while time < tf:
    error_dq_AB = muldualpq( conjdualqsimple( dq_BR ),  dq_AR )
    ###值越接近[0 0 0 1],那么四元数越接近，此处定义为[1 0 0 0 0 0 0 ],所以误差上显示是没有错的！！！！！
    euler_AR = dualpq2eulerTR(dq_AR)
    conver_euler_AR = euler_AR.reshape(2,3)
    conver_euler_AR[1][1] = -1.570796

    setCameraRandomPose(clientID, object_handles[testTarget1], conver_euler_AR)
    IMGS_PATH = './process/'
    img_name = IMGS_PATH + "img" + str(i) + ".jpg"
    if time == 0:
        sleep_time = 0.05
        img = renderSensorImage(clientID, object_handles[CAMERA], img_name,  sleep_time)
        initial_img = img
    else:
        sleep_time = 0
        img = renderSensorImage(clientID, object_handles[CAMERA],img_name,  sleep_time)

    error_path = './error_img/'
    error_img = error_path + "error_img" + str(i) + ".jpg"

    cv2.imshow('test',cv2.subtract(initial_img, img))
    cv2.imwrite(error_img, cv2.subtract(initial_img, img))
    cv2.waitKey(1)
    cv2.imwrite(img_name, img)
    i = i+1



    u_AB, theta_AB, R_AB, t_AB = dualq2uthetaRt(error_dq_AB)
    ##theta_AB越来越小 对的
    err.append(error_dq_AB)
    ## Control Law
    lambdax = 5
    v = -lambdax * np.dot(R_AB.T,t_AB)
    w = -lambdax * theta_AB *u_AB
    control_law_AB = np.hstack([v, w])

    ####Convert Control Law
    T_BR = skew(t_BR)
    A = np.vstack([np.hstack([R_BR, np.dot(T_BR,R_BR)]),np.hstack([np.zeros((3,3)), R_BR])])
    control_law_AR = np.dot(A ,control_law_AB.T)


    v = control_law_AR[0:3]
    w = control_law_AR[3:6]
    v_all.append(v)
    w_all.append(w)

    theta = np.linalg.norm(w)
    #print('theta is:',theta)

    if theta == 0:
        u =np.array([0,0,1])
    else:
        u = w / np.linalg.norm(w)
    #print('u is:',u)
    update_dq_AR= uthetat2dq( u, dt*theta, dt*v )
    dq_AR = muldualpq(update_dq_AR, dq_AR)

    u_AR, theta_AR, R_AR, t_AR1 = dualq2uthetaRt( dq_AR)
    X_AR = np.vstack((np.hstack([R_AR, t_AR1.reshape(t_AR1.shape[0], 1)]), np.array([0, 0, 0, 1])))



    time = time + dt
    time_intervals.append(time)
    traces.append(euler_AR[0].reshape(-1))

err= np.asarray(err)
v_all= np.asarray(v_all)
w_all =np.asarray(w_all)
time_intervals =np.asarray(time_intervals)




###plot 3D the trajectory of the camera frame
traces1 = np.asarray(traces)


x = traces1[:,0]
y = traces1[:,1]
z = traces1[:,2]
np.savetxt('PBVS_trace.txt', traces1, fmt='%.06f')

fig = plt.figure()
ax = fig.gca(projection='3d')

# set figure information
ax.set_title("3D_Curve")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")



ax.scatter(pointA[0], pointA[1], pointA[2], c='g', marker='+')
ax.scatter(pointB[0], pointB[1], pointB[2], c='r', marker='^')




figure = ax.plot(x, y, z, c='b')


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


np.savetxt('PBVS_v_all.txt', v_all, fmt='%.06f')
np.savetxt('PBVS_w_all.txt', w_all, fmt='%.06f')
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

np.savetxt('PBVS_err.txt', err, fmt='%.06f')

plt.legend() # 显示图例
plt.xlabel('time_intervals')



plt.show()
























