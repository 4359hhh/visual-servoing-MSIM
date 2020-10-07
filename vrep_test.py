import math
import numpy as np



def mulpq(p,q):#（四元数乘法）对的 没有问题
    s1 = p[0]
    v1 = p[1:4]  ###索引，多个元素，左闭右开

    s2 = q[0]
    v2 = q[1:4]
    a = s1 * s2 - np.dot(v1, v2)
    b = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    return np.hstack([a,b])

def muldualpq(p,q):##对偶四元数乘法
    p1 = p[0:4]
    p2 = p[4:8]

    q1 = q[0:4]
    q2 = q[4:8]
    return np.hstack([mulpq(p1,q1),mulpq( p1, q2 ) + mulpq( p2, q1 )])

def conjdualqsimple(x):#conjugate： 共轭（对偶四元数共轭）

    return np.hstack([x[0],-x[1:4],x[4],-x[5:8]])

def conjq(q):#四元数共轭

    return np.hstack([q[0],-q[1:4]])

def dualq2uthetaRt(x):##对偶四元数到轴角与旋转矩阵
    qrr = x[0:4]
    qrr = qrr / np.sum(qrr ** 2)
    qdd = x[4:8]

    t = 2 * mulpq(qdd, conjq(qrr))
    t = t[1:4]##四元数求轴角中的平移t=（nx,ny,nz）

    theta = 2*math.acos(qrr[0])##四元数求轴角中的theta
    #theta = 2 * math.asin(qrr[0])  ##四元数求轴角中的theta

    if theta != 0:
        u = qrr[1:4] /math.sin(theta/2)
    else:
        u = np.array([0, 0, 1])

    skw = np.array([[0, -u[2], u[1]],
                    [u[2], 0, -u[0]],
                    [-u[1], u[0], 0]])

    #print('skw is:', skw)

    R = np.eye(3) + math.sin(theta) * skw + np.dot(skw, skw) * (1 - math.cos(theta))
    #R = np.eye(3) + theta/(math.pi*2) * skw + np.dot(skw, skw) * (1 - theta/(2*math.pi))

    return u, theta, R, t

def skew(x):
    s = np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])
    return s

####轴角到对偶四元数
def uthetat2dq(axis,theta,translate):

    q_rot = np.hstack([math.cos(theta/2), math.sin(theta/2)* axis])

    translate = np.hstack([0, translate])

    q_rot = q_rot / np.sum(q_rot ** 2)
    q_tr = 0.5*mulpq(translate,q_rot)

    #print('q_rot is :', q_rot)
    #q_tr = q_tr / np.sum(q_tr**2)

    x = np.hstack([q_rot, q_tr])

    return x

RAD2DEG = 180.0/math.pi
DEG2RAD = math.pi / 180.0
def eulerTR2dualpq(x):##x = []6个,欧拉角：前平移，后旋转

    q_rot = np.array([math.cos(x[3] / 2) * math.cos(x[4] / 2) * math.cos(x[5] / 2) + math.sin(x[3] / 2) * math.sin(x[4] / 2) * math.sin(x[5] / 2),
                   math.sin(x[3] / 2) * math.cos(x[4] / 2) * math.cos(x[5] / 2) - math.cos(x[3] / 2) * math.sin(x[4] / 2) * math.sin(x[5] / 2),
                   math.cos(x[3] / 2) * math.sin(x[4] / 2) * math.cos(x[5] / 2) - math.sin(x[3] / 2) * math.cos(x[4] / 2) * math.sin(x[5] / 2),
                   math.cos(x[3] / 2) * math.cos(x[4] / 2) * math.sin(x[5] / 2) - math.sin(x[3] / 2) * math.sin(x[4] / 2) * math.cos(x[5] / 2)
                   ])
    q_rot = q_rot / np.sum(q_rot ** 2)

    translate = np.hstack([0, x[0:3]])
    q_tr = 0.5*mulpq(translate,q_rot)

    return np.hstack([q_rot, q_tr])#前旋转，后平移



def q2euler(q):#computes the Euler angles from the unit
    '''
    R = Rquat(q)
    if abs(R[2][0]) > 1.0:
        print('solution is singular for theta = +- 90 degrees')

    phi = math.atan2(R[2][1], R[2][2])
    theta = -math.asin(R[2][0])
    psi = math.atan2(R[1][0], R[0][0])
    '''
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    phi = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    theta = math.asin(2 * (w * y - z * x))
    psi = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([phi,theta,psi])


def pq2euler(y):
    euler = np.array([ abs(math.asin(-2 * ( y[0] * y[2] + y[3] * y[1] )) ),
                          -math.atan(2 * (y[1] * y[2] - y[0] * y[3]) / ( y[0] * y[0] + y[1] * y[1]- y[2] * y[2] - y[3] * y[3])  ),
                       abs(math.atan(2 * (y[2] * y[3] - y[0] * y[1]) / ( y[0] * y[0] - y[1] * y[1] - y[2] * y[2] + y[3] * y[3])  ))
                       ])

    return euler

def dualpq2eulerTR(x):
    qrr = x[0:4]
    qtt = x[4:8]
    euler = q2euler(qrr)##pq2euler(qrr)
    #euler = quat.quat2seq(qrr, seq='Euler')

    t = 2*mulpq(qtt,conjq(qrr))
    t = t[1:4]

    y = np.hstack([t, euler])
    return y.reshape(2,3)

'''

def euler2pq(x):#euler = []
    pq = np.array([math.cos(x[0] / 2) * math.cos(x[1] / 2) * math.cos(x[2] / 2) - math.sin(x[0] / 2) * math.sin(x[1] / 2) * math.sin(x[2] / 2),
                   math.sin(x[2] / 2) * math.cos(x[1] / 2) * math.cos(x[0] / 2) + math.cos(x[2] / 2) * math.sin(x[1] / 2) * math.sin(x[0] / 2),
                   math.cos(x[1] / 2) * math.sin(x[0] / 2) * math.cos(x[2] / 2) - math.sin(x[1] / 2) * math.cos(x[0] / 2) * math.sin(x[2] / 2),
                   math.cos(x[0] / 2) * math.cos(x[2] / 2) * math.sin(x[1] / 2) + math.sin(x[0] / 2) * math.sin(x[2] / 2) * math.cos(x[1] / 2)
                   ])
    return pq
'''
def Rzyx(x):
    cphi = math.cos(x[0])
    sphi = math.sin(x[0])
    cth = math.cos(x[1])
    sth = math.sin(x[1])
    cpsi = math.cos(x[2])
    spsi = math.sin(x[2])
    R  = np.array([
                   [cpsi*cth , -spsi*cphi+cpsi*sth*sphi,  spsi*sphi+cpsi*cphi*sth],
                   [spsi*cth , cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi],
                   [-sth,cth*sphi,cth*cphi ]
                    ])
    return R
def euler2q(x):
    R = np.vstack((np.hstack([Rzyx(x), np.array([[0], [0], [0]])]), np.array([0, 0, 0, np.trace(Rzyx(x))])))
    Rmax, i = max(np.diag(R)), np.argmax(np.diag(R))

    p_i = np.sqrt(1 + 2 * R[i][i] - R[3][3])
    if i == 0:
        p1 = p_i
        p2 = (R[1][0] + R[0][1]) / p_i
        p3 = (R[0][2] + R[2][0]) / p_i
        p4 = (R[2][1] - R[1][2]) / p_i
    elif i == 1:
        p1 = (R[1][0] + R[0][1]) / p_i
        p2 = p_i
        p3 = (R[2][1] + R[1][2]) / p_i
        p4 = (R[0][2] - R[2][0]) / p_i
    elif i == 2:
        p1 = (R[0][2] + R[2][0]) / p_i
        p2 = (R[2][1] + R[1][2]) / p_i
        p3 = p_i
        p4 = (R[1][0] - R[0][1]) / p_i
    else:
        p1 = (R[2][1] - R[1][2]) / p_i
        p2 = (R[0][2] - R[2][0]) / p_i
        p3 = (R[1][0] - R[0][1]) / p_i
        p4 = p_i
    q = 0.5 * np.array([p1, p2, p3, p4])
    pq = q / (np.sum(q.T * q))
    return pq

def Rquat(q):#computes the quaternion rotation matrix R in SO(3)
    tol = 1e-3
    if abs(np.linalg.norm(q) - 1) > tol:
        print('norm(q) must be equal aa to 1')

    eta = q[3]
    eps = q[0:3]

    S = np.array([[0, -eps[2], eps[1]],
                    [eps[2], 0, -eps[0]],
                    [-eps[1], eps[0], 0]])

    R = np.eye(3) + 2 * eta * S + 2 * np.dot(S,S)

    return R


'''
for i in range(200):
    all_desired_pose = np.loadtxt('./lable.txt')
    index = np.random.randint(0, 4999)
    a = all_desired_pose[index].reshape(-1)
    print('all_desired_pose[index] is', a[3:6])
    q_BR = euler2q(a[3:6])

    euler= q2euler(q_BR)

    print('\033[1;31m' + 'q2euler       is', euler.reshape(-1),'\033[0m')
'''



#RAD2DEG = 180.0/math.pi
#DEG2RAD = math.pi / 180.0

#init_euler = np.array([1, 1, 1, 0, 1, 0])#欧拉角+(平移，旋转)
#init_dq = eulerTR2dualpq(init_euler)
#u_AR, theta_AR, R_AR, t_AR = dualq2uthetaRt(init_dq)
#print(init_euler)

#euler = dualpq2eulerTR(init_dq)
#print(euler)










"""
#RAD2DEG =180 / math.pi
RAD2DEG = math.pi / 180
##初始位姿Initial Pose
##轴、角度、位置
initial_axis = np.array([0,0,1])
initial_theta = 90*RAD2DEG
initial_translate = np.array([0,1,0])


q_rot = np.hstack([math.cos(initial_theta / 2),math.sin(initial_theta / 2)*initial_axis])
#print('qrot is :',q_rot)
translate =np.hstack( [0, initial_translate])
#print('translate is:',translate)


s1 = translate[0]
v1 = translate[1:4]###索引，多个元素，左闭右开


s2 = q_rot[0]
v2 = q_rot[1:4]
#print('s2 is:',s2)
#print('v2 is:',v2)

#print('s1 is:',s1)
#print('v1 is:',v1)
#print('(v1)*(v2) is:',np.dot(v1.T,v2))
a = s1*s2-np.dot(v1.T,v2)##s1*s2 - v1'*v2
#print('s1*s2-np.dot(v1.T,v2)',a)
#q_tr = 0.5*mulpq(translate,q_rot)
##s1*v2 + s2*v1 + cross( v1, v2 )

b = s1*v2 + s2*v1 +np.cross(v1,v2)
q_tr = 0.5*np.hstack([a,b])
#print('b is:',b)
#print('q_tr is:',q_tr)

x = np.hstack([q_rot,q_tr])
print('x is :',x)####输出值

def mulpq(p,q):
    s1 = translate[0]
    v1 = translate[1:4]  ###索引，多个元素，左闭右开

    s2 = q_rot[0]
    v2 = q_rot[1:4]
    a = s1 * s2 - np.dot(v1.T, v2)
    b = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    return np.hstack([a,b])
"""


"""

qrr = x[0:4]
qdd = x[4:9]

i = conjq(qrr)
j = mulpq(qdd,conjq(qrr))
t1 = 2*mulpq(qdd,conjq(qrr))
t = t1[1:4]

theta = 2*math.acos(qrr[0])

print('theta is:',theta)
if theta != 0:
    u = qrr[1:4] / math.sin(theta/2)
    print('u is:',u)
else:
    u = np.array([0,0,1])
    print(u)

skw = np.array([[0,-u[2],u[1]],
                [u[2],0,-u[0]],
                [-u[1],u[0],0]])

print('skw is:',skw)


R = np.eye(3) + math.sin(theta)*skw +np.dot(skw,skw)*(1-math.cos(theta))
print('R is:',R)
#return [ u, theta, R, t ]----u_AR, theta_AR, R_AR, t_AR

#X_AR = np.vstack((np.hstack(R,t),np.array([0,0,0,1])))

print('R+t is:',np.hstack([R,t.reshape(t.shape[0],1)]))

X_AR = np.vstack((np.hstack([R,t.reshape(t.shape[0],1)]),np.array([0,0,0,1])))

print('X_AR',X_AR)

"""



