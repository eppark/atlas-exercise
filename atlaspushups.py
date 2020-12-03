#!/usr/bin/env python3
#
#   bettermotion.py
#
#   Better version, with fixed time stepping...
#
#   Create a motion by continually sending joint values
#
#   Publish:   /joint_states      sensor_msgs/JointState
#

import rospy
import numpy as np
from numpy.linalg import pinv
from sensor_msgs.msg   import JointState
from hw6code.kinematics import Kinematics
import tf as transforms

#
#  Necessariy Rotation Matrices
#
def Rx(theta):
    return np.array([[ 1, 0            , 0            ],
                     [ 0, np.cos(theta),-np.sin(theta)],
                     [ 0, np.sin(theta), np.cos(theta)]])
                     
def Ry(theta):
    return np.array([[  np.cos(theta), 0, np.sin(theta)],
                     [  0            , 1, 0            ],
                     [ -np.sin(theta), 0, np.cos(theta)]])
                     
                     
def Rz(theta):
    return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                     [ np.sin(theta) , np.cos(theta), 0 ],
                     [ 0            , 0             , 1 ]])

#
#  Inverse Kinematics function
#
def ikin(theta0, t, dt, kinLeftFoot, kinRightFoot, kinLeftHand, kinRightHand):
    leftFootIdx = [24, 25, 26, 27, 28, 29, 0, 1, 2]
    rightFootIdx = [30, 31, 32, 33, 34, 35, 0, 1, 2]
    leftHandIdx = [3, 4, 5, 6, 7, 8, 9]
    rightHandIdx = [13, 14, 15, 16, 17, 18, 19]

    # Allocate the numpy variables for joint position, tip position,
    # tip orientation, and Jacobian for the left foot
    qLeftFoot = np.atleast_2d(theta0[leftFootIdx, 0]).T
    xtipLeftFoot = np.zeros((3,1))
    rtipLeftFoot = np.identity(3)
    JLeftFoot = np.zeros((6,kinLeftFoot.dofs()))
    
    # Allocate the numpy variables for joint position, tip position,
    # tip orientation, and Jacobian for the right foot
    qRightFoot = np.atleast_2d(theta0[rightFootIdx, 0]).T
    xtipRightFoot = np.zeros((3,1))
    rtipRightFoot = np.identity(3)
    JRightFoot = np.zeros((6,kinRightFoot.dofs()))
    
    # Allocate the numpy variables for joint position, tip position,
    # tip orientation, and Jacobian for the left hand
    qLeftHand = np.atleast_2d(theta0[leftHandIdx, 0]).T
    xtipLeftHand = np.zeros((3,1))
    rtipLeftHand = np.identity(3)
    JLeftHand = np.zeros((6,kinLeftHand.dofs()))
    
    # Allocate the numpy variables for joint position, tip position,
    # tip orientation, and Jacobian for the right hand
    qRightHand = np.atleast_2d(theta0[rightHandIdx, 0]).T
    xtipRightHand = np.zeros((3,1))
    rtipRightHand = np.identity(3)
    JRightHand = np.zeros((6,kinRightHand.dofs()))

    # Calculate s
    if t < 1:
        s = 3 * t**2 - 2 * t**3
        sdot = 6 * t - 6 * t**2
    elif t < 2:
        s = -4 + 12 * t - 9 * t**2 + 2 * t**3
        sdot = 12 - 18 * t + 6 * t**2
        
    # Calculate position and velocity for utorso
    x = np.array([[0.89703+0.2*s], [0.0], [0.7415783-0.4*s]])
    xdot = np.array([[0.2*sdot], [0.0], [-0.4*sdot]])

    startR = np.array([[0.5401808, 0.0, 0.8415490], [0.0, 1.0, 0.0], [-0.8415490, 0.0, 0.5401808]])
    R = Ry(np.pi/8 * s) @ startR
    omega = np.array([[0.0], [np.pi/8 * sdot], [0.0]])

    # Calculate rotation for the left and right respective foot frames
    LeftFootwrtWorld = np.array([[0.0059488, 0.0, 0.9999823], [0.0, 1.0, 0.0], [-0.9999823, 0.0, 0.0059488]])
    RightFootwrtWorld = np.array([[0.0059488, 0.0, -0.9999823], [0.0, 1.0, 0.0], [0.9999823, 0.0, 0.0059488]])
    rLeftFoot = (LeftFootwrtWorld).transpose() @ R
    rRightFoot = (RightFootwrtWorld).transpose() @ R
    omegaLeftFoot = (LeftFootwrtWorld).transpose() @ omega
    omegaRightFoot = (RightFootwrtWorld).transpose() @ omega

    # Calculate position and velocity for the left and right respective foot frames
    xLeftFoot = (LeftFootwrtWorld).transpose() @ (x + np.array([[0.0], [0.1115], [0.0]]))
    xRightFoot = (RightFootwrtWorld).transpose() @ (x - np.array([[0.0], [0.1115], [0.0]]))
    xdotLeftFoot = (LeftFootwrtWorld).transpose() @ np.copy(xdot)
    xdotRightFoot = (RightFootwrtWorld).transpose() @ np.copy(xdot)

    # Calculate rotation for the left and right respective hand frames
    LeftHandwrtWorld = np.array([[-0.9810139, 0.0824245, -0.1755504], [0.1925838, 0.5207862, -0.8316810], [0.0228733, -0.8496988, -0.5267721]])
    RightHandwrtWorld = np.array([[-0.8520682, 0.3215352, -0.4130313], [-0.1874464, 0.5493122, 0.8143218], [0.4887162, 0.7712789, -0.4077809]])

    rLeftHand = (LeftHandwrtWorld).transpose() @ R
    rRightHand = (RightHandwrtWorld).transpose() @ R
    omegaLeftHand = (LeftHandwrtWorld).transpose() @ omega
    omegaRightHand = (RightHandwrtWorld).transpose() @ omega

    # Calculate the position and velocity for the left and right respective hand frames
    xLeftHand = (LeftHandwrtWorld).transpose() @ (x + np.array([[-1.3167], [0.564], [0.4]]))
    xRightHand = (RightHandwrtWorld).transpose() @ (x + np.array([[-1.3167], [-0.564], [0.4]]))
    xdotLeftHand = (LeftHandwrtWorld).transpose() @ np.copy(xdot)
    xdotRightHand = (RightHandwrtWorld).transpose() @ np.copy(xdot)
    
    # Get the positions and orientations for the left and right foot frames and hand frames
    kinLeftFoot.fkin(qLeftFoot, xtipLeftFoot, rtipLeftFoot)
    kinRightFoot.fkin(qRightFoot, xtipRightFoot, rtipRightFoot)
    kinLeftHand.fkin(qLeftHand, xtipLeftHand, rtipLeftHand)
    kinRightHand.fkin(qRightHand, xtipRightHand, rtipRightHand)
    
    # Get the Jacobian mapping to position and orientation for the foot and hand frames
    kinLeftFoot.Jac(qLeftFoot, JLeftFoot)
    kinRightFoot.Jac(qRightFoot, JRightFoot)
    kinLeftHand.Jac(qLeftHand, JLeftHand)
    kinRightHand.Jac(qRightHand, JRightHand)
    
    # Get the psuedoinverse of the Jacobian
    rpinvJLeftFoot = pinv(JLeftFoot)
    rpinvJRightFoot = pinv(JRightFoot)
    rpinvJLeftHand = pinv(JLeftHand)
    rpinvJRightHand = pinv(JRightHand)

    # Find the left foot error
    posError = xLeftFoot - xtipLeftFoot
    rotError = 0.5 * (np.cross(rtipLeftFoot[:,0], rLeftFoot[:,0]) + np.cross(rtipLeftFoot[:,1], rLeftFoot[:,1]) + np.cross(rtipLeftFoot[:,2], rLeftFoot[:,2]))
    eLeftFoot = np.atleast_2d(np.append(posError, rotError)).T
    
    # Find the right foot error
    posError = xRightFoot - xtipRightFoot
    rotError = 0.5 * (np.cross(rtipRightFoot[:,0], rRightFoot[:,0]) + np.cross(rtipRightFoot[:,1], rRightFoot[:,1]) + np.cross(rtipRightFoot[:,2], rRightFoot[:,2]))
    eRightFoot = np.atleast_2d(np.append(posError, rotError)).T

    # Find the left hand error
    posError = xLeftHand - xtipLeftHand
    rotError = 0.5 * (np.cross(rtipLeftHand[:,0], rLeftHand[:,0]) + np.cross(rtipLeftHand[:,1], rLeftHand[:,1]) + np.cross(rtipLeftHand[:,2], rLeftHand[:,2]))
    eLeftHand = np.atleast_2d(np.append(posError, rotError)).T
    
    # Find the right hand error
    posError = xRightHand - xtipRightHand
    rotError = 0.5 * (np.cross(rtipRightHand[:,0], rRightHand[:,0]) + np.cross(rtipRightHand[:,1], rRightHand[:,1]) + np.cross(rtipRightHand[:,2], rRightHand[:,2]))
    eRightHand = np.atleast_2d(np.append(posError, rotError)).T
    
    # Find the feet and hand velocity
    vLeftFoot = (np.atleast_2d(np.append(xdotLeftFoot, omegaLeftFoot)).T + 1 / (10.0 * 2) * eLeftFoot)
    vRightFoot = (np.atleast_2d(np.append(xdotRightFoot, omegaRightFoot)).T + 1 / (10.0 * 2) * eRightFoot)
    vLeftHand = (np.atleast_2d(np.append(xdotLeftHand, omegaLeftHand)).T + 1 / (10.0 * 2) * eLeftHand)
    vRightHand = (np.atleast_2d(np.append(xdotRightHand, omegaRightHand)).T + 1 / (10.0 * 2) * eRightHand)
   
    # Velocity inverse kinematics
    qdotLeftFoot = rpinvJLeftFoot @ vLeftFoot
    qdotRightFoot = rpinvJRightFoot @ vRightFoot
    qdotLeftHand = rpinvJLeftHand @ vLeftHand
    qdotRightHand = rpinvJRightHand @ vRightHand

    theta0[leftFootIdx, 0] = (np.atleast_2d(theta0[leftFootIdx, 0]).T + qdotLeftFoot * dt).flatten()
    theta0[rightFootIdx, 0] = (np.atleast_2d(theta0[rightFootIdx, 0]).T + qdotRightFoot * dt).flatten()
    theta0[leftHandIdx, 0] = (np.atleast_2d(theta0[leftHandIdx, 0]).T + qdotLeftHand * dt).flatten()
    theta0[rightHandIdx, 0] = (np.atleast_2d(theta0[rightHandIdx, 0]).T + qdotRightHand * dt).flatten()

    return (x, R, theta0)

#
#  Main Code
#
if __name__ == "__main__":
    # Prepare the node.
    rospy.init_node('motion')

    # Create a publisher to send the joint values (joint_states).
    # Note having a slightly larger queue prevents dropped messages!
    pub = rospy.Publisher("/joint_states", JointState, queue_size=100)

    # Wait until connected.  You don't have to wait, but the first
    # messages might go out before the connection and hence be lost.
    rospy.sleep(0.25)

    # Instantiate a tf broadcaster
    broadcaster = transforms.TransformBroadcaster()

    # Create a joint state message.  Each joint is explicitly named.
    msg = JointState()
    msg.name.append('back_bkz')
    msg.name.append('back_bky')
    msg.name.append('back_bkx')
    
    msg.name.append('l_arm_wry2') # 3
    msg.name.append('l_arm_wrx') # 4
    msg.name.append('l_arm_wry') # 5
    msg.name.append('l_arm_elx') # 6
    msg.name.append('l_arm_ely') # 7
    msg.name.append('l_arm_shx') # 8
    msg.name.append('l_arm_shz') # 9
    
    msg.name.append('l_situational_awareness_camera_joint') 
    msg.name.append('l_situational_awareness_camera_optical_frame_joint')
    msg.name.append('neck_ry')
    
    msg.name.append('r_arm_wry2') # 13
    msg.name.append('r_arm_wrx') # 14
    msg.name.append('r_arm_wry') # 15
    msg.name.append('r_arm_elx') # 16
    msg.name.append('r_arm_ely') # 17
    msg.name.append('r_arm_shx') # 18
    msg.name.append('r_arm_shz') # 19
    
    msg.name.append('r_situational_awareness_camera_joint')
    msg.name.append('r_situational_awareness_camera_optical_frame_joint')
    msg.name.append('rear_situational_awareness_camera_joint')
    msg.name.append('rear_situational_awareness_camera_optical_frame_joint')
    
    msg.name.append('l_leg_akx') # 24
    msg.name.append('l_leg_aky') # 25
    msg.name.append('l_leg_kny') # 26
    msg.name.append('l_leg_hpy') # 27
    msg.name.append('l_leg_hpx') # 28
    msg.name.append('l_leg_hpz') # 29
    
    msg.name.append('r_leg_akx') # 30
    msg.name.append('r_leg_aky') # 31
    msg.name.append('r_leg_kny') # 32
    msg.name.append('r_leg_hpy') # 33
    msg.name.append('r_leg_hpx') # 34
    msg.name.append('r_leg_hpz') # 35

    # Grab the URDF from the parameter server.
    urdf = rospy.get_param('/robot_description')

    # Set up the kinematics, from world to tip.
    kinLeftFoot = Kinematics(urdf, 'l_foot', 'utorso')
    kinRightFoot = Kinematics(urdf, 'r_foot', 'utorso')
    kinLeftHand = Kinematics(urdf, 'l_hand', 'utorso')
    kinRightHand = Kinematics(urdf, 'r_hand', 'utorso')

    # Set up kinematics from the utorso to the pelvis
    kinUtorsoPelvis = Kinematics(urdf, 'utorso', 'pelvis')
    qPelvis = np.zeros((kinUtorsoPelvis.dofs(), 1))
    xPelvis = np.zeros((3,1))
    rPelvis = np.identity(3)
    
    # Save our initial guess
    theta0 = np.zeros((36, 1))
    theta0[9, 0] = -0.95 # l_arm_shz
    theta0[8, 0] = -0.27 # l_arm_shx
    theta0[6, 0] = 0.08 # l_arm_elx
    theta0[7, 0] = 1.87 # l_arm_ely

    theta0[19, 0] = 0.95 # r_arm_shz
    theta0[18, 0] = 0.27 # r_arm_shx
    theta0[16, 0] = -0.08 # r_arm_elx
    theta0[17, 0] = 1.87 # r_arm_ely

    theta0[25, 0] = -0.14 # l_leg_aky
    theta0[31, 0] = 0.14 # r_leg_aky
    
    msg.position = np.squeeze(theta0.T).tolist()
    
    # Place the robot
    p_pw = np.array([[0.725369], [-0.1115], [0.36573]]) # Get the pelvis w.r.t. world position
    R_pw = np.array([[0.5401351, 0.0, 0.8415783], [0.0, 1.0, 0.0], [-0.8415783, 0.0, 0.5401351]]) # Get the pelvis w.r.t. world orientation

    # Determine the quaternions for the orientation, using a T matrix:
    T_pw = np.vstack((np.hstack((R_pw, p_pw)),
                      np.array([[0, 0, 0, 1]])))
    quat_pw = transforms.transformations.quaternion_from_matrix(T_pw)

    # Place the utorso  w.r.t. world.
    broadcaster.sendTransform(p_pw, quat_pw, rospy.Time.now(), 'pelvis', 'world')

    # Prepare a servo loop at 100Hz.
    rate  = 100
    servo = rospy.Rate(rate)
    dt    = servo.sleep_dur.to_sec()
    rospy.loginfo("Running the servo loop with dt of %f seconds (%fHz)" %
                  (dt, rate))

    # Run the servo loop until shutdown (killed or ctrl-C'ed).
    t   = 0.0
    tff  = 2.0 - dt
    lam = 0.1/dt
    while not rospy.is_shutdown():

        # Move to a new time step, assuming a constant step!
        t = t + dt

        # Inverse kinematics
        result = ikin(theta0, t, dt, kinLeftFoot, kinRightFoot, kinLeftHand, kinRightHand)
        theta0 = result[2]

        # Set the positions as a function of time.
        msg.position = np.squeeze(theta0.T).tolist()

        # Send the command (with the current time).
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        
        # Place the robot by converting the upper torso position to the pelvis position with respect to the world
        kinUtorsoPelvis.fkin(qPelvis, xPelvis, rPelvis)
        R_pw = result[1] @ rPelvis
        p_pw = result[0] + result[1] @ xPelvis

        # Determine the quaternions for the orientation, using a T matrix:
        T_pw = np.vstack((np.hstack((R_pw, p_pw)),
                          np.array([[0, 0, 0, 1]])))
        quat_pw = transforms.transformations.quaternion_from_matrix(T_pw)

        # Place the pelvis w.r.t. world.
        broadcaster.sendTransform(p_pw, quat_pw, rospy.Time.now(), 'pelvis', 'world')
        
        # Sleep
        servo.sleep()

        # Break if we have completed the full time.
        if (t > tff):
            break
