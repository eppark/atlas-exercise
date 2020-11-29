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
def ikin(theta0, t, dt, kinLeft, kinRight):
    NLeft = kinLeft.dofs()
    NRight = kinRight.dofs()

    # Allocate the numpy variables for joint position, tip position,
    # tip orientation, and Jacobian for the left leg
    qLeft = np.atleast_2d(theta0[24:30, 0]).T
    xtipLeft = np.zeros((3,1))
    rtipLeft = np.identity(3)
    JLeft = np.zeros((6,NLeft))
    
    # Allocate the numpy variables for joint position, tip position,
    # tip orientation, and Jacobian for the right leg
    qRight = np.atleast_2d(theta0[30:, 0]).T
    xtipRight = np.zeros((3,1))
    rtipRight = np.identity(3)
    JRight = np.zeros((6,NRight))

    # Calculate s
    if t < 1:
        s = 3 * t**2 - 2 * t**3
        sdot = 6 * t - 6 * t**2
    else:
        s = -4 + 12 * t - 9 * t**2 + 2 * t**3
        sdot = 12 - 18 * t + 6 * t**2
        
    ###################################
    # initial position standing up
    # np.array([[0.0], [0.0], [0.86201]])
    #
    # initial position down
    # np.array([[-0.33471], [0.0], [0.60384]])
        #
    # Left foot w.r.t. world : np.array([[0.1115], [0.0], [0.0]])
    # Right foot w.r.t. world : np.array([[-0.1115], [0.0], [0.0]])
    ####################################
        
    # Calculate position and velocity with respect to the world frame
    x = np.array([[-0.33471 * s], [0.0], [0.86201 - 0.25817 * s]])
    xdot = np.array([[-0.33471 * sdot], [0.0], [-0.25817 * sdot]])
        
    # Calculate position and velocity for the left and right respective frames
    xLeft = x - np.array([[0.1115], [0.0], [0.0]])
    xRight = x + np.array([[0.1115], [0.0], [0.0]])
    xdotLeft = np.copy(xdot)
    xdotRight = np.copy(xdot)

    # Calculate the rotation and angular velocity for the left and right legs
    rLeft = np.identity(3)
    omegaLeft = np.zeros((3, 1))
    rRight = np.identity(3)
    omegaRight = np.zeros((3, 1))

    
    # Get the pelvis positions and rotations
    kinLeft.fkin(qLeft, xtipLeft, rtipLeft)
    kinRight.fkin(qRight, xtipRight, rtipRight)
    
    # Get the Jacobian mapping to position and orientation of both perspectives
    kinLeft.Jac(qLeft, JLeft)
    kinRight.Jac(qRight, JRight)
    
    # Get the psuedoinverse of the Jacobian
    rpinvJLeft = pinv(JLeft)
    rpinvJRight = pinv(JRight)
    
    # Find the left error
    posError = xLeft - xtipLeft
    rotError = 0.5 * (np.cross(rtipLeft[:,0], rLeft[:,0]) + np.cross(rtipLeft[:,1], rLeft[:,1]) + np.cross(rtipLeft[:,2], rLeft[:,2]))
    eLeft = np.atleast_2d(np.append(posError, rotError)).T
    
    # Find the right error
    posError = xRight - xtipRight
    rotError = 0.5 * (np.cross(rtipRight[:,0], rRight[:,0]) + np.cross(rtipRight[:,1], rRight[:,1]) + np.cross(rtipRight[:,2], rRight[:,2]))
    eRight = np.atleast_2d(np.append(posError, rotError)).T
    
    # Find the pelvis velocity
    vLeft = (np.atleast_2d(np.append(xdotLeft, omegaLeft)).T + 1 / (10.0 * 2) * eLeft)
    vRight = (np.atleast_2d(np.append(xdotRight, omegaRight)).T + 1 / (10.0 * 2) * eRight)
   
    # Velocity inverse kinematics
    qdotLeft = rpinvJLeft @ vLeft 
    qdotRight = rpinvJRight @ vRight
    
    # 24-29
    lefttheta0 = np.atleast_2d(theta0[24:30, 0]).T + qdotLeft * dt
    
    # 30-35
    righttheta0 = np.atleast_2d(theta0[30:, 0]).T + qdotRight * dt
    
    theta0[24:30, 0] = lefttheta0.flatten()
    theta0[30:, 0] = righttheta0.flatten()

    return (x, rLeft, theta0) # Since rLeft = rRight = R w.r.t. world, as the feet are oriented the same

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
    msg.name.append('l_arm_shz') # 3
    msg.name.append('l_arm_shx')
    msg.name.append('l_arm_ely')
    msg.name.append('l_arm_elx')
    msg.name.append('l_arm_wry')
    msg.name.append('l_arm_wrx')
    msg.name.append('l_arm_wry2')
    msg.name.append('l_situational_awareness_camera_joint') 
    msg.name.append('l_situational_awareness_camera_optical_frame_joint')
    msg.name.append('neck_ry')
    msg.name.append('r_arm_shz') # 13
    msg.name.append('r_arm_shx') 
    msg.name.append('r_arm_ely')
    msg.name.append('r_arm_elx')
    msg.name.append('r_arm_wry')
    msg.name.append('r_arm_wrx')
    msg.name.append('r_arm_wry2')
    msg.name.append('r_situational_awareness_camera_joint')
    msg.name.append('r_situational_awareness_camera_optical_frame_joint')
    msg.name.append('rear_situational_awareness_camera_joint')
    msg.name.append('rear_situational_awareness_camera_optical_frame_joint')
    msg.name.append('l_leg_akx') # 24
    msg.name.append('l_leg_aky')
    msg.name.append('l_leg_kny')
    msg.name.append('l_leg_hpy')
    msg.name.append('l_leg_hpx') # 28
    msg.name.append('l_leg_hpz') # 29
    msg.name.append('r_leg_akx') # 30
    msg.name.append('r_leg_aky')
    msg.name.append('r_leg_kny')
    msg.name.append('r_leg_hpy')
    msg.name.append('r_leg_hpx') # 34
    msg.name.append('r_leg_hpz') # 35

    # Grab the URDF from the parameter server.
    urdf = rospy.get_param('/robot_description')

    # Set up the kinematics, from world to tip.  Report the DOFs.
    kinLeft = Kinematics(urdf, 'l_foot', 'pelvis')
    kinRight = Kinematics(urdf, 'r_foot', 'pelvis')

    # Save our initial guess
    theta0 = np.zeros((36, 1))
    theta0[3, 0] = -1.57
    theta0[13, 0] = 1.57

    # Bend the legs forward a little bit
    theta0[25, 0] = -0.2
    theta0[26, 0] = 0.2
    theta0[27, 0] = -0.2

    theta0[31, 0] = -0.2
    theta0[32, 0] = 0.2
    theta0[33, 0] = -0.2

    msg.position = np.squeeze(theta0.T).tolist()

    # Prepare a servo loop at 100Hz.
    rate  = 100;
    servo = rospy.Rate(rate)
    dt    = servo.sleep_dur.to_sec()
    rospy.loginfo("Running the servo loop with dt of %f seconds (%fHz)" %
                  (dt, rate))

    # Run the servo loop until shutdown (killed or ctrl-C'ed).
    t   = 0.0
    tf  = 2.0
    lam = 0.1/dt
    while not rospy.is_shutdown():

        # Move to a new time step, assuming a constant step!
        t = t + dt

        # Inverse kinematics
        result = ikin(theta0, t, dt, kinLeft, kinRight)
        theta0 = result[2]

        # Set the positions as a function of time.
        msg.position = np.squeeze(theta0.T).tolist()

        # Send the command (with the current time).
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        
        # Place the robot
        p_pw = result[0] # Get the pelvis w.r.t. world position
        R_pw = result[1] # Get the pelvis w.r.t. world orientation

        # Determine the quaternions for the orientation, using a T matrix:
        T_pw = np.vstack((np.hstack((R_pw, p_pw)),
                          np.array([[0, 0, 0, 1]])))
        quat_pw = transforms.transformations.quaternion_from_matrix(T_pw)

        # Place the pelvis w.r.t. world.
        broadcaster.sendTransform(p_pw, quat_pw, rospy.Time.now(), 'pelvis', 'world')
        
        # Sleep
        servo.sleep()

        # Break if we have completed the full time.
        if (t > tf):
            break
