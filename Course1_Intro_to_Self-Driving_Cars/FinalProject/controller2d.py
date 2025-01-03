#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
            self.vars.closest_waypoints_idx = min_idx
        else:
            desired_speed = self._waypoints[-1][2]
            self.vars.closest_waypoints_idx = len(self._waypoints) - 1
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################
        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.

            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)

            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0

            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('t_previous', 0.0)
        self.vars.create_var('sum_of_integral_error', 0.0)
        self.vars.create_var('prev_v_error', 0.0)
        self.vars.create_var('closest_waypoints_idx', 0)

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            
            # Change these outputs with the longitudinal controller. Note that
            # brake_output is optional and is not required to pass the
            # assignment, as the car will naturally slow down over time.
            throttle_output = 0
            brake_output    = 0

            ## PID Controller
            K_p = 0.2
            K_d = 0.01
            K_i = 0.05

            v_error = v_desired - v
            delta_t = t - self.vars.t_previous

            integral_error = v_error * delta_t
            self.vars.sum_of_integral_error += integral_error

            derivative_error = (v_error - self.vars.prev_v_error) / (delta_t + 1e-6)

            desired_acceleration = K_p * v_error + K_d * derivative_error + K_i * self.vars.sum_of_integral_error

            if desired_acceleration > 0:
                throttle_output = desired_acceleration
                brake_output = 0
            else:
                throttle_output = 0
                brake_output = -desired_acceleration

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            #### Pure Pursuit Controller
            look_ahead_idx = self.vars.closest_waypoints_idx + 1000 ## Look 10 meters Ahead

            x_2 = np.array([waypoints[look_ahead_idx][0], waypoints[look_ahead_idx][1]]) ## Look Ahead Point
            x_1 = np.array([x, y])  ## Current Position

            vector_towards_lookahead_point = x_2 - x_1
            angle1 = np.arctan2(vector_towards_lookahead_point[1], vector_towards_lookahead_point[0])
            angle2 = yaw
            angle = angle1 - angle2

            l_d = np.linalg.norm(x_2 - x_1)
            
            ## Pure Pursuit With lookahead distance
            vehicle_length = 1.5 ## 1.5 meters
            epsilon = 1 ## To Stabilize at low speeds
            stear_angle_ld = np.arctan((2 * vehicle_length* np.sin(angle)) / (l_d + epsilon)) 

            ## Pure Pursuit with Velocity Consideration
            K_dd = 0.1
            stear_angle_vel = np.arctan((2 * vehicle_length * np.sin(angle))/(K_dd*v + epsilon))

            ## Stanley Controller
            ## Heading Error
            trajectory_vector = np.array(waypoints[50 + self.vars.closest_waypoints_idx]) - np.array(waypoints[self.vars.closest_waypoints_idx])
            trajectory_angle = np.arctan2(trajectory_vector[1], trajectory_vector[0])
            heading_error = trajectory_angle - yaw

            ## Cross Track Error
            k = 1
            cross_track_e = np.array([x, y]) - np.array(waypoints[self.vars.closest_waypoints_idx][:2])
            sign = np.sign(np.cross(cross_track_e, trajectory_vector[:2]))
            cross_track_error = np.arctan(k * sign *np.linalg.norm(cross_track_e) / (v+epsilon))

            stear_angle_stanley = heading_error + cross_track_error

            #Change the steer output with the lateral controller. 
            steer_output = stear_angle_stanley

            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.vars.v_previous = v  # Store forward speed to be used in next step
        self.vars.t_previous = t
        self.vars.prev_v_error = v_error
        self.vars.sum_of_integral_error = self.vars.sum_of_integral_error