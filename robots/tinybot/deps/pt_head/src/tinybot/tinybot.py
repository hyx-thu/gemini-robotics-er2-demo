import sys
import os

from .dxl_manager import DXLManager as DXL
from time import sleep
from .beeps import audio_playback

BEEPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'beeps')

MY_DXL_CHAIN = [1,2]

# Joint limits for the head motors:
# - Motor 1 (distal joint): 110 (looks right) to 250 (looks left)
# - Motor 2 (proximal joint): 150 (looks up) to 220 (looks down)
JOINT_LIMITS = {
    1: (110, 250),
    2: (150, 220)
}

MAX_SPEED = 180

class TinyBot:
    """
    Class designed for the Head for HRI testing
    """

    def __init__(
        self,
        DEVICENAME,
        DXL_CHAIN,
        BAUDRATE = 115200,

    ):
        self.head = DXL(MY_DXL=DXL_CHAIN, DEVICENAME=DEVICENAME, BAUDRATE=BAUDRATE)
        self.dxl_chain = DXL_CHAIN
        self.head.read_all_curr_pos(DXL_CHAIN)
        self.enable_all_torque()


    def enable_all_torque(self):
        """
        Enables torque on all Dynamixel motors in the chain.
        """
        for dxl_id in self.dxl_chain:
            self.head.enable_torque(dxl_id)

    def disable_all_torque(self):
        """
        Disables torque on all Dynamixel motors in the chain.
        """
        for dxl_id in self.dxl_chain:
            self.head.disable_torque(dxl_id)


    def home_pose(self, speed=150):
        """
        Moves the head to a neutral 'home' position.
        """
        print("Moving to home pose...")
        self.head.write_goal_pose_with_speed(1, 180, speed)
        self.head.write_goal_pose_with_speed(2, 180, speed)
        self.head.read_all_curr_pos(self.dxl_chain)
        sleep(0.5)

    def nod(self, speed=150):
        """
        Performs a 'nod' head gesture.
        """
        print("Performing nod gesture...")
        self.home_pose(speed)
        sleep(0.3)
        self.head.read_all_curr_pos(self.dxl_chain)
        self.head.write_goal_pose_with_speed(2, 200, speed)
        sleep(0.3)
        audio_playback.play_audio_on_device(os.path.join(BEEPS_DIR, "happy_beep.wav"), 10)
        self.head.read_all_curr_pos(self.dxl_chain)
        self.head.write_goal_pose_with_speed(2, 180, speed)
        sleep(0.3)
        self.head.read_all_curr_pos(self.dxl_chain)



    def no(self, speed=150):
        """
        Performs a 'no' head shake gesture.
        """
        print("Performing 'no' gesture...")
        self.home_pose(speed)
        sleep(0.3)
        self.head.read_all_curr_pos(self.dxl_chain)
        self.head.write_goal_pose_with_speed(1, 140, speed)
        sleep(0.3)
        self.head.read_all_curr_pos(self.dxl_chain)
        self.head.write_goal_pose_with_speed(1, 220, speed)
        sleep(0.3)
        audio_playback.play_audio_on_device(os.path.join(BEEPS_DIR, "disapproval_boop.wav"), 10)
        self.head.read_all_curr_pos(self.dxl_chain)
        self.head.write_goal_pose_with_speed(1, 140, speed)
        sleep(0.3)
        self.head.read_all_curr_pos(self.dxl_chain)
        self.head.write_goal_pose_with_speed(1, 220, speed)
        sleep(0.3)
        self.head.read_all_curr_pos(self.dxl_chain)
        self.home_pose(speed)

    def make_gesture(self, gesture_name, speed=150):
        """
        Performs a gesture by name.
        Supported gestures: 'nod', 'no', 'home'
        """
        if gesture_name == 'nod':
            self.nod(speed)
        elif gesture_name == 'no':
            self.no(speed)
        elif gesture_name in ('home', 'home_pose'):
            self.home_pose(speed)
        else:
            print(f"Unknown gesture: {gesture_name}")

    def get_current_angles(self):
        """
        Returns the current angles of the head joints in degrees.
        """
        angles = []
        for dxl_id in self.dxl_chain:
            pulses = self.head.read_curr_pos(dxl_id)
            angles.append(pulses * 0.088)
        return angles

    def move_absolute(self, angles, speeds, blocking=True):
        """
        Moves the head joints to the specified absolute angles at specified speeds.
        
        Args:
            angles (list of float): Goal positions in degrees for each motor.
            speeds (list of int): Profile velocities for each motor.
            blocking (bool, optional): If True, waits for all motors to stop moving.
                                       Defaults to True.
        """
        if len(angles) != len(self.dxl_chain) or len(speeds) != len(self.dxl_chain):
            print(f"Error: Length of angles ({len(angles)}) and speeds ({len(speeds)}) must match the DXL chain length ({len(self.dxl_chain)}).")
            return False

        clamped_angles = []
        for i, dxl_id in enumerate(self.dxl_chain):
            angle = angles[i]
            if dxl_id in JOINT_LIMITS:
                min_lim, max_lim = JOINT_LIMITS[dxl_id]
                clamped_angle = max(min_lim, min(max_lim, angle))
                if clamped_angle != angle:
                    print(f"Warning: Angle {angle} for motor {dxl_id} clamped to {clamped_angle} due to joint limits.")
                clamped_angles.append(clamped_angle)
            else:
                clamped_angles.append(angle)

        clamped_speeds = []
        for speed in speeds:
            clamped_speed = max(1, min(speed, MAX_SPEED))
            if clamped_speed != speed:
                print(f"Warning: Speed {speed} clamped to {clamped_speed} due to MAX_SPEED limit.")
            clamped_speeds.append(clamped_speed)

        # Start movement for all motors (non-blocking)
        for i, dxl_id in enumerate(self.dxl_chain):
            self.head.write_goal_pose_with_speed(dxl_id, clamped_angles[i], clamped_speeds[i], blocking=False)

        if blocking:
            sleep(0.05)
            while True:
                any_moving = False
                for dxl_id in self.dxl_chain:
                    if self.head.is_moving(dxl_id):
                        any_moving = True
                        break
                if not any_moving:
                    break
                sleep(0.01)
            print("All movements completed.")
            self.head.read_all_curr_pos(self.dxl_chain)

        return True

    def move_relative(self, relative_angles, speeds, blocking=True):
        """
        Moves the head joints by a relative amount from their current positions.
        
        Args:
            relative_angles (list of float): Relative angles in degrees.
            speeds (list of int): Profile velocities for each motor.
            blocking (bool, optional): If True, waits for all motors to stop moving.
        """
        if len(relative_angles) != len(self.dxl_chain) or len(speeds) != len(self.dxl_chain):
            print(f"Error: Length of relative_angles ({len(relative_angles)}) and speeds ({len(speeds)}) must match the DXL chain length ({len(self.dxl_chain)}).")
            return False

        current_angles = self.get_current_angles()
        target_angles = [current_angles[i] + relative_angles[i] for i in range(len(self.dxl_chain))]
        
        print(f"Relative move: current={current_angles}, relative={relative_angles} -> target={target_angles}")
        return self.move_absolute(target_angles, speeds, blocking)



