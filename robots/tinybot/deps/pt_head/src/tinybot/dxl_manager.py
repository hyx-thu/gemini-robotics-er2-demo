import os

from dynamixel_sdk import *

# Dynamixel SDK: https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/api_reference/python/python_porthandler/#python
# Dynamixel XL330-M288-T: https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/
# Dynamixel XC330-M288-T: https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/

# These are values that come form the motor spec (parameter table addresses and value ranges. The X series is all the same)

# Addresses in the control tables 
_ADDR_TORQUE_ENABLE = 64
_ADDR_GOAL_POSITION = 116
_ADDR_PROFILE_VELOCITY = 112
_ADDR_PRESENT_POSITION = 132
_ADDR_MOVING = 122
_ADDR_OPERATING_MODE = 11

# Lengths of bytes of date for each table value
_LEN_PROFILE_VELOCITY = 4
_LEN_GOAL_POSITION = 4
_LEN_PRESENT_POSITION = 4
_LEN_MOVING = 1
_LEN_OPERATING_MODE = 1

_DXL_MINIMUM_POSITION_VALUE = 0
_DXL_MAXIMUM_POSITION_VALUE = 4095
_PROTOCOL_VERSION = 2.0
_TORQUE_ENABLE = 1
_TORQUE_DISABLE = 0
_DXL_MOVING_STATUS_THRESHOLD = 20
_dxl_goal_position = [_DXL_MINIMUM_POSITION_VALUE, _DXL_MAXIMUM_POSITION_VALUE]

class DXLManager:
    """
    This class is designed to make using the dyanmixel SDK simpler and help a user get going on basic functions 
    """
    def __init__(
        self,
        MY_DXL,
        DEVICENAME,
        BAUDRATE = 115200,

    ):
        
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(_PROTOCOL_VERSION)
        # self.goalPose_groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, _ADDR_GOAL_POSITION, _LEN_GOAL_POSITION)
        self.groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, _ADDR_PRESENT_POSITION, _LEN_PRESENT_POSITION)
        self.dxl_chain = MY_DXL # Store the chain as an instance variable

        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")

        #Set port's baudrate
        if self.portHandler.setBaudRate(BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")

    def set_extended_position_mode(self):
        """
        Sets all motors in the chain to Extended Position Control Mode (ID: 4).
        """
        for dxl_id in self.dxl_chain:
            # First, disable torque to change the operating mode
            self.disable_torque(dxl_id)
            
            # Write 4 to the Operating Mode register for Extended Position Control Mode
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler,
                dxl_id,
                _ADDR_OPERATING_MODE,
                4
            )
            
            if dxl_comm_result != COMM_SUCCESS:
                print(f"[ID:{dxl_id}] Operating Mode set failed: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"[ID:{dxl_id}] Operating Mode set error: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                print(f"Dynamixel#{dxl_id} has been successfully set to Extended Position Control Mode")

    def enable_torque(
        self,
        DXL_ID,
    ):
        """
        Enables torque on a given DXL ID
        """
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, _ADDR_TORQUE_ENABLE, _TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d Torque has been successfully enabled" % DXL_ID)

    def disable_torque(
        self,
        DXL_ID,
    ):
        """
        Disables torque on a given DXL ID
        """
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, _ADDR_TORQUE_ENABLE, _TORQUE_DISABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d Torque has been successfully disabled" % DXL_ID) 

    def read_curr_pos(
            self,
            DXL_ID,
    ):
        """
        Reads the current position on a given DXL ID
        """

        dxl_curr_pos_result = self.groupSyncRead.addParam(DXL_ID)
        if dxl_curr_pos_result != True: 
            print("[ID:%03d] groupSyncRead addparam failed" % DXL_ID)
        dxl_comm_result = self.groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        
        dxl_curr_pos = self.groupSyncRead.getData(DXL_ID, _ADDR_PRESENT_POSITION, _LEN_PRESENT_POSITION)
        print(f"[ID:{DXL_ID}] Present position: {dxl_curr_pos*0.088 }")
        self.groupSyncRead.clearParam()
        return dxl_curr_pos
        
    def read_all_curr_pos(
            self,
            DXL_CHAIN = list
            ):
        """
        Reads current positon of all dynamixels in the chain (recommend a small buffer time after a move command is sent).
        """
        self.readClear()
        for i in DXL_CHAIN:
            self.read_curr_pos(i)
    

    def write_goal_pose(
            self,
            DXL_ID,
            degs,
    ):
        """
        Writes a new goal postion for a given DXL ID
        """
        groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, _ADDR_GOAL_POSITION, _LEN_GOAL_POSITION)

        index = int(degs/0.088) # convert degrees to pulses (based on steps in the encoder can read) 
        param_goal_position = [DXL_LOBYTE(DXL_LOWORD(index)), DXL_HIBYTE(DXL_LOWORD(index)), DXL_LOBYTE(DXL_HIWORD(index)), DXL_HIBYTE(DXL_HIWORD(index))]
        dxl_addparam_result = groupSyncWrite.addParam(DXL_ID, param_goal_position)
        if dxl_addparam_result != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID)

        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        groupSyncWrite.clearParam()
    
    def goalPose_writeClear(self):
        """
        GTFO Write Params
        may not be needed anymore after initializing the groupsyncwrites in their respective functions 
        """
        
        groupSyncWrite.clearParam()

    def readClear(self):
        """
        GTFO Read Params
        """
        
        self.groupSyncRead.clearParam()

    def write_goal_pose_with_speed(
        self,
        DXL_ID,
        degs,
        speed,
        blocking=True
    ):
        """
        Drives the motor to a specified position at a given speed and stops.
        
        Args:
            DXL_ID (int): ID of the Dynamixel motor.
            degs (float): Goal position in degrees.
            speed (int): Profile velocity.
            blocking (bool, optional): If True, the function waits for the motor 
                                       to stop moving before returning. Defaults to True.
        """
        # First, set the profile velocity to control the speed
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler,
            DXL_ID,
            _ADDR_PROFILE_VELOCITY,
            int(speed)
        )
        
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[ID:{DXL_ID}] Profile velocity set failed: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            return False
        elif dxl_error != 0:
            print(f"[ID:{DXL_ID}] Profile velocity set error: {self.packetHandler.getRxPacketError(dxl_error)}")
            return False

        # Convert degrees to pulses
        index = int(degs / 0.088)

        # Now, set the goal position
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler,
            DXL_ID,
            _ADDR_GOAL_POSITION,
            index
        )
        
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[ID:{DXL_ID}] Goal position write failed: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            return False
        elif dxl_error != 0:
            print(f"[ID:{DXL_ID}] Goal position write error: {self.packetHandler.getRxPacketError(dxl_error)}")
            return False

        # If blocking is enabled, wait for the motor to stop moving
        if blocking:
            while True:
                # Read the 'Moving' status from the motor's control table
                dxl_moving_result, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(
                    self.portHandler,
                    DXL_ID,
                    _ADDR_MOVING
                )
                
                if dxl_comm_result != COMM_SUCCESS:
                    print(f"Failed to read moving status: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
                    return False
                
                if dxl_error != 0:
                    print(f"Error reading moving status: {self.packetHandler.getRxPacketError(dxl_error)}")
                    return False
                
                # If the motor is not moving, break the loop
                if dxl_moving_result == 0:
                    print(f"Movement to position {degs} completed for ID:{DXL_ID}.")
                    break
        
        return True

    def is_moving(self, DXL_ID):
        """
        Returns True if the motor is moving, False otherwise.
        """
        dxl_moving_result, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(
            self.portHandler,
            DXL_ID,
            _ADDR_MOVING
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Failed to read moving status for ID:{DXL_ID}: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            return False
        if dxl_error != 0:
            print(f"Error reading moving status for ID:{DXL_ID}: {self.packetHandler.getRxPacketError(dxl_error)}")
            return False
        return dxl_moving_result == 1
