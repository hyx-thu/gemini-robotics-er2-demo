# Test of DXL Manager & HRI Head file. 

from tinybot import TinyBot as HRI
from time import sleep

MY_DXL_CHAIN = [1, 2]
DEVICENAME = '/dev/ttyUSB0'  

print("Initializing HRI_head...")
try:
    head_controller = HRI(DEVICENAME=DEVICENAME, DXL_CHAIN=MY_DXL_CHAIN,BAUDRATE = 115200)

    # Now you can call the pre-made motions
    print('home')
    head_controller.make_gesture('home')
    
    print('nod')
    head_controller.make_gesture('nod', speed=200)
    
    sleep(0.5)

    print('no')
    head_controller.make_gesture('no', speed=100)

    print('custom move')
    head_controller.move_absolute(angles=[150, 210], speeds=[40, 60])
    sleep(0.5)
    head_controller.move_absolute(angles=[180, 180], speeds=[50, 50])

    print('Script completed')


except Exception as e:
    print(f"An error occurred: {e}")

