#!/usr/bin/env python3
"""Gripper diagnostic: try mode change with error checking, then use PWM if needed."""

import glob, sys, time
from dynamixel_sdk import *

ports = glob.glob('/dev/tty.usbserial*') + glob.glob('/dev/ttyUSB*')
port = PortHandler(ports[0])
pkt  = PacketHandler(2.0)
port.openPort()
port.setBaudRate(1000000)

GID = 5
ADDR_OP_MODE   = 11
ADDR_TORQUE    = 64
ADDR_GOAL_POS  = 116
ADDR_CUR_POS   = 132
ADDR_GOAL_PWM  = 100
ADDR_GOAL_CUR  = 102
ADDR_HW_ERR    = 70

print(f"Connected to {ports[0]}")
print("=" * 50)

# Check hardware error status
hw_err, _, _ = pkt.read1ByteTxRx(port, GID, ADDR_HW_ERR)
print(f"Hardware error status: {hw_err} (0=OK)")

# Disable torque with error check
res, err = pkt.write1ByteTxRx(port, GID, ADDR_TORQUE, 0)
print(f"Disable torque: comm={res}, err={err}")
time.sleep(0.3)

# Verify torque is off
torque, _, _ = pkt.read1ByteTxRx(port, GID, ADDR_TORQUE)
print(f"Torque state: {torque} (should be 0)")

# Try to set mode 5 (current-based position)
print("\nAttempting mode change to 5...")
res, err = pkt.write1ByteTxRx(port, GID, ADDR_OP_MODE, 5)
print(f"  Write result: comm={res}, err={err}")
time.sleep(0.2)
mode, _, _ = pkt.read1ByteTxRx(port, GID, ADDR_OP_MODE)
print(f"  Operating mode now: {mode}")

if mode == 5:
    print("  Mode change SUCCESS!")
elif mode == 16:
    print("  Mode change FAILED — motor locked in PWM mode.")
    print("  Using PWM control instead (this works fine for grippers).")

# Try mode 3 as fallback
if mode == 16:
    print("\nTrying mode 3 (position control)...")
    res, err = pkt.write1ByteTxRx(port, GID, ADDR_OP_MODE, 3)
    print(f"  Write result: comm={res}, err={err}")
    time.sleep(0.2)
    mode, _, _ = pkt.read1ByteTxRx(port, GID, ADDR_OP_MODE)
    print(f"  Operating mode now: {mode}")

# Enable torque
pkt.write1ByteTxRx(port, GID, ADDR_TORQUE, 1)
time.sleep(0.3)

print("\n" + "=" * 50)
if mode == 16:
    print("GRIPPER IS IN PWM MODE — Testing with PWM commands")
    print("PWM range: -885 to 885")
    print()

    tests = [
        (200, "Open (PWM +200)"),
        (0,   "Stop"),
        (400, "Open more (PWM +400)"),
        (0,   "Stop"),
        (-200, "Close (PWM -200)"),
        (0,   "Stop"),
        (-400, "Close more (PWM -400)"),
        (0,   "Stop"),
    ]

    for pwm, label in tests:
        print(f"  {label}...", end=" ", flush=True)
        # write2ByteTxRx needs unsigned, but PWM is signed
        # For negative values, use two's complement
        if pwm < 0:
            pwm_unsigned = pwm + 65536
        else:
            pwm_unsigned = pwm
        pkt.write2ByteTxRx(port, GID, ADDR_GOAL_PWM, pwm_unsigned)
        time.sleep(1.0)
        pos, _, _ = pkt.read4ByteTxRx(port, GID, ADDR_CUR_POS)
        print(f"position now: {pos}")

    # Stop
    pkt.write2ByteTxRx(port, GID, ADDR_GOAL_PWM, 0)
    print("\n  Gripper stopped.")
else:
    print(f"GRIPPER IN MODE {mode} — Testing with position commands")
    if mode == 5:
        pkt.write2ByteTxRx(port, GID, ADDR_GOAL_CUR, 350)
    for target, label in [(2600,"Open"), (1500,"Close"), (2048,"Center")]:
        print(f"  {label} (pos={target})...", end=" ", flush=True)
        pkt.write4ByteTxRx(port, GID, ADDR_GOAL_POS, target)
        time.sleep(1.5)
        pos, _, _ = pkt.read4ByteTxRx(port, GID, ADDR_CUR_POS)
        print(f"reached {pos}")

port.closePort()
print("\nDone!")
