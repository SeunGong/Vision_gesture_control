import serial
import time

ser = serial.Serial('/dev/ttyUSB0',115200)

try:
    
    while True:
        ser.write(b'hello\n')
        time.sleep(1)
except KeyboardInterrupt:
    ser.close()