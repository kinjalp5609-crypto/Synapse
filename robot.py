import serial
import serial.tools.list_ports
import time

class SynapseRobot:
    
    def __init__(self):
        self.connection = None
        self.connected = False
        self.current_direction = "STOP"
        self.current_speed = 2  # 1=slow 2=medium 3=fast
    
    def find_bluetooth_port(self):
        """Automatically find HC-05 Bluetooth port"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(f"Found port: {port.device} - {port.description}")
            # HC-05 usually shows up as these names
            if any(name in port.description.upper() for name in 
                   ['HC-05', 'BLUETOOTH', 'SERIAL', 'CH340', 'CP210']):
                return port.device
        return None
    
    def connect(self, port=None):
        """Connect to robot via Bluetooth"""
        try:
            if port is None:
                print("Searching for Bluetooth port...")
                port = self.find_bluetooth_port()
            
            if port is None:
                print("No Bluetooth port found automatically.")
                print("Available ports:")
                ports = serial.tools.list_ports.comports()
                for i, p in enumerate(ports):
                    print(f"  {i}: {p.device} - {p.description}")
                choice = input("Enter port number to use: ")
                port = list(serial.tools.list_ports.comports())[int(choice)].device
            
            self.connection = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)  # Wait for connection to stabilize
            self.connected = True
            print(f"Connected to robot on {port}")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            self.connected = False
            return False
    
    def send_command(self, command):
        """Send single character command to robot"""
        if not self.connected:
            print("Robot not connected!")
            return False
        try:
            self.connection.write(command.encode())
            time.sleep(0.05)  # Small delay between commands
            return True
        except Exception as e:
            print(f"Send failed: {e}")
            self.connected = False
            return False
    
    def move(self, direction, speed=None):
        """
        Main function — call this from main.py
        direction: 'FORWARD' 'BACKWARD' 'LEFT' 'RIGHT' 'CENTER' 'STOP'
        speed:     1 (slow) 2 (medium) 3 (fast)
        """
        if not self.connected:
            return
        
        # Update speed if changed
        if speed is not None and speed != self.current_speed:
            self.current_speed = speed
            self.send_command(str(speed))
        
        # Only send command if direction changed
        # This prevents spamming same command 30 times per second
        if direction == self.current_direction:
            return
        
        self.current_direction = direction
        
        command_map = {
            'FORWARD':   'F',
            'BACKWARD':  'B',
            'LEFT':      'L',
            'RIGHT':     'R',
            'CENTER':    'S',
            'STOP':      'S',
            'BLINK':     'S',
        }
        
        command = command_map.get(direction, 'S')
        success = self.send_command(command)
        
        if success:
            print(f"Robot: {direction} (speed={self.current_speed})")
    
    def stop(self):
        """Emergency stop"""
        self.send_command('S')
        self.current_direction = "STOP"
        print("Robot: EMERGENCY STOP")
    
    def self_test(self):
        """Run Arduino self test"""
        print("Running robot self test...")
        self.send_command('T')
    
    def disconnect(self):
        """Safely disconnect"""
        if self.connection:
            self.stop()
            time.sleep(0.5)
            self.connection.close()
            self.connected = False
            print("Robot disconnected.")


# ── Test this file standalone ─────────────────────────────────
if __name__ == "__main__":
    robot = SynapseRobot()
    
    if robot.connect():
        print("Testing robot commands...")
        time.sleep(1)
        
        robot.move('FORWARD')
        time.sleep(2)
        
        robot.move('LEFT')
        time.sleep(1)
        
        robot.move('RIGHT')
        time.sleep(1)
        
        robot.move('BACKWARD')
        time.sleep(2)
        
        robot.stop()
        robot.disconnect()
    else:
        print("Could not connect to robot.")
        print("This is normal if hardware is not connected yet.")
        print("This file is ready for July!")
