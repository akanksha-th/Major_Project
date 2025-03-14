import airsim
import time

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# Enable API control
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
print("Taking off...")
client.takeoffAsync().join()

# Hover for a few seconds
time.sleep(3)

# Land
print("Landing...")
client.landAsync().join()

# Release API control
client.armDisarm(False)
client.enableApiControl(False)

print("Test complete!")
