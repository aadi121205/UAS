import subprocess
import getpass

# Get the password
password = getpass.getpass()

# Define the commands you want to execute
command1 = "echo {} | sudo -S gopro webcam".format(password)
command2 = "echo {} | sudo -S ffmpeg -nostdin -threads 1 -i 'udp://@0.0.0.0:8554?overrun_nonfatal=1&fifo_size=50000000' -f:v mpegts -fflags nobuffer -vf format=yuv420p -f v4l2 /dev/video42 -c:v copy -f fpl rtmp://192.168.1.68:9999".format(password)
command3 = "echo {} | sudo -S ffmpeg -i udp://<source_ip>:<port>  -c:v copy -c:a aac -f mpegts udp://<multicast_ip>:<port>".format(password)
command4 = "echo {} | sudo -S gopro webcam".format(password)

# Use subprocess to execute the commands
process1 = subprocess.Popen(command1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
process2 = subprocess.Popen(command2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
process3 = subprocess.Popen(command3, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
process4 = subprocess.Popen(command4, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

# Get the output and errors, if any, from the first command
output1, error1 = process1.communicate()

# Print the output
print("Output: ", output1.decode())

# Print the error
if error1:
    print("Error: ", error1.decode())

# Get the output and errors, if any, from the second command
output2, error2 = process2.communicate()

# Print the output
print("Output: ", output2.decode())

# Print the error
if error2:
    print("Error: ", error2.decode())

# Get the output and errors, if any, from the third command
    
output3, error3 = process3.communicate()

# Print the output

print("Output: ", output3.decode())

# Print the error

if error3:
    print("Error: ", error3.decode())

# Get the output and errors, if any, from the fourth command
        
output4, error4 = process4.communicate()

# Print the output

print("Output: ", output4.decode())

# Print the error

if error4:
    print("Error: ", error4.decode())