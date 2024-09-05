#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install opencv-python


# In[4]:


import torchvision.models as models
# Old way
model = models.resnet18(pretrained=True)

# New way
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


# In[5]:


model = models.resnet18(weights=ResNet18_Weights.DEFAULT)


# In[8]:


import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    print("Camera is working")
    print(f"Frame shape: {frame.shape}")
else:
    print("Failed to capture image from camera")

cap.release()


# In[9]:


import sys
import cv2
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")


# In[10]:


import cv2

def list_ports():
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
        else:
            is_reading, img = camera.read()
            if is_reading:
                working_ports.append(dev_port)
            else:
                available_ports.append(dev_port)
        dev_port +=1
    return working_ports, available_ports

working_ports, available_ports = list_ports()
print(f"Working ports: {working_ports}")
print(f"Available (but not working) ports: {available_ports}")


# In[11]:


import subprocess
import re

def get_camera_index():
    result = subprocess.run(['system_profiler', 'SPCameraDataType'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    cameras = re.findall(r'Camera ID: (.*)', output)
    if cameras:
        return int(cameras[0])
    return None

camera_index = get_camera_index()
print(f"Detected camera index: {camera_index}")

if camera_index is not None:
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    if ret:
        print("Successfully captured an image from the camera")
    else:
        print("Failed to capture an image from the camera")
    cap.release()
else:
    print("No camera detected")


# In[12]:


import cv2
import numpy as np
from IPython.display import display, Image

def capture_and_display(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Failed to open camera at index {camera_index}")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print(f"Successfully captured image from camera at index {camera_index}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, jpeg = cv2.imencode('.jpg', frame_rgb)
        display(Image(data=jpeg.tobytes()))
    else:
        print(f"Failed to capture image from camera at index {camera_index}")

# Try both index 0 and 1
print("Trying camera index 0:")
capture_and_display(0)

print("\nTrying camera index 1:")
capture_and_display(1)


# In[24]:


import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from PIL import Image

# (Keep the ProductivityNet class and other function definitions as they were)

def main():
    cap = cv2.VideoCapture(1)  # Use camera index 1
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('productivity_tracker.mp4', fourcc, fps, (frame_width, frame_height))

    start_time = time.time()
    productivity_data = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            faces = detect_faces(frame)
            current_time = time.time() - start_time
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                gaze_x, gaze_y, focus_score = estimate_gaze_and_focus(face_img)
                
                # Update productivity data
                productivity_data.append(focus_score)
                
                # Draw face rectangle
                color = (0, 255, 0) if focus_score > 0.5 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Check if person is talking
                if is_person_talking(face_img):
                    if talking_start_time is None:
                        talking_start_time = time.time()
                    elif time.time() - talking_start_time > 120:  # 2 minutes
                        cv2.putText(frame, "Distracted: Talking too long", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    talking_start_time = None
            
            # Add productivity chart to the frame
            if len(productivity_data) > 0:
                chart = update_productivity_chart(productivity_data)
                chart = cv2.resize(chart, (320, 240))
                frame[0:240, 0:320] = chart
            
            # Add timer to the frame
            cv2.putText(frame, f"Time: {int(current_time)}s", (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write the frame to the video file
            out.write(frame)
            
            # Display the frame in the notebook (optional, for real-time feedback)
            clear_output(wait=True)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            display(pil_img)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Optional: Stop after a certain duration (e.g., 60 seconds)
            if current_time > 60:
                break

    finally:
        # Release everything when job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Video recording complete. Saved as 'productivity_tracker.mp4'")

if __name__ == "__main__":
    main()


# In[23]:


import os

print(f"Current working directory: {os.getcwd()}")


# In[16]:


import os

files = os.listdir()
print("Files in the current directory:")
for file in files:
    if file.endswith('.mp4'):
        print(f" - {file} (Video file)")
    else:
        print(f" - {file}")


# In[17]:


import os

video_path = os.path.abspath('productivity_tracker.mp4')
if os.path.exists(video_path):
    print(f"The video file is located at: {video_path}")
else:
    print("The video file was not found in the expected location.")


# In[18]:


import os

current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

if os.access(current_dir, os.W_OK):
    print("We have write permission in this directory.")
else:
    print("We do not have write permission in this directory.")


# In[25]:


import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from PIL import Image
import os

# (Keep the ProductivityNet class and other function definitions as they were)

def main():
    # Specify a save location (e.g., Desktop)
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    video_path = os.path.join(desktop_path, "productivity_tracker.mp4")
    print(f"Attempting to save video to: {video_path}")

    cap = cv2.VideoCapture(1)  # Use camera index 1
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video properties: {frame_width}x{frame_height} at {fps} fps")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not create video file at {video_path}")
        return

    start_time = time.time()
    productivity_data = []
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            frame_count += 1
            
            # (Keep the rest of the processing code as it was)
            
            # Write the frame to the video file
            out.write(frame)
            
            # Optional: Stop after a certain duration (e.g., 10 seconds)
            if time.time() - start_time > 10:
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Release everything when job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        if os.path.exists(video_path):
            print(f"Video recording complete. Saved as '{video_path}'")
            print(f"File size: {os.path.getsize(video_path)} bytes")
            print(f"Frames processed: {frame_count}")
        else:
            print(f"Error: Video file was not created at {video_path}")

if __name__ == "__main__":
    main()


# In[26]:


import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from PIL import Image

# Define the neural network for gaze estimation and focus classification
class ProductivityNet(nn.Module):
    def __init__(self):
        super(ProductivityNet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 3)  # 3 outputs: x, y coordinates of gaze and focus score

    def forward(self, x):
        return self.resnet(x)

# Initialize the model
model = ProductivityNet()
model.eval()

# Define transforms for input images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables for productivity tracking
productivity_data = []
start_time = time.time()
talking_start_time = None
is_talking = False

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def estimate_gaze_and_focus(face_img):
    input_tensor = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    gaze_x, gaze_y, focus_score = output[0].tolist()
    return gaze_x, gaze_y, focus_score

def is_person_talking(frame):
    # This is a placeholder function. In a real implementation, you would use
    # audio processing or more advanced video analysis to detect talking.
    # For demonstration purposes, we'll randomly decide if a person is talking.
    return np.random.random() < 0.1

def update_productivity_chart(productivity_data):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(productivity_data)
    ax.set_title('Productivity Over Time')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Focus Score')
    
    canvas = fig.canvas
    canvas.draw()
    chart = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    chart = chart.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return chart

def main():
    cap = cv2.VideoCapture(1)  # Use camera index 1
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    for _ in range(100):  # Run for 100 frames
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        faces = detect_faces(frame)
        current_time = time.time() - start_time
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            gaze_x, gaze_y, focus_score = estimate_gaze_and_focus(face_img)
            
            # Update productivity data
            productivity_data.append(focus_score)
            
            # Draw face rectangle
            color = (0, 255, 0) if focus_score > 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Check if person is talking
            if is_person_talking(face_img):
                if talking_start_time is None:
                    talking_start_time = time.time()
                elif time.time() - talking_start_time > 120:  # 2 minutes
                    cv2.putText(frame, "Distracted: Talking too long", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                talking_start_time = None
        
        # Add productivity chart to the frame
        if len(productivity_data) > 0:
            chart = update_productivity_chart(productivity_data)
            chart = cv2.resize(chart, (320, 240))
            frame[0:240, 0:320] = chart
        
        # Add timer to the frame
        cv2.putText(frame, f"Time: {int(current_time)}s", (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame in the notebook
        clear_output(wait=True)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        display(pil_img)
        
        time.sleep(0.1)  # Add a small delay to make the output visible
    
    cap.release()

if __name__ == "__main__":
    main()


# In[7]:


import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Define the neural network for gaze estimation and focus classification
class ProductivityNet(nn.Module):
    def __init__(self):
        super(ProductivityNet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 3)  # 3 outputs: x, y coordinates of gaze and focus score

    def forward(self, x):
        return self.resnet(x)

# Initialize the model
model = ProductivityNet()
model.eval()

# Define transforms for input images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables for productivity tracking
productivity_data = []
start_time = time.time()
talking_start_time = None
is_talking = False

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def estimate_gaze_and_focus(face_img):
    input_tensor = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    gaze_x, gaze_y, focus_score = output[0].tolist()
    return gaze_x, gaze_y, focus_score

def is_person_talking(frame):
    # This is a placeholder function. In a real implementation, you would use
    # audio processing or more advanced video analysis to detect talking.
    # For demonstration purposes, we'll randomly decide if a person is talking.
    return np.random.random() < 0.1

def update_productivity_chart(productivity_data):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(productivity_data)
    ax.set_title('Productivity Over Time')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Focus Score')
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    chart = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    chart = chart.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return chart

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detect_faces(frame)
        current_time = time.time() - start_time
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            gaze_x, gaze_y, focus_score = estimate_gaze_and_focus(face_img)
            
            # Update productivity data
            productivity_data.append(focus_score)
            
            # Draw face rectangle
            color = (0, 255, 0) if focus_score > 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Check if person is talking
            if is_person_talking(face_img):
                if talking_start_time is None:
                    talking_start_time = time.time()
                elif time.time() - talking_start_time > 120:  # 2 minutes
                    cv2.putText(frame, "Distracted: Talking too long", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                talking_start_time = None
        
        # Add productivity chart to the frame
        if len(productivity_data) > 0:
            chart = update_productivity_chart(productivity_data)
            chart = cv2.resize(chart, (320, 240))
            frame[0:240, 0:320] = chart
        
        # Add timer to the frame
        cv2.putText(frame, f"Time: {int(current_time)}s", (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Productivity Tracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[ ]:




