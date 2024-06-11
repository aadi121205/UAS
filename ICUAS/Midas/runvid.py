import cv2
import cv2
import torch
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def main():

    # Open the video file
    cap = cv2.VideoCapture("in.mp4")

    # Get the video's width, height, and frames per second (fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform


    while cap.isOpened():
        
        ret, frame = cap.read()

        cv2.imshow('frame_og', frame)

        if not ret:
            break

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply the transformations
        input_batch = transform(frame).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

        output = prediction.cpu().numpy()

        # Scale the output array to the [0, 255] range
        output_scaled = ((output - output.min()) * (255 / (output.max() - output.min()))).astype(np.uint8)

        # Write the frame to the output video 
        
        cv2.imshow('frame', output_scaled)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            continue

        out.write(output_scaled)

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

    # Close all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    for i in tqdm(range(10)):
        main()
        print(i)
        time.sleep(31)