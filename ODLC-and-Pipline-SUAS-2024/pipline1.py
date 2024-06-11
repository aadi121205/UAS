from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import subprocess
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image       
import numpy as np
import webcolors
import random
from ultralytics import YOLO
modle_rpn = YOLO('best.pt')
model=YOLO('last.pt')

vehicle_ip = "127.0.0.1:14555"

vehicle = connect(vehicle_ip,wait_ready=False)

i=1


def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def rgb_to_color_name(rgb):
    color_name = closest_color(rgb)
    return color_name

def create_color_masks(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to be a simple list of RGB pixels
    pixels = image.reshape(-1, 3)

    # Convert to floating point
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    K = 5  # Number of clusters
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image
    segmented_image = segmented_image.reshape(image.shape)

    # Get the center of the image
    center_x, center_y = segmented_image.shape[0] // 2, segmented_image.shape[1] // 2

    # Get the colors at the center of the image
    center_colors = segmented_image[center_x-10:center_x+10, center_y-10:center_y+10].reshape(-1, 3)

    # Get the two most common colors
    unique_colors, counts = np.unique(center_colors, return_counts=True, axis=0)
    most_common_colors = unique_colors[np.argsort(-counts)][:3]

    masks = []
    for color in most_common_colors:
        lower = np.array(color, dtype = "uint8")
        upper = np.array(color, dtype = "uint8")
        mask = cv2.inRange(segmented_image, lower, upper)
        masks.append(mask)

    return masks, most_common_colors

def is_color_at_edges(mask, threshold=5):
    """Check if a color is present at the sides of a mask."""
    left_edge = mask[:, :threshold].flatten()
    right_edge = mask[:, -threshold:].flatten()

    edges = np.concatenate([left_edge, right_edge])
    num_white_pixels = np.sum(edges == 255)
    total_pixels = edges.size

    z = (num_white_pixels / total_pixels) * 100

    if z > 5:
        return True
    else:
        return False

def arm_and_takeoff(targetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """
    print("Basic pre-arm checks")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print (" Waiting for vehicle to initialise...")
        time.sleep(1)

    print ("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode    = VehicleMode("GUIDED")
    vehicle.armed   = True

    # Confirm vehicle armed before attempting to take off
    while not vehicle.armed:
        print (" Waiting for arming...")
        time.sleep(1)

    print ("Taking off!")
    vehicle.simple_takeoff(targetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print (" Altitude: ", vehicle.location.global_relative_frame.alt)
        #Break and return from function just below target altitude.
        if vehicle.location.global_relative_frame.alt>=targetAltitude*0.95:
            print ("Reached target altitude")
            break
        time.sleep(1)

def haversine(lat1, lon1, lat2, lon2):
     
    # distance between latitudes
    # and longitudes
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0
 
    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0
 
    # apply formulae
    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) *
             math.cos(lat1) * math.cos(lat2))
    rad = 6371
    c = 2 * math.asin(math.sqrt(a))
    return rad * c * 1000

def flyMission():
    global i
    arm_and_takeoff(10)
    with open('waypoints.txt', 'r') as file:
        for line in file:
            lat, lon, alt = line.strip().split(',')
            target_location = LocationGlobalRelative(float(lat), float(lon), float(alt))
            vehicle.simple_goto(target_location)

            uav_lat, uav_lon = vehicle.location.global_frame.lat, vehicle.location.global_frame.lon
            subprocess.call('python3 doit.py', shell=True)

            print(f"Distance to target: {remaining_distance}m")

            while True:
                uav_lat, uav_lon = vehicle.location.global_frame.lat, vehicle.location.global_frame.lon

                remaining_distance = haversine(float(lat), float(lon), uav_lat, uav_lon)

                if remaining_distance < 1:
                    break
            
            print("Reached target")
            time.sleep(1)
            #replace with zoom
            target_location = LocationGlobalRelative(float(lat), float(lon), float(10))
            vehicle.simple_goto(target_location)
            time.sleep(10)
            uav_lat, uav_lon, uav_alt = vehicle.location.global_frame.lat, vehicle.location.global_frame.lon, vehicle.location.global_frame.alt
            print(f"UAV location: {uav_lat}, {uav_lon}, {uav_alt}")
            with open('sd.txt', 'w') as file:
                file.write(str(i))
            subprocess.call('python3 doit.py', shell=True)
            time.sleep(10)
            i += 1

flyMission()
vehicle.mode= VehicleMode("RTL")
