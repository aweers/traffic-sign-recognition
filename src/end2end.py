import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time

import os
import sys

# add parent directory to python search path
sys.path.append(os.path.abspath('../'))

import config_helper

config = config_helper.read_config('../config.json')

# Set to True to write the video to a file
# otherwise the video will be shown in a window
WRITE_VIDEO = True
OUTPUT_FILE = 'output.mp4'

# The class names can be obtained from the model_training notebook
class_names = ['10', '100', '30', '40', '50', '60', '70', '80', 'other']

# Path to the frames
frames_path = config['data'] + 'video01/'

frames = os.listdir(frames_path)
frames = [frames_path + f for f in frames]
frames.sort()

# Load the model
model = tf.keras.models.load_model(config['advanced_cnn_model'])

if not WRITE_VIDEO:
    # Initialize the plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    plot_object = ax1.imshow(cv2.imread(frames[0]))
    fig.canvas.draw()

    plt.show(block=False)

# Simple weighted average of sign predictions / exclude 'other'
sign = [0] * (len(class_names) - 1)

if WRITE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, config['video_fps'], cv2.imread(frames[0]).shape[:2][::-1])

half_width = int(config['sign_crop_width'] / 2)
half_height = int(config['sign_crop_height'] / 2)

# Keep track of the current frame
timestamp = 0
start_time = time.time()

while timestamp < len(frames):
    # Load current frame
    frame = cv2.imread(frames[timestamp])

    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    gray = gray[
        config['preprocessing']['crop_top']:height - config['preprocessing']['crop_bottom'], 
        config['preprocessing']['crop_left']:width - config['preprocessing']['crop_right']
    ]

    # Add Gaussian blur
    gray = cv2.GaussianBlur(gray, (config['gaussian']['ksize'], config['gaussian']['ksize']), config['gaussian']['sigma'])
    
    # Detect circles in the image using the Hough transform
    circles = cv2.HoughCircles(gray, 
        cv2.HOUGH_GRADIENT, 
        config['hough']['dp'], 
        config['hough']['min_distance'], 
        param1=config['hough']['param1'], 
        param2=config['hough']['param2'], 
        minRadius=config['hough']['min_radius'], 
        maxRadius=config['hough']['max_radius'])

    # If circles were detected
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype('int')

        candidates = []
        # Loop over the circles
        for (x, y, r) in circles:
            # Crop an image around the circle
            frame_x = config['preprocessing']['crop_left'] + x
            frame_y = config['preprocessing']['crop_top'] + y
            crop = frame[frame_y - half_height :frame_y + half_height, frame_x - half_width:frame_x + half_width]

            # Pad the image if it is too small
            if crop.shape[0] < config['sign_crop_height'] or crop.shape[1] < config['sign_crop_width']:
                crop = cv2.copyMakeBorder(crop, 0, config['sign_crop_height'] - crop.shape[0], 0, config['sign_crop_width'] - crop.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            # mask the background
            mask = np.zeros(crop.shape[:2], np.uint8)
            cv2.circle(mask, (half_width, half_height), r, (255, 255, 255), -1)
            masked = cv2.bitwise_and(crop, crop, mask=mask)

            candidates.append(masked)
        
        # Predict the class of each candidate
        predictions = model.predict(np.array(candidates), verbose=False)
        predictions = np.argmax(predictions, axis=1)

        if predictions[0] != 8:
            # Update weighted average
            sign[predictions[0]] = sign[predictions[0]] * 0.8 + 1 * 0.2

            # add rectangle around the sign
            frame_x = config['preprocessing']['crop_left'] + circles[0][0]
            frame_y = config['preprocessing']['crop_top'] + circles[0][1]
            r = circles[0][2]
            cv2.rectangle(frame, (frame_x - r, frame_y - r), (frame_x + r, frame_y + r), (0, 255, 0), 4)
            cv2.putText(frame, class_names[predictions[0]], (frame_x - r, frame_y - r), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Update all weighted averages
    for i in range(len(sign)):
        sign[i] = sign[i] * 0.95
    
    # Print sign, if weighted average is above threshold
    for i in range(len(sign)):
        if sign[i] > config['sign_threshold']:
            print(class_names[i])

    # Add frame to video / show frame
    if WRITE_VIDEO:
        out.write(frame)
        timestamp += 1
        
        if timestamp % 30 == 0:
            print(timestamp)
    else:
        plot_object.set_data(frame)
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # calculate the next frame number to keep the video in real time
        timestamp = int((time.time() - start_time) * config['video_fps'])

if WRITE_VIDEO:
    out.release()