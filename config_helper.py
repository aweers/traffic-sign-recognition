import json
import os
from sys import exit

def read_config(path):
    with open(path) as json_file:
        config = json.load(json_file)
        
        if not 'data' in config:
            print("Please specify a 'data' path in the config file")
            exit(1)
        elif not os.path.isdir(config['data']):
            print("Please specify a valid path for the 'data' key in the config file")
            exit(1)

        if not 'preprocessing' in config:
            # specify the amount of pixels to crop of the image
            config['preprocessing'] = {
                'crop_left': 0,
                'crop_right': 0,
                'crop_top': 0,
                'crop_bottom': 0
            }
        elif not 'crop_left' in config['preprocessing']:
            config['preprocessing']['crop_left'] = 0
        elif not 'crop_right' in config['preprocessing']:
            config['preprocessing']['crop_right'] = 0
        elif not 'crop_top' in config['preprocessing']:
            config['preprocessing']['crop_top'] = 0
        elif not 'crop_bottom' in config['preprocessing']:
            config['preprocessing']['crop_bottom'] = 0
        
        if not 'gaussian' in config:
            config['gaussian'] = {
                'ksize': 5,
                'sigma': 0
            }
        elif not 'ksize' in config['gaussian']:
            config['gaussian']['ksize'] = 5
        elif not 'sigma' in config['gaussian']:
            config['gaussian']['sigma'] = 0
        
        if not 'hough' in config:
            config['hough'] = {
                'dp': 0.5,
                'min_distance': 20,
                'param1': 50,
                'param2': 50,
                'min_radius': 10,
                'max_radius': 50
            }
        elif not 'dp' in config['hough']:
            config['hough']['dp'] = 0.5
        elif not 'min_distance' in config['hough']:
            config['hough']['min_distance'] = 20
        elif not 'param1' in config['hough']:
            config['hough']['param1'] = 50
        elif not 'param2' in config['hough']:
            config['hough']['param2'] = 50
        elif not 'min_radius' in config['hough']:
            config['hough']['min_radius'] = 10
        elif not 'max_radius' in config['hough']:
            config['hough']['max_radius'] = 50
        
        if not 'simple_cnn_model' in config:
            config['simple_cnn_model'] = '../models/simple_cnn.h5'
        
        if not 'advanced_cnn_model' in config:
            config['advanced_cnn_model'] = '../models/advanced_cnn.h5'

        if not 'video_fps' in config:
            config['video_fps'] = 30

        if not 'traffic_signs' in config:
            print("Please specify a 'traffic_signs' path in the config file")
            exit(1)
        elif not os.path.isdir(config['traffic_signs']):
            print("Please specify a valid path for the 'traffic_signs' key in the config file")
            exit(1)
        
        if not 'sign_threshold' in config:
            config['sign_threshold'] = 0.5
        
        if not 'sign_crop_width' in config:
            config['sign_crop_width'] = 80
        
        if not 'sign_crop_height' in config:
            config['sign_crop_height'] = 80
    
    return config