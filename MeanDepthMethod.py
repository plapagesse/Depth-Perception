import torch
from torchvision import transforms, utils
import torchvision
from PIL import Image
from io import BytesIO
import numpy as np
import os

'''
Calculates the mean depth of set of training data
'train_data' is a list
'''
def calculate_mean_depth(train_data):
    sum = torch.zeros_like(train_data[0]['scene_depth'])
    count = 0
    for point in train_data:
        sum += point['scene_depth']
        count += 1
    return sum/count

'''
Returns the average_relative_error and root_mean_squared_error given a set of testing data and a mean depth map
'test_data' is a list and 'mean_depth' is the output of 'calculate_mean_depth'
'''
def get_naive_method_error(test_data, mean_depth):
    total_relative_error = 0
    total_squared_error = 0
    num_points = 0
    
    for point in test_data:
        default = point['scene']
        truth = point['scene_depth']
        masks = point["scene_depth_mask"]
    
        #predict
        prediction = mean_depth
    
        # Calculate the averge relative error
        relative_error = torch.abs(prediction - truth) / torch.clamp(truth, min=1e-6)  # Avoid division by zero
        total_relative_error += torch.sum(relative_error).item()
        num_points += torch.numel(truth)
    
        # Calcuklate the root mean squared error
        squared_error = (prediction - truth) ** 2
        total_squared_error += torch.sum(squared_error).item()

    average_relative_error = total_relative_error / num_points
    root_mean_squared_error = np.sqrt(total_squared_error / num_points)
    return average_relative_error, root_mean_squared_error