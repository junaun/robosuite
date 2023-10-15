#!/bin/bash

# Run the first Python file
python lift_camera_fixboxpose_16.py
python lift_camera_fixboxpose_16.py test

# Run the second Python file
python lift_camera_randomboxpose_16.py
python lift_camera_randomboxpose_16.py test

# Run the third Python file
python twoarm_camera_fixedrobotpos_16.py
python twoarm_camera_fixedrobotpos_16.py test

# Run the fourth Python file
python twoarm_camera_randomrobotpos_16.py
python twoarm_camera_randomrobotpos_16.py test
