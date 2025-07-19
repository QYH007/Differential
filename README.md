## Differentiable Geometry Simplification

## Start

Run app/app.py to run a optimization job in default setup. You are welcome to modify the initial state part in the code.

For example, change model_name, LOD_level to define job.

The output would be stored in folder './opt_result'.

Run app/app_experiment_B.py to run experiment B described in paper. Notice that we need to repeat three times for each candidate for stable line, we decrease the iteration number to 1000 for time saving in this demo(while 4000 iteration in the paper). However, it still take around 20 min on my computer.

## Licenses

This work in done under the licenses of Nvdiffrast, NVIDIA.

Copyright (c) 2020, NVIDIA Corporation. All rights reserved.

See LICENSE.txt
