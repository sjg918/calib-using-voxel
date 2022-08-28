# mkscene_xxx.py </br>
Store preprocessed data. Preprocessed data is saved in the 'gendata' folder. </br>
For kitti dataset, stereo matching is required. See readme.md in /src/sgmgpu/ path. </br>

# train_xxx.py </br>
We do not support multi gpu learning. </br>
A gpu with more than 11gb of memory is required. </br>
To train 5 networks, 5 trainings are required. (and proper cfg file required) </br>
It takes more than 2 days to train 1 network. </br>

# val_xxx_it.py </br>
Evaluation with iterative refinement applied. Results will vary from run to run. </br>

# val_xxx_tf_v1.py </br>
Evaluation with iterative refinement and temporal filtering applied. </br>
The entire data set is used for temporal filtering. Results will be similar from run to run. </br>
It takes more than 6 hours to run this on the kitti dataset. </br>

# val_xxx_tf_v2.py </br>
Evaluation with iterative refinement and temporal filtering applied. </br>
Pre-determined frames (eg 10, 25, 100) are used for temporal filtering. Results will be similar from run to run. </br>

# vis_xxx.py </br>
Create visual results. </br>
The kitti data is obtained as an image as a result. </br>
The oxford data get bin file as output. </br>
Therefore, use a separate library for oxford data. (Example: open3d, pptk) </br>
