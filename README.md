
# Online Self-Calibration of 3D Measurement Sensors Using a Voxel-Based Network
Implementation of the paper “Online Self-Calibration of 3D Measurement Sensors Using a Voxel-Based Network”.
[paper link](https://www.mdpi.com/1424-8220/22/17/6447)

# Problem 
There is a problem with equation (1) in the paper.</br>
"Y = base*(v-c_v)/disp" is incorrect. . . .</br>
"Y = f_u*base *(v-c_v)/(f_v *disp)"is right. . . .</br>
![joatmang](https://github.com/sjg918/calib-using-voxel/blob/main/image.png?raw=true)</br>
I'm sorry for the inconvenience. But the code in this repository is fine.</br>

# Weights
[kitti-20](https://drive.google.com/file/d/1NLUlefJneNaEobW2UAOfEuaAeQy2QMDJ/view?usp=sharing)
[kitti-10](https://drive.google.com/file/d/1wYBOvj-7mnxLn9sHM64SLcbbvo7yHmVI/view?usp=sharing)
[kitti-5](https://drive.google.com/file/d/1AthkLpAqcSC9ISgS15GyUOp2rOq2NMue/view?usp=sharing)
[kitti-2](https://drive.google.com/file/d/1n3WNJm3KWww4FnJLyNc9HDCiV5paHYyv/view?usp=sharing)
[kitti-1](https://drive.google.com/file/d/1UBlR7DXxaHBPAL0APhhFkdFA6Z18YT5S/view?usp=sharing)
</br>
[oxford-20](https://drive.google.com/file/d/1IeAwl52uiWoeLV31AXfLmP5ycqPj1wFl/view?usp=sharing)
[oxford-10](https://drive.google.com/file/d/1hm1SUMxjWxZniyfiC2B1ZPEvA6o5WrW0/view?usp=sharing)
[oxford-5](https://drive.google.com/file/d/1MUiRXbYlGBVyQAJSjoLTogJg6e_c-0dX/view?usp=sharing)
[oxford-2](https://drive.google.com/file/d/10Nnd1422X4exfgXmWc7-afQr1setLQlr/view?usp=sharing)
[oxford-1](https://drive.google.com/file/d/1zSUcMKMNmxVRIwWgM3wJhrebtvuwZALy/view?usp=sharing)

# Setup
Download the kitti odometry dataset. </br>
Replace all calib.txt in data folder with calib.txt downloaded from "Download odometry data set (calibration files, 1 MB)". </br>
(('Tr') line does not exist, giving an error.) </br>
Download the oxford radar robotcar dataset. </br>
Download left LiDAR and right LiDAR. </br>
Download Bumblebee XB3 Visual Odometry. </br>
I used 2019-01-10-12-32-52 for training and 2019-01-17-12-48-25 for testing. </br>

Open the cfg files in the CFG folder and modify the following two paths. </br>
cfg.odometry_home = '/home/jnu-ie/Dataset/odometry/' </br>
cfg.proj_home = '/home/jun-ie/kew/calib-using-voxel/' </br>

# Requirement
python >= 3.9.7. I worked in an anaconda environment. </br>
pytorch >= 1.10.2 (https://pytorch.org/) </br>
spconv >= 2.1.21 (https://github.com/traveller59/spconv) </br>
opencv (pip install opencv-python) </br>
pykitti (pip install pykitti) </br>
easydict (pip install easydict) </br>
numba >= 0.55.1 (pip install numba) </br>
matplotlib (pip install matplotlib) </br>

GPU with more than 11 GB of memory </br>

# Run
Check out the readme.md in /scripts

# Results
![kitti-input](https://github.com/sjg918/calib-using-voxel/blob/main/results/input.png?raw=true)
![kitti-pred](https://github.com/sjg918/calib-using-voxel/blob/main/results/pred01.png?raw=true)
</br>
![oxford-input](https://github.com/sjg918/calib-using-voxel/blob/main/results/2022-08-28%2015-26-46.png?raw=true)
![oxford-pred](https://github.com/sjg918/calib-using-voxel/blob/main/results/2022-08-28%2015-26-24.png?raw=true)
</br>

# Special Thanks
I got a lot of help from the public implementation of [LCCNet](https://github.com/LvXudong-HIT/LCCNet) and [CIA-SSD](https://github.com/Vegeta2020/CIA-SSD). </br>
Thanks to LvXudong-HIT and Vegeta2020.
