# Introduction

This program can map an SMPL-X style pose data on to the character getting from https://www.mixamo.com/#/

# Setup
```
conda create -n hpt python=3.9
conda activate hpt
pip install -r requirement.txt
```

# Run Steps

```
python video_data_vis.py
```

Firstly, you will see an opengl window, and later, a Qt-style window will open. You can choose the video from ./videos in this new window. Then the program will try to load the pose data from
./pose_data based on the video you choose and map the pose on to the animation character in ./resources.

I extract the video pose from the video by using OSX-sign. Because it is from an unpublished paper, as requested by the authors, I cannot publish the pose extraction codes.

# Reference
This project is based on https://github.com/xing-shuai/PyOpenGL-skeleton-aniamtion