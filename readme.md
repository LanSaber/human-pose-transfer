# Introduction

This program can map an SMPL-X style pose data on to the character getting from https://www.mixamo.com/#/

# Setup
```
conda create -n hpt python=3.9
conda activate hpt
pip install -r requirement.txt
```

# Run Steps

Download Pose data from https://drive.google.com/file/d/1DqsJhRAC2fLMC65YWwSZEmvHUBskv1fI/view?usp=drive_link and put it under the directory
./pose_data

```
python animation.py
```

# Formula of Linear Skin Blending

Each joint matrix in .dae file is the inverse of the bone’s bind-pose transform, which is a matrix transform the original position of the vertex from the world space to the joint space.
Denote the inverse matrix as of the $i$-th joint is $B_{j}^{-1}$. The animation matrix of this joint in $t$-th frame is noted as $L_{j}(t)$, which is a rotation matrix in joint space. The default position of the binded vertex is $p_i$ in world space.

Therefore, the position of the vertex in the world space in this frame is calculated as

$$  \mathbf{v}_i(t)=
  \sum_{j=1}^{k_i}
    w_{ij}\,
    \bigl(M_j(t)\,B_j^{-1}\bigr)
    \begin{bmatrix}
      \mathbf{p}_i\\
      1
    \end{bmatrix}$$

Where $M_j(t)=M_{\text{parent}(j)}(t)L_(j)(t)$, $M_{root}(t)=L_{root}(t)$, $w_{ij}$ is the weight determine how much the $j$-th joint will influence the position of this vertex. And this vertex has been binded with $k_i$ number of joints.

# Reference
This project is based on https://github.com/xing-shuai/PyOpenGL-skeleton-aniamtion