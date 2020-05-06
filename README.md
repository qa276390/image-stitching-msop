# <p style="text-align: center;"> Image Stitching </p> 
This is a python project to combine several images into a larger image.

## Table of Contents

- [Project Description](#project-description)
- [Algorithms](#algorithms)
- [Usages](#usages)
- [Results](#results)
- [Acknowledgements and Links](#acknowledgements-and-Links)

## Project Description
There are several steps(shown below) including feature detection, feature description, feature matching, alignment and blending, we use  *MSOP([Multi-Image Matching using Multi-Scale Oriented Patches](http://matthewalunbrown.com/papers/cvpr05.pdf))* as [Feature Descriptor](#feature-descriptor) and use *KNN(k nearest neighbor)* with *RANSAC* to do the [Feature Matching](#feature-matching). We also implemented [Alignment and Blending](#alignment-and-blending) for images to show our result.





## Algorithms 
### Feature Descriptor
We implemented the *MSOP* to do the feature dectection and also use *non maximumal suppression* to make sure our features are well distributed. 
Here are some example:

<center><img src="https://i.imgur.com/n4a6lPE.png" alt="drawing" width="500"/></img></center>
<p style="text-align: center;">figure 1: image pyramid's feature map detect by harris corner detector</p> 


### Feature Matching
In this part, we use K nearest neighbor with `k = 2` to find their matching. To make sure we choose the good matching pairs, we also implemented ***David Lowe’s ratio test*** shown in function `feature_matching()` in `utils/stitch.py`.
Here are some example:

<center><img src="https://i.imgur.com/IEX34WT.png" alt="drawing" width="500"/></img></center>
<p style="text-align: center;">figure 2: feature matching in differnent level of pyramid</p> 

### Alignment and Blending
After we have the matching pair, we can caluculate the tanslation between image. To get the best motion models, we implemented ***RANSAC*** shown in function `pairwise_alignment(...)` in `utils/stitch.py`. Then, we can do the blending, the result is not quite good until we modified the blending weighted in different dimension according to the magnitude of its motion parameter.


## Usages
There are *python file(`main.py`)* and *ipython notebook(`Stitching.ipynb`)* for you to choose.
### Prepare Images and Meta Data
Put your images in a single folder and prepare your meta data file. The meta file should contains filename and focal length separated with spaces.(see `./images/yard-002/pano.txt`)
### Start 
here is an example to run the code.
```shell
python3 main.py --img-dir ./images/yard-001/ --meta-path ./images/yard-001/pano.txt
```
to see more parameters
```shell
python3 main.py --help
```

## Results
### Original Image
<table>
<tr>
<th><br>....</th>
<th><img src="https://i.imgur.com/sWvFyJ9.jpg" /><br>image010</th>
<th><img src="https://i.imgur.com/8rbbscW.jpg" /><br>image011</th>
<th><img src="https://i.imgur.com/qNzxSyR.jpg" /><br>image012</th>
<th><br>....</th>
</tr>
</table>

### Stitching Image
<img src="https://i.imgur.com/b8hCK6y.jpg" width="1000"/>


## Acknowledgements and Links
- [Digital Visual Effects](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/20spring/overview/)
- [Github Code](https://github.com/qa276390/image-stitching-msop) for this Project
