## AR/VR *Avatar* Project: Fitting SMPL body model to depth data in real time on CPU (Fall 2019)

**Demo video**: <https://drive.google.com/file/d/1KQ0g_R77x80c6WbFKTXefvsNO9F1ITxW/view?usp=sharing>

**Contains**
- SMPL model loader and representation in C++ (AvatarModel/Avatar)
- Fast SMPL parameter optimizer (wrt. a point cloud) based on Ceres-solver (AvatarOptimizer)
- Real-time human body segmentation system using random forest, with weights provided
   - Custom random forest implementation and parallelized training system provided
- Basic first-frame background subtraction system (BGSubtract)

![Pipeline](https://raw.githubusercontent.com/sxyu/smplsynth/master/images/pipeline.png)

![Demo Screenshot (Quite Old)](https://raw.githubusercontent.com/sxyu/smplsynth/master/images/result.png)

A smaller reimplementation of OpenARK Avatar using only analytic derivatives.

## Building

### Dependencies
- Boost 1.58
- OpenCV 3.3+ (OpenCV 4 not supported)
- Eigen 3.3.4
- Ceres Solver 1.14 (Ceres 2 not supported).
    - This is very performance critical, and it is strongly recommended to manually build Ceres with LAPACK and OpenMP support.   
    - If you are using an Intel processor, it is also recommended to use MKL as BLAS/LAPACK. Otherwise ATLAS is recommended.
    - Finally, make sure you build Ceres in release mode.
- K4A (Azure Kinect SDK), optional but required for live-demo
- PCL 1.8+, optional

Earlier versions of these libraries may work, but I have not tested them

### How to build

If you haven't already, install CMake.

```sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```
Replace `4` with an appropriate number of threads. Add `-DWITH_PCL=ON` to enable PCL, add `-DWITH_K4A=OFF` to disable looking for Azure Kinect SDK, add `-DBUILD_RTREE_TOOLS=OFF` to disable building RTree tools such as rtree-train, rtree-run-dataset.

For unknown reasons, sometimes I encounter linker errors when not manually linking OpenMP. If this happens configure with `-DWITH_OMP=ON`.

### Outputs

#### Core
- `live-demo`: from `live-demo.cpp`. Live demo, runs the system end-to-end on Azure Kinect camera input. Requires K4A library to be installed
- `bgsubtract` : from `bgsubtract.cpp`. Somewhat of a misnomer, runs the system end-to-end on an OpenARK dataset in standard format (depth_exr, etc)
- `data-recording` : from `DataRecording.cpp`. Tool for recording datasets from the Azure Kinect camera. Mostly copied from OpenARK, but fixes memory bug.
- `libsmplsynth.a` : the static library which the above depend on. I configure the project like this to improve build times when editing different outputs.

#### SMPL Model Tools
- `smplsynth` : from `smplsynth.cpp`. Synthetic human dataset generator
- `smpltrim` : fom `smpltrim.cpp`. A tool for generating partial SMPL models, including creating a smaller model with a specific joint as root, or cutting off limbs

#### Random Forest Tools
- `rtree-train`: from `rtree-train.cpp`. High performance random tree trainer. Find trained trees in releases on Github
- `rtree-transfer`: from `rtree-transfer.cpp`. Tool to refine a trained random tree by recomputing leaf distributions over a huge amount of images.
- `rtree-run`: from `rtree-run.cpp`. Run rtree on images (not important).
- `rtree-run-dataset`: from `rtree-run-dataset.cpp`. Run rtree on OpenARK dataset in standard format (depth_exr, etc)

#### Miscellaneous
- `scratch` : from `scratch.cpp`. Currently configured to show human avatar when ran, with (limited) options to adjust pose and shape. Generally, used for scratch.
- `optim` : from `optim.cpp`. **Currently disabled** since not updated after API change; optimizes avatar pose to fit a synthetic point cloud.

### Getting model data

Please get the data from me via email sxyu (at) berkeley.edu. (This is not allowed to be shared so I am not putting the link here). Then put it in this directory:
`<smplsynth-repo-root>/data`

So that the following exists:
`<smplsynth-repo-root>/data/avatar-model/skeleton.txt`

## License
Apache 2.0
