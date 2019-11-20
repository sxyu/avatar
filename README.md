## OpenARK Avatar Development Project

A smaller reimplementation of OpenARK Avatar using only analytic derivatives.

## Building

### Dependencies
- OpenCV 3
- PCL 1.8+ (and its dependencies; may remove this later since not used much)
- Eigen 3.4
- Ceres Solver
- K4a (Azure Kinect SDK), optional

### How to build

If you haven't already, install CMake.

```sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```
Replace `4` with an appropriate number of threads.

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
