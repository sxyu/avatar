## OpenARK Avatar Development Project

A smaller reimplementation of OpenARK Avatar using only analytic derivatives.

## Building

### Dependencies
- OpenCV 3
- PCL 1.8+ (and its dependenciesa)
- Eigen 3
- Ceres Solver

### How to build

If you haven't already, install CMake.

```sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```
Replace `4` with an appropriate number of threads.

### Outputs

- `smplsynth` : from `smplsynth.cpp`. Synthetic human dataset generator
- `scratch` : from `scratch.cpp`. Currently configured to show human avatar when ran, with (limited) options to adjust pose and shape. Generally, used for scratch.
- `optim` : from `optim.cpp`. Currently optimizes avatar pose to fit a synthetic point cloud.
- `libsmplsynth.a` : the static library which the above depend on. I configure the project like this to improve build times when editing different outputs.

### Getting model data

Please get the data from me via email. (This is not allowed to be shared so I am not putting the link here). Then put it in this directory:
`<smplsynth-repo-root>/data`

So that the following exists:
`<smplsynth-repo-root>/data/avatar-model/skeleton.txt`

## License
Apache 2.0
