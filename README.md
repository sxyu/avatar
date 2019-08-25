## Fast marching method

- To use in a project, include fmm.hpp and use `fmm::fmm<image_data_type>(image, seeds, weight_map_type[, image_segmentation_threshold])`

- This is header-only and you do not need to build this, unless you want to build the sample program.

- All functions are templated to be OpenCV Mat compatible, also includes bare-bones Image struct which allows you to use any row-major contiguous matrix type: `Image<filed_type>(rows, cols, data_ptr)`

Comment on top of fmm.hpp copied here:
```cpp
/** Fast marching method implementation
 *  usage: fmm::fmm(image, seeds, weight_map_type = IDENTITY,
 *                  segmentation_threshold = disabled,
 *                  normalize_output_geodesic_distances = true,
 *                  output = nullptr)
 *  > returns an image, either geodesic distance map or, if
 *    segmentation_threshold is given, a segmentation mask
 *  image: input image (can be OpenCV Mat or fmm::Image<T>)
 *  seeds: std::vector of points (each can be OpenCV Point or fmm::Point)
 *  weight_map_type: transformation to apply to input image to use as FMM
*                    weight function. Can be one of:
 *                    fmm::weight::IDENTITY  no transformation (avoids a copy)
 *                    fmm::weight::GRADIENT  gradient magnitude (using Sobel) 
 *                    fmm::weight::ABSDIFF   absolute difference from average
 *                                           grayscale value of seeds
 *                    fmm::weight::LAPLACIAN image Laplacian magnitude
 *  segmentation_threshold: if specified, sets pixels with geodesic value less
 *                          than or equal to this threshold to 1 and others to 0
 *                          in the output image. If not given,the geodesic
 *                          distances map will be returned.
 *  normalize_output_geodesic_distances: if true, normalizes geodesic distances
 *                                       values to be in the interval [0, 1].
 *                                       If segmentation_threshold is specified,
 *                                       this occurs prior to segmentation.
 *                                       Default true.
 *  output: optionally, a pointer to an already-allocated output image.
 *          This allows you to avoid a copy if you already have one
 *          allocated. By default a new image is created, and this
 *          is not necessary.
 *
 *  fmm::Image<T> usage (optional)
 *  - To make owning image fmm::Image<T>(rows, cols)
 *  - To make non-owning image that maps to row-major data (of same type, or char/uchar):
 *    fmm::Image<T>(rows, cols, data_ptr)
 * */
```

## License

Apache 2.0
