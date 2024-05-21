#### Version 0.6.13
- Added image check of converted image
- Added break on error option
- Various improvements on array data and metadata

#### Version 0.6.12
- Simplified dask functions
- Improved image rendering function including performance
- Various bug fixes, better tiff metadata handling
- Experimental code for storing separate tile images as ome-zarr including metadata

#### Version 0.6.11
- Restored TiffSource legacy loading compressed / decompress options

#### Version 0.6.10
- Simplified rgb source rendering
- Improved performance of precise resize operation using skimage

#### Version 0.6.9
- Combine action now working: combine separate channel images into single multichannel image(s)
- Multi-dimensional image generator now working (GeneratorSource)
- Fixed bugs in image conversion, general dask generation, and thumbnails for non-RGB multi-channel images

#### Version 0.6.8
- Simplified slicing in source classes
- Bug fixes in image resizing and dimension orders
- Pinned some versions in conda envs 

#### Version 0.6.7
- Fixed bugs in asdask() and GeneratorSource
- Pinned some versions in pip requirements 

#### Version 0.6.6
- Rewritten TiffSource using dask/zarr functionality
- Added high level Dask function
- Completed Image Generator
- Improved ome-zarr metadata for napari support

#### Version 0.6.5
- Expanded Omero image selection
- Improved image info format, including pyramid sizes
- Various bug fixes

#### Version 0.6.4
- Fixed bug in TiffSource decoding & indexing

#### Version 0.6.3
- Improved Dask support
- Fixed minor issue in channel color conversion

#### Version 0.6.2
- Added Dask converter for ome.tiff reader

#### Version 0.6.1
- Rewritten code base supporting full multi-dimensional images
- Improved (ome) zarr support
- Added convenience rgb render function

#### Version 0.5.3
- Decoupled hard requirements for Omero / bioformats
- Minor fixes to ome-zarr metadata

#### Version 0.5.2
- Bug fix for new imagecodecs compression naming

#### Version 0.5.1
- Added project TOML for publication
- Various bug fixes

#### Version 0.4.1
- Use dictionary to internally store richer channel information
- Ome metadata fixes

#### Version 0.4.0
- Improved support for ome.zarr / NGFF
- (ome.)zarr lossless/lossy image compression
- Independent import/export improving import/export image support
- Use pixel size instead of magnification for scaling
- Use optimal Omero pixel source pyramid resolution
- Many metadata fixes
- Removed channel split support as no use cases
- Label extraction no longer in image pipeline, use separate script instead

#### Version 0.3.3
- Omero class OmeroSource now in line with OmeSource functionality
- OME metadata now fully implemented using xmltodict
- Various bug fixes to image/OME metadata

#### Version 0.3.2
- Numerous fixes to OME metadata
- Support volumetric images

#### Version 0.3.1
- Refactored Source classes
- Added bare bones Ome Zarr support
