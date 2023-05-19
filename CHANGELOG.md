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
