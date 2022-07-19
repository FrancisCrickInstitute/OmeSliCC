# OmeSliCC
## Ome(ro) Slide Image Conversion and Compression pipeline

OmeSliCC is designed to convert slides from common formats, to optimal OME formats for deep learning.

This includes converting from Omero and extracting metadata as label information.

For support and discussion, please use the [Image.sc forum](https://forum.image.sc) and post to the forum with the tag 'OmeSliCC'.

## Main features

- Import WSI files: Omero, Ome.tiff, Tiff, basic image formats, Zarr*
- Export images: Ome.tiff, Zarr*, thumbnails
- Export meta-data: Ome.tiff, csv
- Omero credentials helper
- \*Zarr currently partially implemented

## Running OmeSliCC

OmeSliCC is 100% Python and can be run as follows:
- On a local environment using requirements.txt
- With conda environment using the conda yaml file
- As Docker container

## Acknowledgements

The [Open Microscopy Environment (OME)](https://www.openmicroscopy.org/) project

The Francis Crick Institute
- The [Software Engineering and Artificial Intelligence](https://intranet.crick.ac.uk/our-crick/software-engineering-and-ai-stp) team
- The [Turajlic](https://www.crick.ac.uk/research/labs/samra-turajlic) lab