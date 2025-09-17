# OmeSliCC
## Ome(ro) Slide Image Conversion and Compression pipeline

![OmeSliCC logo](docs/images/logo.png)

OmeSliCC is designed to convert slides from common formats, to optimal OME formats for deep learning.

This includes converting from Omero and extracting metadata as label information.

For support and discussion, please use the [Image.sc forum](https://forum.image.sc) and post to the forum with the tag 'OmeSliCC'.

## Installation

There are multiple conda environments.
- `environment.yml` full environment with all dependencies
- `environment_minimal.yml` environment without omero and bioformats dependencies
- `environment_linux.yml` uses a pre-built zeroc-ice package

To install:
```
conda env create -f .\environment.yml
```

## Main features

- Import WSI files: Omero, Ome.Tiff, Tiff, Zarr, Ome.Zarr/NGFF, common slide formats, common image formats
- Export images: Tiff, Ome.Tiff, Zarr, Ome.Zarr, common image formats, thumbnails
- Integrated Dask support
- Zarr image compression (lossless/lossy)
- Image scaling using target pixel size
- Omero credentials helper

For more info on OME/NGFF see [OME NGFF](https://ngff.openmicroscopy.org)

## Running OmeSliCC

OmeSliCC is 100% Python and can be run as follows:
- On a local environment using requirements.txt
- With conda environment using the conda yaml file
- As Docker container

## Quickstart

To start the conversion pipeline:
```
python run.py --params path/to/params.yml
```

See [params.yml](resources/params.yml) for an example parameter file.
The main sections are:
- input: providing either a file/folder path, or Omero URL
- output: specifying the location and desired format of the output
- actions: which actions to perform:
    - info: show input file information
	- thumbnail: extract image thumbnail
	- convert: convert to desired image output
    - combine: combine separate channel images into multi-channel image(s)

To encode credentials for Omero access:
```
python encode_omero_credentials.py --params path/to/params.yml
```

To extract Omero label metadata to text file:
```
python extract_omero_labels.py --params path/to/params.yml
```
## Documentation

See [documentation](https://franciscrickinstitute.github.io/OmeSliCC)

## Changelog

See [ChangeLog](CHANGELOG.md)

## Acknowledgements

The [Open Microscopy Environment (OME)](https://www.openmicroscopy.org/) project

The Francis Crick Institute
- The [Software Engineering and Artificial Intelligence](https://www.crick.ac.uk/research/platforms-and-facilities/software-engineering-and-artificial-intelligence) team
- The [Turajlic](https://www.crick.ac.uk/research/labs/samra-turajlic) lab