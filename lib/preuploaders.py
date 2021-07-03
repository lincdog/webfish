import gc
import numpy as np
import skimage.util as skiu
import skimage.exposure as skie
from pathlib import Path
from lib.util import pil_imread, safe_imwrite

"""
preuploaders.py
-------------
This file contains classes and functions referred to as "preuploads":
they take raw input files that are located on the master storage (HPC),
and perform processing on them before they are uploaded to S3. This function
is analogous to `generators.py` but happens on the master storage computer,
*before* the files are uploaded to S3. 
When the cron job on the HPC
finds new files that haven't been uploaded to S3 yet, it first calls 
`server.DataServer.run_preuploads` on them and then only uploads the successfully
processed results of that. 

The preuploader's job is to do some processing and return the path to the
processed file. It is expected to check if the file exists already - this is not enforced
by DataServer. Because this is happening on the HPC, the output file location is specified
by the config entry `preupload_root`, currently in my (Lincoln) personal folder. DataServer
updates the filenames as the preupload jobs succeed, to point to this new location.
"""


def compress_raw_im(*args):
    return _compress_raw_im(
        *args,
        rescaler=None
    )


def compress_raw_im_2(*args):
    return _compress_raw_im(
        *args,
        rescaler=rescale_using_percentile
    )


def _compress_raw_im(
    inrow,
    outpattern,
    savedir,
    rescaler=None
):
    gc.enable()
    if not inrow:
        return None

    im = inrow['filename']
    outfile = Path(savedir, outpattern.format_map(inrow))

    if outfile.is_file():
        return outfile.relative_to(savedir), False

    try:
        compress_8bit(im, 'DEFLATE', outfile, rescaler=rescaler)
        gc.collect()
    except Exception as e:
        return im, e

    return Path(outfile).relative_to(savedir), False


def compress_8bit(
    imgfilename,
    compression='DEFLATE',
    outfile=None,
    rescaler=None
):
    """
    compress_8bit
    -------------
    Opens an image file and saves it to outfile after optional
    rescaling, conversion to 8 bits, and compression.

    rescaler should be a function that takes the numpy array of the image
    and returns a numpy array of the same shape.
    """
    err = None
    im = None

    if not callable(rescaler):
        def rescaler(arr):
            return arr

    try:
        im = pil_imread(imgfilename)

        safe_imwrite(
            skiu.img_as_ubyte(rescaler(im)),
            outfile,
            compression=compression
        )
    except (PermissionError, IOError, OSError) as e:
        err = e
    finally:
        del im
        gc.collect()

    if err:
        raise err

    return outfile


def rescale_using_percentile(
    imarr,
    q=99.999,
    in_start=0.0,
    out_range='dtype'
):
    """
    rescale_using_percentile
    -----------------------
    Rescales an image linearly to the (by default) full range of its
    dtype by using a reasonable guess as to what the "relevant" pixel values
    in it are to start. That is, instead of rescaling its min and max to
    the dtype range, we take a high percentile of the pixel values -
    99.999 seems to work decently - as the maximum and rescale that to the
    dtype maximum.

    Performs this separately if the image array is 4D, assuming the first
    axis is channels.
    """
    if imarr.ndim == 4:
        # Loop through each channel **assuming first axis**
        # Find quantile in each one and rescale separately

        in_stops = [np.percentile(imc, q)
                    for imc in imarr]

        return np.array([
            skie.rescale_intensity(
                imc,
                in_range=(in_start, in_stop),
                out_range=out_range
            ) for imc, in_stop in zip(imarr, in_stops)
        ], dtype=imarr.dtype)
    else:
        # Single channel image case
        in_stop = np.percentile(imarr, q)

        return skie.rescale_intensity(
            imarr,
            in_range=(in_start, in_stop),
            out_range=out_range
        )
