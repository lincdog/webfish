import numpy as np
import pandas as pd
import skimage.transform as skit
import skimage.measure as skim
import json
from matplotlib.pyplot import get_cmap
from pathlib import Path
from lib.util import safe_imread
from tifffile import imread  # for segmentation images

"""
generators.py
-------------
This file contains classes and functions referred to as "generators":
they take raw input files that are downloaded from S3 to the local
computer running Webfish, and perform processing on them to **generate**
output files that are used by the webapp.

The classes are only namespaces with class/static methods defined; we could
equally use namedtuples or similar. The `output_files` section of
the Webfish config file for each page can specify a generator class
and a function to call for each output file. When an output file is
`client.DataClient.request`ed, the `requires` field of that output file
is used to find input files, and the `generator` field is used to find the 
required function from this file like: 
`gen_class = getattr(generators, page['generator_class']`)
`gen_func = getattr(gen_class, page['output_files']['keyname']['generator'])`.

The gen_func is then called with the input file paths and parsed fields as 
arguments as well as the output pattern that tells it how to save the output.

The generator's job is to do some processing and return the path to the
output file. It is expected to check if the file exists already - this is not enforced
by DataClient. And it should handle errors such as missing input files gracefully,
returning some False-y value that the webapp can check and display a proper notice with.
"""


"""
DatavisProcessing
-----------------
Namespace class used to get generating functions for datasets

FUNCTION TEMPLATE:
 - inrows: Dataframe where each row is a file, includes ALL fields from the file
    discovery methods - from dataset_root and whatever source_patterns are required
 - outpattern: pattern from the config file to specify the output filename - this is
    the dataset_root pattern joined to the output_pattern for this output file.
 - savedir: local folder to store output

returns: Filename(s)
"""


def generate_mesh(
    inrows,
    outpattern,
    savedir
):
    """
    generate_mesh
    ------------
    """

    if inrows.empty:
        return None

    outfile = Path(savedir, str(outpattern).format_map(inrows.iloc[0].to_dict()))
    print(outfile)

    if outfile.is_file():
        return outfile

    im = inrows.query('source_key == "segmentation"')['local_filename'].values[0]
    # generate the mesh from the image
    gen_mesh(
        im,
        separate_regions=False,
        region_data=None,
        outfile=outfile)

    return outfile


def generate_dots(
    inrows,
    outpattern,
    savedir
):
    if inrows.empty:
        return None

    inrows = inrows.astype(str)
    outfile = Path(savedir, str(outpattern).format_map(inrows.iloc[0].to_dict()))

    # If the processed file already exists just return it
    if outfile.is_file():
        return outfile

    pcds = []

    genecol = 'gene'
    query_parts = []

    if 'dots_csv' in inrows['source_key'].values:
        query_parts.append('source_key == "dots_csv"')
    elif 'dots_csv_unseg' in inrows['source_key'].values:
        query_parts.append('source_key == "dots_csv_unseg"')

    if 'dots_csv_sm' in inrows['source_key'].values:
        query_parts.append('source_key == "dots_csv_sm"')
    elif 'dots_csv_sm_unseg' in inrows['source_key'].values:
        query_parts.append('source_key == "dots_csv_sm_unseg"')

    query = ' or '.join(query_parts)

    infiles = inrows.query(query)[['channel', 'local_filename']]

    for chan, csv in infiles.values:
        pcd_single = pd.read_csv(csv)

        if 'geneID' in pcd_single.columns:
            pcd_single.rename(columns={'geneID': 'gene'}, inplace=True)

        if not chan:
            chan = 'sequential'

        pcd_single['channel'] = chan
        pcds.append(pcd_single)

    pcds_combined = pd.concat(pcds)
    del pcds

    gen_pcd_df(pcds_combined, genecol='gene', outfile=outfile)
    del pcds_combined

    return outfile


def gen_mesh(
    imgfilename,
    px_size=(1, 1, 1),
    scale_factor=(1., 1. / 16, 1. / 16),
    separate_regions=False,
    region_data=None,
    outfile=None
):
    """
    gen_mesh
    ---------
    Takes an image along with pixel size and scale factor, and generates
    a triangular mesh from a scaled-down version of the image.

    If `separate_regions` is True, triangulates each labeled region (identified by
    skimage.measure.regionprops) separately, and optionally associates data
    `region_data` with each region. `region_data` should be an iterable that
    yields a datum for each label in the image.

    Returns: JSON structure as follows:
        {
          "verts": <2D list of vertex coordinates, shape Nx3>,
          "faces": <2D list of vertex indices forming triangles, shape Tx3>,
          "data": <null or list of data for each face, length T>
        }
    """

    # imagej=False because TiffFile throws a TypeError otherwise
    im = imread(imgfilename, is_imagej=False)

    # RGB images seem to be present for some 2D segmentations
    if im.shape[-1] == 3:
        im = im[..., 0]

    # For a 2D image, make it a 3-slice copy of itself
    if im.ndim == 2:
        im = np.array([im, im, im])

    px_scaled = tuple(a / b for a, b in zip(px_size, scale_factor))

    im_small = skit.rescale(
        im,
        scale_factor,
        order=0,
        mode='constant',
        cval=0,
        clip=True,
        preserve_range=True,
        anti_aliasing=False,
        anti_aliasing_sigma=None
    ).astype(np.uint8)

    del im

    def triangulate_bin(
            binim,
            spacing=px_scaled,
            data=None,
            corner=(0, 0, 0)
    ):
        tris = skim.marching_cubes(
            np.pad(binim, ((1, 1), (1, 1), (1, 1))),
            level=0.5,
            spacing=spacing,
            step_size=1
        )

        # get the real coordinates of the top left corner
        corner_real = np.multiply(spacing, corner)
        # Offset all coords to the corner coord
        # then subtract 1 voxel due to previous pad operation
        new_pts = tris[0] + corner_real - np.array(spacing)

        if data is not None:
            tridata = [data] * len(tris[1])
        else:
            tridata = []

        return new_pts, tris[1], tridata

    from itertools import zip_longest

    comb_pts = []
    comb_tris = []
    comb_data = []

    if separate_regions:

        maxpt = 0

        for r, d in zip_longest(
                skim.regionprops(im_small),
                region_data,
                fillvalue=None
        ):
            rtris = triangulate_bin(r.image, px_scaled, r.bbox[:3], d)

            comb_pts.extend(rtris[0])
            comb_tris.extend(rtris[1] + maxpt)
            comb_data.extend(rtris[2])
            maxpt += len(rtris[0])
    else:
        # make the whole image binary
        comb_pts, comb_tris, _ = triangulate_bin(im_small > 0, px_scaled)

    assert len(comb_tris) == len(comb_data) or len(comb_data) == 0, \
        "Something went wrong in the mesh processing..."

    mesh = {
        'verts': comb_pts.tolist(),
        'faces': comb_tris.tolist(),
        'data': comb_data
    }

    if outfile is not None:
        with open(outfile, 'w') as fp:
            json.dump(mesh, fp)

    return mesh


def gen_pcd_df(
        csv,
        px_size=(1, 1, 1),
        cmap='tab20',
        genecol='gene',
        outfile=None
):
    """
    gen_pcd_df
    ----------
    Takes a CSV file of dot locations and genes and converts pixel units to
    real space units as well as adding hex colors to differentiate each gene
    in a plot.

    Returns: Pandas DataFrame
    """

    # FIXME: Setting pixel size to 1,1,1 means that we are ignoring the
    #   Z step size of the data, and plotting all datasets as if they have
    #   the same Z step size. We should at least have a plotting option
    #   to specify the pixel - Z step aspect ratio.

    if isinstance(csv, str):
        dots = pd.read_csv(csv)
    elif isinstance(csv, pd.DataFrame):
        dots = csv.copy()
    else:
        raise TypeError

    dots['geneInd'] = dots[genecol].factorize()[0] % 20

    def cmap2hex(cmap):
        return '#{:02X}{:02X}{:02X}'.format(cmap[0], cmap[1], cmap[2])

    cm = get_cmap(cmap).colors
    cm = (255 * np.array(cm)).astype(np.uint8)  # convert to bytes

    dots['geneColor'] = [cmap2hex(cm[c]) for c in dots['geneInd']]

    # adjust Z to start from 0
    dots[['z', 'y', 'x']] -= np.array([1, 0, 0])
    # convert to real units
    dots[['z', 'y', 'x']] *= np.array(px_size)

    if genecol != 'gene':
        dots = dots.rename(columns={genecol: 'gene'})

    dots['gene'] = dots['gene'].fillna('<unknown>')

    if outfile is not None:
        dots.to_csv(outfile, index=False)

    return dots

