import tifffile as tif
import skimage.measure as skim
import skimage.transform as skit
import numpy as np
import pandas as pd
import json
import re
import os
import string
from pathlib import Path, PurePath
import base64
from matplotlib.pyplot import get_cmap


def gen_mesh(
        imgfilename,
        px_size=(0.5, 0.11, 0.11),
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

    print(f'entering gen_mesh with infile {imgfilename} and outfile {outfile}')

    im = tif.imread(imgfilename)

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
                fill_value=None
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

    print('leaving gen_mesh')

    return mesh


def gen_pcd_df(
        csv,
        px_size=(0.5, 0.11, 0.11),
        cmap='tab20',
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

    print('entering gen_pcd_df')

    if isinstance(csv, str):
        dots = pd.read_csv(csv)
    elif isinstance(csv, pd.DataFrame):
        dots = csv.copy()
    else:
        raise TypeError

    dots['geneInd'] = dots['gene'].factorize()[0] % 20

    def cmap2hex(cmap):
        return '#{:02X}{:02X}{:02X}'.format(cmap[0], cmap[1], cmap[2])

    cm = get_cmap(cmap).colors
    cm = (255 * np.array(cm)).astype(np.uint8)  # convert to bytes

    dots['geneColor'] = [cmap2hex(cm[c]) for c in dots['geneInd']]

    # adjust Z to start from 0
    dots[['z', 'y', 'x']] -= np.array([1, 0, 0])
    # convert to real units
    dots[['z', 'y', 'x']] *= np.array(px_size)

    if outfile is not None:
        dots.to_csv(outfile, index=False)

    return dots


##### Helper functions ######

def mesh_from_json(jsonfile):
    """
    mesh_from_json:
    take a json filename, read it in,
    and convert the verts and faces keys into numpy arrays.
    
    returns: dict of numpy arrays
    """
    if isinstance(jsonfile, str):
        cell_mesh = json.load(open(jsonfile))
    elif isinstance(jsonfile, PurePath):
        cell_mesh = json.load(open(str(jsonfile)))
    elif isinstance(jsonfile, dict):
        cell_mesh = jsonfile
    else:
        raise TypeError('mesh_from_json requires a string, Path, or dict.')

    assert 'verts' in cell_mesh.keys(), f'Key "verts" not found in file {jsonfile}'
    assert 'faces' in cell_mesh.keys(), f'Key "faces" not found in file {jsonfile}'

    cell_mesh['verts'] = np.array(cell_mesh['verts'])
    cell_mesh['faces'] = np.array(cell_mesh['faces'])

    return cell_mesh


def populate_mesh(cell_mesh):
    """
    populate_mesh:
    take a mesh dictionary (like returned from `mesh_from_json`) and return the
    six components used to specify a plotly.graph_objects.Mesh3D
    
    returns: 6-tuple of numpy arrays: x, y, z are vertex coords; 
    i, j, k are vertex indices that form triangles in the mesh.
    """

    if cell_mesh is None:
        return None, None, None, None, None, None

    z, x, y = cell_mesh['verts'].T
    i, j, k = cell_mesh['faces'].T

    return x, y, z, i, j, k


def populate_genes(dots_pcd):
    """
    populate_genes:
    takes a dots dataframe and computes the unique genes present,
    sorting by most frequent to least frequent.
    
    returns: list of genes (+ None and All options) sorted by frequency descending
    """
    unique_genes, gene_counts = np.unique(dots_pcd['gene'], return_counts=True)

    possible_genes = ['All'] + list(np.flip(unique_genes[np.argsort(gene_counts)]))

    return possible_genes


def populate_files(
        directory,
        dirs_only=True,
        prefix='MMStack_Pos',
        postfix='',
        converter=int
):
    """
    populate_files
    ------------------
    Takes either a *list* of files/folders OR a directory name 
    and searches in it for entries that match `regex` of the form 
    <Prefix><Number><Postfix>,capturing the number.
    
    Also takes `converter`, a function to convert the number from a string
    to a number. default is int(). If this fails it is kept as a string.
    
    Returns: List of tuples of the form (name, number), sorted by
    number.
    """
    regex = re.escape(prefix) + '(\d+)' + re.escape(postfix)
    pos_re = re.compile(regex)

    result = []

    def extract_match(name, regex=pos_re, converter=converter):
        m = regex.search(name)
        if m is not None:
            try:
                ret = m.group(0), converter(m.group(1))
            except ValueError:
                ret = m.group(0), m.group(1)

            return ret
        else:
            return None

    if isinstance(directory, list):
        dirs = directory
    else:
        if dirs_only:
            dirs = [entry.name for entry in os.scandir(directory)
                    if entry.is_dir()]
        else:
            dirs = [entry.name for entry in os.scandir(directory)]

    for d in dirs:
        m = extract_match(d)
        if m is not None:
            result.append(m)

    # sort by the number
    return sorted(result, key=lambda n: n[1])


def base64_image(filename, with_header=True):
    if filename is not None:
        data = base64.b64encode(open(filename, 'rb').read()).decode()
    else:
        data = ''

    if with_header:
        prefix = 'data:image/png;base64,'
    else:
        prefix = ''

    return prefix + data


def fmt2regex(fmt, delim=os.path.sep):
    """
    fmt2regex:
    convert a curly-brace format string with named fields
    into a regex that captures those fields as named groups,

    Returns:
    * reg: compiled regular expression to capture format fields as named groups
    * globstr: equivalent glob string (with * wildcards for each field) that can
        be used to find potential files that will be analyzed with reg.
    """
    sf = string.Formatter()

    regex = []
    globstr = []
    keys = set()

    numkey = 0

    if delim:
        parts = fmt.split(delim)
    else:
        delim = ''
        parts = [fmt]

    re_delim = re.escape(delim)

    for part in parts:
        for a in sf.parse(part):
            r = re.escape(a[0])

            newglob = a[0]
            if a[1]:
                newglob = newglob + '*'
            globstr.append(newglob)

            if a[1] is not None:
                k = re.escape(a[1])

                if len(k) == 0:
                    k = f'k{numkey}'
                    numkey += 1

                if k in keys:
                    r = r + f'(?P={k})'
                else:
                    r = r + f'(?P<{k}>[^{re_delim}]+)'

                keys.add(k)

            regex.append(r)

    reg = re.compile('^'+re_delim.join(regex))
    globstr = delim.join(globstr)

    return reg, globstr


def find_matching_files(base, fmt, paths=None):
    """
    findAllMatchingFiles: Starting within a base directory,
    find all files that match format `fmt` with named fields.

    Returns:
    * files: list of filenames, including `base`, that match fmt
    * keys: Dict of lists, where the keys are each named key from fmt,
        and the lists contain the value for each field of each file in `files`,
        in the same order as `files`.
    """

    reg, globstr = fmt2regex(fmt)

    base = PurePath(base)

    files = []
    keys = {}

    if paths is None:
        paths = Path(base).glob(globstr)
    else:
        paths = [Path(p) for p in paths]

    for f in paths:
        m = reg.match(str(f.relative_to(base)))

        if m:
            files.append(f)

            for k, v in m.groupdict().items():
                if k not in keys.keys():
                    keys[k] = []

                keys[k].append(v)

    return files, keys


def fmts2file(*fmts, fields={}):
    fullpath = str(Path(*fmts))
    return Path(fullpath.format(**fields))


def k2f(
    k,
    delimiter='/'
):
    return PurePath(str(k).replace(delimiter, os.sep))


def f2k(
    f,
    delimiter='/'
):
    return str(f).replace(os.sep, delimiter)


def ls_recursive(root='.', level=1, ignore=[], dirsonly=True, flat=False):
    if flat:
        result = []
    else:
        result = {}

    if not isinstance(level, int):
        raise ValueError('level must be an integer')

    def _ls_recursive(
        contents=None,
        folder='.',
        root='',
        maxlevel=level,
        curlevel=0,
        dirsonly=dirsonly,
        flat=flat
    ):
        if curlevel == maxlevel:
            if flat:
                contents.extend([f.relative_to(root) for f in Path(folder).iterdir()
                        if f.is_dir() or not dirsonly])
                return contents
            else:
                return [f.name for f in Path(folder).iterdir()
                        if f.is_dir() or not dirsonly]

        args = dict(
            contents=contents,
            root=root,
            maxlevel=level,
            curlevel=curlevel+1,
            dirsonly=dirsonly,
            flat=flat
        )

        subfolders =[f for f in Path(folder).iterdir() if (
            f.is_dir() and not any([f.match(p) for p in ignore]))]

        if flat:
            [_ls_recursive(folder=f, **args) for f in subfolders]
        else:
            contents = {f.name: _ls_recursive(folder=f, **args) for f in subfolders}

        return contents

    result = _ls_recursive(
        result,
        folder=root,
        root=root,
        maxlevel=level,
        curlevel=0,
        dirsonly=dirsonly,
        flat=flat
    )
    return result
