import tifffile as tif
import skimage.measure as skim
import skimage.transform as skit
import skimage.util as skiu
import numpy as np
import pandas as pd
import json
import re
import os
import string
import numbers
from pathlib import Path, PurePath
import base64
from matplotlib.pyplot import get_cmap


class ImageMeta(tif.TiffFile):
    """
    ImageMeta:
    class for grabbing important metadata from a micromanager tif file.

    Basic cases:
    1. We know which axis is C, which is Z via the MM metadata, or the series
    axes order

    We then just need to make sure Y and X are the last two axes, then reshape to
    CZYX, inserting length-1 dimensions for C or Z if one is lacking.

    2. We don't know which axis is C and which is Z because the series axes order
    contains alternate letters (I, Q, S, ...)

    We still need to make sure Y and X are the last two axes. But we have to guess
    at the C and Z axes. The policy currently used is:
    * If a non-YX axis is longer than `max_channels` (default 6), it will be assigned to Z no matter what
    * If two non-YX axes are present, the longer one will be assigned to Z, the shorter to C
    * If one non-YX axis is present, it will be assigned to C (unless it is longer than max_channels)

    3. We know the number of channels and slices, but in the series they are combined into
    one axis

    We still make sure Y and X are the last axes. We verify that the length of the third
    axis is equal to channels*slices. By default, channels are assumed to vary slower (i.e. first axis)),
    unless the IJ/MM metadata says otherwise. Reshape using (channels, slices, Y, X).
    """

    def __init__(
        self,
        tifffile,
        pixelsize_yx=None,
        pixelsize_z=None,
        axes=None,
        slices_first=None,
        max_channels=6
    ):

        if isinstance(tifffile, type(super())):
            self = tifffile
        else:
            super().__init__(tifffile)

        self.shape = None
        self.channels = 1
        self.slices = 1

        self.channelnames = None

        self.slices_first = False
        self.indextable = None
        self.rawarray = None
        self.array = None

        # This is a string like 'CZXY'
        self.series_axes = self.series[0].axes
        # This is the actual shape of the raw image that will be loaded in
        self.series_shape = self.series[0].shape
        # The positions of relevant axes in the axes string
        self.series_order = {a: self.series_axes.find(a) for a in ('C', 'Z', 'Y', 'X', 'I', 'Q')}

        if self.series_order['Y'] < 0 or self.series_order['X'] < 0:
            raise np.AxisError(
                f'ImageMeta: TIF axes string was {self.series_axes}, needs to have Y and X')

        # The position of the Y and X axes
        self.yx = (self.series_order['Y'], self.series_order['X'])
        # The position of all other axes
        self.nonyx = tuple(set(i for i in range(len(self.series_axes))) - set(self.yx))

        self.height = self.series_shape[self.series_order['Y']]
        self.width = self.series_shape[self.series_order['X']]

        self.series_dtype = self.series[0].dtype

        try:
            self.ij_metadata = self.imagej_metadata
            self.mm_metadata = self.micromanager_metadata
            self.sh_metadata = self.shaped_metadata
        except AttributeError:
            # Even if the image lacks these, if opened with tifffile they exist as None
            raise ValueError('ImageMeta: supplied image lacks `imagej_metadata`'
                             ' or `micromanager_metadata` attribute. Open with `tifffile`.')

        self.indextable = None

        # MICROMANAGER METADATA
        # Has IndexMap which unambiguously tells us the non-YX axis order
        # We later set "SlicesFirst" using this rather than the metadata
        if self.mm_metadata is not None:

            self.metadata = self.mm_metadata['Summary']

            if 'IndexMap' in self.mm_metadata.keys():
                self.indextable = self.mm_metadata['IndexMap']
            else:
                self.slices_first = self.metadata['SlicesFirst']

        # IMAGEJ METADATA
        # 'Info' is identical (I think) to MM metadata
        # If not present, there is still an outer level with channels, slices count.
        elif self.ij_metadata is not None:

            if 'Info' in self.ij_metadata.keys():
                self.ij_metadata['Info'] = json.loads(self.ij_metadata['Info'])
                self.metadata = self.ij_metadata['Info']
                self.slices_first = self.metadata['SlicesFirst']
            else:
                self.metadata = self.ij_metadata

        # SHAPED METADATA
        # This is added for any file written with tifffile, I think.
        # It minimally just describes the shape of the array at the time of writing.
        elif self.sh_metadata is not None:

            self.metadata = None

            if 'shape' in self.sh_metadata[0].keys():

                self.shape = self.sh_metadata[0]['shape']

                self.height = self.series_shape[self.series_order['Y']] # y extent
                self.width = self.series_shape[self.series_order['X']] # x extent

                # Prepare to guess which is C, which is Z
                shape_temp = list(self.shape).copy()
                shape_temp.remove(self.height)
                shape_temp.remove(self.width)

                # If we can find C, set it. Else, take the minimum non-YX axis.
                if self.series_order['C'] != -1:
                    self.channels = self.series_shape[self.series_order['C']]
                elif len(shape_temp) > 0:
                    self.channels = min(shape_temp)
                    shape_temp.remove(self.channels)
                else:
                    self.channels = 1

                # If we can find Z, set it. Else, take the maximum remaining non-YX axis
                if self.series_order['Z'] != -1:
                    self.slices = self.series_shape[self.series_order['Z']]
                elif len(shape_temp) > 0:
                    self.slices = max(shape_temp)
                    shape_temp.remove(self.slices)
                else:
                    self.slices = 1

        # NO METADATA
        # Without any metadata, we have to guess just like above.
        else:

            self.metadata = None
            self.shape = self.series_shape

            self.height = self.series_shape[self.series_order['Y']]
            self.width = self.series_shape[self.series_order['X']]

            shape_temp = list(self.shape).copy()
            shape_temp.remove(self.height)
            shape_temp.remove(self.width)

            # If we can find C, set it. Else, take the minimum non-YX axis.
            if self.series_order['C'] != -1:
                self.channels = self.series_shape[self.series_order['C']]
            elif len(shape_temp) > 0:
                self.channels = min(shape_temp)
                shape_temp.remove(self.channels)
            else:
                self.channels = 1

            # If we can find Z, set it. Else, take the maximum remaining non-YX axis
            if self.series_order['Z'] != -1:
                self.slices = self.series_shape[self.series_order['Z']]
            elif len(shape_temp) > 0:
                self.slices = max(shape_temp)
                shape_temp.remove(self.slices)
            else:
                self.slices = 1

        # If we were able to find IJ/MM metadata,
        # use it to set all the attributes
        if self.metadata is not None:
            for k, v in self.metadata.items():

                if k.lower() == 'slices':
                    self.slices = v

                if k.lower() == 'channels':
                    self.channels = v

                if k.lower() == 'pixelsize_um' and pixelsize_yx is None:
                    pixelsize_yx = v

                if k.lower() == 'chnames':
                    self.channelnames = v

                if k.lower() == 'height':
                    self.height = v

                if k.lower() == 'width':
                    self.width = v

        if self.indextable is not None:

            try:
                slice1_ind = self.indextable['Slice'].index(1)
            except ValueError:
                slice1_ind = 0

            try:
                chan1_ind = self.indextable['Channel'].index(1)
            except ValueError:
                chan1_ind = 0

            # if slices increase slower than channels
            if slice1_ind > chan1_ind:
                self.slices_first = True

        # Regardless of other things, if we have assigned a channel count
        # higher than max_channels, switch channels and slices.
        if self.channels > max_channels:
            ctmp = self.channels
            self.channels = self.slices
            self.slices = ctmp

        # Set default pixel dimensions if they were not supplied
        # nor set from metadata
        if pixelsize_yx is None:
            pixelsize_yx = (1.0, 1.0)

        if pixelsize_z is None:
            pixelsize_z = 0.5

        # Assemble 3D pixel size
        if isinstance(pixelsize_yx, numbers.Number):
            self.pixelsize = (pixelsize_z, pixelsize_yx, pixelsize_yx)
        elif hasattr(pixelsize_yx, '__iter__'):
            self.pixelsize = (pixelsize_z,) + tuple(i for i in pixelsize_yx)
        else:
            self.pixelsize = (pixelsize_z, 1., 1.)

        if slices_first is not None:
            self.slices_first = slices_first

        self.shape = (self.channels, self.slices, self.height, self.width)

    def validate(
        self,
        channels,
        slices,
        height,
        width,
        shape=None
    ):
        if hasattr(shape, '__iter__'):
            return all([a == b for a, b in zip(self.shape, shape)])

        return self.shape == (channels, slices, height, width)

    def asarray(
        self,
        raw=False,
        **kwargs
    ):
        """
        asarray
        -------
        Get the numpy ndarray of this image, correctly reshaped
        according to the metadata.
        """

        self.rawarray = super().asarray(**kwargs)
        self.array = None

        if raw:
            return self.rawarray

        if self.rawarray.shape != self.shape:

            # first, make sure Y and X are last two axes
            transpose = self.nonyx + self.yx

            self.array = self.rawarray.transpose(transpose)

            if self.rawarray.ndim == 4:

                # If slices vary slower than channels, we actually have to reshape,
                # not just transpose. I think.
                if self.slices_first:
                    self.array = self.array.reshape(self.shape)

                # The only way this could happen is if channels and slices are switched
                if self.array.shape != self.shape:
                    self.array = self.array.transpose((1, 0, 2, 3))

            elif self.rawarray.ndim == 3:
                # find which axis equals channels*slices
                try:
                    axis_to_split = self.rawarray.shape.index(self.channels*self.slices)
                except ValueError:
                    raise np.AxisError(
                        f'3D image must have one axis of size channels*slices = {self.channels*self.slices}')
                # This handles cases where C or Z is 1 too
                self.array = self.array.reshape(self.shape)

            elif self.rawarray.ndim == 2:

                self.array = self.rawarray.reshape((1, 1, self.height, self.width))

        return self.array


def compress_8bit(
    imgfilename,
    compression='DEFLATE',
    outfile=None
):

    im = ImageMeta(imgfilename).asarray()
    with tif.TiffWriter(outfile) as imw:
        imw.write(skiu.img_as_ubyte(im), compression=compression)

    return outfile


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

    print('entering gen_pcd_df')

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

    fmt = str(fmt).rstrip(delim)

    if delim:
        parts = fmt.split(delim)
    else:
        delim = ''
        parts = [fmt]

    re_delim = re.escape(delim)

    for part in parts:
        part_regex = ''
        part_glob = ''

        for a in sf.parse(part):
            r = re.escape(a[0])

            newglob = a[0]
            if a[1]:
                newglob = newglob + '*'
            part_glob += newglob

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

            part_regex += r

        globstr.append(part_glob)
        regex.append(part_regex)

    reg = re.compile('^'+re_delim.join(regex))
    globstr = delim.join(globstr)

    return reg, globstr


def find_matching_files(base, fmt, paths=None, modified_since=0):
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
    mtimes = []
    keys = {}

    if paths is None:
        paths = Path(base).glob(globstr)
    else:
        paths = [Path(p) for p in paths]

    for f in paths:
        m = reg.match(str(f.relative_to(base)))

        if m:
            mtimes.append(os.stat(f).st_mtime)
            files.append(f)

            for k, v in m.groupdict().items():
                if k not in keys.keys():
                    keys[k] = []

                keys[k].append(v)

    return files, keys, mtimes


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


def process_requires(requires):
    reqs = []

    for entry in requires:
        reqs.extend([r.strip() for r in entry.split('|')])

    return reqs


def source_keys_conv(sks):
    # convert a string rep of a list to an actual list
    return sks.split('|')


def process_file_entries(entries):
    result = {}

    for key, value in entries.items():
        info = dict.fromkeys([
            'pattern',
            'requires',
            'generator',
            'preupload'], None)

        if isinstance(value, str):
            info['pattern'] = value
        elif isinstance(value, dict):
            info.update(value)
        else:
            raise TypeError('Each file in config must be either a string or a dict')

        result[key] = info

    return result


def empty_or_false(thing):
    if isinstance(thing, pd.DataFrame):
        return thing.empty

    return not thing


def notempty(dfs):
    return [not empty_or_false(df) for df in dfs]