import tifffile as tif
import numpy as np
import pandas as pd
import json
import re
import os
import string
import numbers
from pathlib import Path, PurePath
import base64


def safe_imread(fname, is_ome=False, is_imagej=True):
    imarr = np.array([])

    try:
        imarr = tif.imread(fname, is_ome=is_ome, is_imagej=is_imagej)
    except (AttributeError, RuntimeError):
        imarr = tif.imread(fname, is_ome=is_ome, is_imagej=False)

    return imarr


def safe_imwrite(
    arr,
    fname,
    compression='DEFLATE',
    ome=False,
    imagej=True
):
    arr = arr.copy()
    try:
        with tif.TiffWriter(fname, ome=ome, imagej=imagej) as tw:
            tw.write(arr, compression=compression)
    except RuntimeError:
        with tif.TiffWriter(fname, ome=ome, imagej=False) as tw:
            tw.write(arr, compression=compression)

    del arr


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
        slices_first=True,
        max_channels=6,
        is_ome=False,
        is_imagej=True
    ):

        if isinstance(tifffile, type(super())):
            self = tifffile
        else:
            try:
                super().__init__(
                    tifffile,
                    is_ome=is_ome,
                    is_imagej=is_imagej
                )
            except (AttributeError, RuntimeError):
                super().__init__(
                    tifffile,
                    is_ome=False,
                    is_imagej=False
                )

        self.shape = None
        self.channels = 1
        self.slices = 1

        #self.is_ome = is_ome
        #self.is_imagej = is_imagej

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

    z, x, y = np.array(cell_mesh['verts']).T
    i, j, k = np.array(cell_mesh['faces']).T

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


def sanitize(
    k,
    delimiter='/',
    delimiter_allowed=True,
    raiseonfailure=False
):
    badchars = '\\[]{}^%#` <>~|'

    if raiseonfailure:
        err = delimiter in badchars or (delimiter in k and not delimiter_allowed)
        if err:
            raise ValueError(f'Delimiter:  {delimiter}  is not allowed.')

    if delimiter and delimiter_allowed:
        parts = k.split(delimiter)
    else:
        parts = [k]

    # put it in the middle so there's less chance of it messing up
    # the exp
    # Note: I added unix shell special characters like $, &, ;, *, ! because
    # the analysis names will be used as folder names
    exp = '[\\\\{^}%$&*@!/?;` ' + re.escape(delimiter) + '\\[\\]>~<#|]'
    parts_sanitized = [re.sub(exp, '', part) for part in parts]

    if any([len(part) == 0 for part in parts_sanitized]):
        raise ValueError(f'After sanitizing, a part of the string "{k}"'
                         f' disappeared completely.')

    return delimiter.join(parts_sanitized)


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


def process_file_locations(locs):
    result = {}

    for key, value in locs.items():
        info = value

        dformat = info.get('dataset_format', '')

        dfr, dfg = fmt2regex(dformat)
        fields = list(dfr.groupindex.keys())

        info['dataset_format_re'] = dfr
        info['dataset_format_glob'] = dfg
        info['dataset_format_fields'] = fields
        info['dataset_format_nest'] = len(Path(dformat).parts) - 1

        result[key] = info

    return result


def empty_or_false(thing):
    if isinstance(thing, pd.DataFrame):
        return thing.empty

    return not thing


def notempty(dfs):
    return [not empty_or_false(df) for df in dfs]


def copy_or_nop(df):
    try:
        result = df.copy()
    except AttributeError:
        result = df

    return result
