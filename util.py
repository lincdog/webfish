import tifffile as tif
import skimage.measure as skim
import skimage.feature as skif
import skimage.transform as skit
import numpy as np
import pandas as pd
import yaml
import json
from matplotlib.pyplot import get_cmap

config_file = 'consts.yml'
config = yaml.load(open(config_file), Loader=yaml.Loader)

def gen_mesh(
    imgfilename,
    px_size=(0.5, 0.11, 0.11),
    scale_factor=(1., 1./16, 1./16),
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
    
    px_scaled = tuple(a/b for a, b in zip(px_size, scale_factor))
    
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
        corner=(0,0,0)
    ):
        tris = skim.marching_cubes(
            np.pad(binim, ((1,1),(1,1),(1,1))),
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
            comb_tris.extend(rtris[1]+maxpt)
            comb_data.extend(rtris[2])
            maxpt += len(rtris[0])     
    else:
        # make the whole image binary
        comb_pts, comb_tris, _ = triangulate_bin(im_small>0, px_scaled)
    
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
    csvfilename,
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
    
    dots = pd.read_csv(csvfilename)
    
    dots['geneInd'] = dots['geneID'].factorize()[0] % 20
    
    def cmap2hex(cmap):
        return '#{:02X}{:02X}{:02X}'.format(cmap[0], cmap[1], cmap[2])
    
    cm = get_cmap(cmap).colors
    cm = (255*np.array(cm)).astype(np.uint8) # convert to bytes
    
    dots['geneColor'] = [ cmap2hex(cm[c]) for c in dots['geneInd'] ]
    
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
    elif isinstance(jsonfile, dict):
        cell_mesh = jsonfile
    
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

    z,y,x = cell_mesh['verts'].T
    i,j,k = cell_mesh['faces'].T
    
    return x, y, z, i, j, k

def populate_genes(dots_pcd):
    """
    populate_genes:
    takes a dots dataframe and computes the unique genes present,
    sorting by most frequent to least frequent.
    
    returns: list of genes (+ None and All options) sorted by frequency descending
    """
    unique_genes, gene_counts = np.unique(dots_pcd['geneID'], return_counts=True)

    possible_genes = ['None', 'All'] + list(np.flip(unique_genes[np.argsort(gene_counts)]))
        
    return possible_genes


#def populate_positions()