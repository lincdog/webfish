import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import json
import re
import base64
import logging

from copy import deepcopy
from pathlib import PurePath, Path

from lib.util import (
    pil_imread,
)


class PageHelper:
    
    def __init__(
        self,
        data_client,
        component_manager,
        default_graph=None,
        logger=None
    ):
        
        self.logger = logger or logging.getLogger('webfish.' + __name__)
        self.data_client = data_client
        self.cm = component_manager
        
        if default_graph:
            try:
                self._dash_graph = self.cm.component(default_graph)
            except KeyError:
                self._dash_graph = dcc.Graph(id=default_graph)
        else:
            self._dash_graph = dcc.Graph(id='graph')

    def dash_graph(self, **kwargs):
        new_graph = deepcopy(self._dash_graph)

        for k, v in kwargs.items():
            setattr(new_graph, k, v)

        return new_graph
    
    
class DotDetectionHelper(PageHelper):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gen_image_figure(
        self,
        imfile,
        dots_csv=None,
        offsets=(0, 0),
        hyb='0',
        z_slice='0',
        channel='0',
        contrast_minmax=(0, 2000),
        strictness=None
    ):
        self.logger.info('Entering gen_image_figure')
    
        if len(imfile) > 0:
            image = pil_imread(imfile[0])
        else:
            return {}
    
        self.logger.info(f'gen_image_figure: Read image from {imfile[0]}')
    
        print(f'hyb {hyb} z_slice {z_slice} channel {channel}')
        print(image.shape)
    
        hyb = int(hyb)
        hyb_q = hyb
        # 'z' column in locations.csv starts at 1
        z_slice = int(z_slice)
        z_slice_q = z_slice + 1
        # 'ch' column in locations.csv starts at 1
        channel = int(channel)
        channel_q = channel + 1
    
        if z_slice >= 0:
            img_select = image[channel, z_slice]
    
            dots_query = 'hyb == @hyb_q and ch == @channel_q and z == @z_slice_q'
        else:
            img_select = np.max(image[channel], axis=0)
            dots_query = 'hyb == @hyb_q and ch == @channel_q'
    
        fig = px.imshow(
            img_select,
            zmin=contrast_minmax[0],
            zmax=contrast_minmax[1],
            width=1000,
            height=1000,
            binary_string=True,
            binary_compression_level=4,
            binary_backend='pil'
        )
    
        fig.data[0].customdata = (img_select/2.55).astype(np.uint8)
        fig.data[0].hovertemplate = '(%{x}, %{y})<br>%{customdata}'
    
        self.logger.info('gen_image_figure: constructed Image figure')
        self.logger.info('gen_image_figure: length of data source: %d',
                    len(fig.data[0].source))
        self.logger.info('gen_image_figure: total length of JSON serialized figure is: %d',
                    len(fig.to_json()))
    
        if dots_csv:
            dots_select = pd.read_csv(dots_csv[0])
    
            if strictness and 'strictness' in dots_select.columns:
                dots_query += ' and strictness >= @strictness'

            minz, maxz = dots_select['z'].min(), dots_select['z'].max()

            if minz == 0:
                z_slice_q -= 1
    
            dots_select = dots_select.query(dots_query)
    
            self.logger.info(f'gen_image_figure: read and queried dots CSV file '
                        f'{dots_csv[0]}')
    
            if 'strictness' in dots_select.columns:
                strictnesses = dots_select['strictness'].values
    
                color_by = np.nan_to_num(strictnesses, nan=0)
                cbar_title = 'Strictness'
            else:
                color_by = dots_select['z'].values
                cbar_title = 'Z slice'
    
            if 'int' in dots_select.columns:
                intensities = dots_select['int'].values
                hovertext = ['{0}: {1:.0f} <br>Intensity: {2:.0f}'.format(
                    cbar_title, cb, i) for cb, i in zip(color_by, intensities)]
            else:
                hovertext = ['{0}: {1:.0f}'.format(cbar_title, cb) for cb in color_by]
    
            if len(set(color_by)) > 1:
                cmin, cmax = min(color_by), max(color_by)
            elif len(set(color_by)) == 1:
                cmin, cmax = color_by[0], color_by[0]
            else:
                cmin, cmax = 0, 0
    
            fig.add_trace(go.Scattergl(
                name='detected dots',
                x=dots_select['x'].values - offsets[1],
                y=dots_select['y'].values - offsets[0],
                mode='markers',
                marker_symbol='cross',
                text=color_by,
                hovertemplate='(%{x}, %{y})<br>' + cbar_title + ': %{text}',
                marker=dict(
                    #maxdisplayed=1000,
                    size=5,
                    cmax=cmin,
                    cmin=cmax,
                    colorbar=dict(
                        title=cbar_title
                    ),
                    colorscale='YlOrRd_r',
                    color=color_by))
            )
    
            fig.update_layout(coloraxis_showscale=True)
    
            self.logger.info('gen_image_figure: constructed and added dots Scatter trace')
            self.logger.info('gen_image_figure: total length of JSON serialized figure is: %d',
                        len(fig.to_json()))
    
        return fig
    
    def prepare_dotdetection_figure(
        self,
        z,
        channel,
        contrast,
        strictness,
        position,
        hyb,
        analysis,
        dataset,
        user,
        current_layout
    ):
        if any([v is None for v in (z, channel, contrast)]):
            return self.dash_graph()
    
        self.logger.info('prepare_dotdetection_figure: requesting raw image filename')
    
        hyb_fov = self.data_client.request(
            {'user': user, 'dataset': dataset, 'position': position, 'hyb': hyb},
            fields='hyb_fov'
        )['hyb_fov']
    
        self.logger.info('prepare_dotdetection_figure: got raw image filename')
    
        if analysis:
            self.logger.info('prepare_dotdetection_figure: requesting dot locations and offsets')
    
            requests = self.data_client.request(
                {'user': user, 'dataset': dataset, 'position': position, 'analysis': analysis},
                fields=['dot_locations', 'offsets_json']
            )
            dot_locations = requests['dot_locations']
            offsets_json = requests['offsets_json']
    
            self.logger.info('prepare_dotdetection_figure: got dot locations and offsets')
    
        else:
            dot_locations = None
            offsets_json = None
    
        if offsets_json:
            all_offsets = json.load(open(offsets_json[0]))
            offsets = all_offsets.get(
                f'HybCycle_{hyb}/MMStack_Pos{position}.ome.tif',
                (0, 0)
            )
        else:
            offsets = (0, 0)
    
        self.logger.info('prepare_dotdetection_figure: calling gen_image_figure')
    
        figure = self.gen_image_figure(
            hyb_fov,
            dot_locations,
            offsets,
            hyb,
            z,
            channel,
            contrast,
            strictness
        )
    
        if current_layout:
            if 'xaxis.range[0]' in current_layout:
                figure['layout']['xaxis']['range'] = [
                    current_layout['xaxis.range[0]'],
                    current_layout['xaxis.range[1]']
                ]
            if 'yaxis.range[0]' in current_layout:
                figure['layout']['yaxis']['range'] = [
                    current_layout['yaxis.range[0]'],
                    current_layout['yaxis.range[1]']
                ]
    
        self.logger.info('prepare_dotdetection_figure: returning updated figure')

        return self.dash_graph(figure=figure, relayoutData=current_layout)
    
    def prepare_preprocess_figure(
        self,
        position,
        hyb,
        channel,
        analysis,
        dataset,
        user
    ):
        self.logger.info('entering prepare_preprocess_figure')
    
        pp_im = self.data_client.request({
            'user': user,
            'dataset': dataset,
            'analysis': analysis,
            'position': position,
            'hyb': hyb,
            'channel': channel
        }, fields='preprocess_check')['preprocess_check']
    
        self.logger.info('prepare_preprocess_figure: got preprocess check file')
    
        fig = go.Figure()
        fig.update_layout(width=1000, height=1000)

        if pp_im:
            fig.add_image(source=base64_image(pp_im[0]))
            return self.dash_graph(figure=fig)
        else:
            alert = dbc.Alert('No preprocessing check image found', color='warning')
            return [alert, self.dash_graph(figure=fig)]

    def prepare_locations_figure(
        self,
        position,
        analysis,
        dataset,
        user
    ):
        self.logger.info('entering prepare_locations_figure')
    
        loc_ims = self.data_client.request({
            'user': user,
            'dataset': dataset,
            'analysis': analysis,
            'position': position
        }, fields=('location_check_xy', 'location_check_z'))
    
        self.logger.info('prepare_locations_figure: got check filenames')
    
        results = []
    
        if loc_ims['location_check_xy']:
            results.append(
                html.Img(src=base64_image(loc_ims['location_check_xy'][0]))
            )
    
        if loc_ims['location_check_z']:
            results.append(
                html.Img(src=base64_image(loc_ims['location_check_z'][0]))
            )
    
        if not results:
            results.append(dbc.Alert('No location checks found for this analysis.'
                                     , color='warning'))
    
        results.append(self.dash_graph())
    
        return results

    def prepare_alignment_figure(
        self,
        position,
        analysis,
        dataset,
        user
    ):
        self.logger.info('Entering prepare_alignment_figure')
    
        align_im_file = self.data_client.request({
            'user': user,
            'dataset': dataset,
            'analysis': analysis,
            'position': position
        }, fields='alignment_check')['alignment_check']
    
        self.logger.info('prepare_alignment_figure: got alignment check file')
    
        fig = go.Figure()
        fig.update_layout(width=1000, height=1000)
    
        if align_im_file:
            align_im = pil_imread(align_im_file, False, False)
            self.logger.info(f'prepare_alignment_figure: read alignment image of shape '
                    f'{align_im.shape}')
    
            fig = px.imshow(
                align_im,
                width=1000,
                height=1000,
                animation_frame=0,
                binary_string=True
            )

            return self.dash_graph(figure=fig)
        else:
            alert = dbc.Alert('No alignment check found for this analysis.', color='warning')
            return [alert, self.dash_graph()]


class DatavisHelper(PageHelper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def query_df(df, selected_genes):
        """
        query_df:

        returns: Filtered DataFrame
        """
        if 'All' in selected_genes:
            return df
        elif 'All Real' in selected_genes:
            selected_genes.extend([gene for gene in df['gene'] if 'fake' not in gene])
        elif 'All Fake' in selected_genes:
            selected_genes.extend([gene for gene in df['gene'] if 'fake' in gene])

        return df.query('gene in @selected_genes')

    def gen_figure_2d(
        self,
        selected_genes,
        active,
        color_option,
        channel
    ):

        dots = active.get('dots')

        self.logger.info('Entering gen_figure_2d')

        fig = go.Figure()

        if 'background_im' in active:
            imfile = active.get('background_im')
            imtype = 'background_im'
        elif 'presegmentation_im' in active:
            imfile = active.get('presegmentation_im')
            imtype = 'presegmentation_im'
        elif 'hyb_fov' in active:
            imfile = active.get('hyb_fov')
            imtype = 'hyb_fov'
        else:
            imfile = None
            imtype = ''

        if imfile:
            img = pil_imread(imfile[0])

            self.logger.info('gen_figure_2d: read in 2d image')

            # TODO: Allow choosing Z slice?
            # FIXME: Choose channel or use channel that was used in decoding
            if img.ndim == 4:
                img = np.max(img[channel], axis=0)
            elif img.ndim == 3:
                img = img[channel]

            fig = px.imshow(
                img,
                zmin=0,
                zmax=200,
                width=1000,
                height=1000,
                binary_string=True
            )

            self.logger.info('gen_figure_2d: created image trace')

        # If dots is populated, grab it.
        # Otherwise, set the coords to None to create an empty Scatter3d.
        if dots is not None:

            dots_df = pd.read_csv(dots)
            dots_filt = self.query_df(dots_df, selected_genes).copy()
            del dots_df

            self.logger.info('gen_figure_2d: read and queried dots DF')

            p_x, py = dots_filt[['x', 'y']].values.T

            color = dots_filt['geneColor']
            if color_option == 'fake':
                real_fake = ('cyan', 'magenta')
                color = [real_fake[int('fake' in g)] for g in dots_filt['gene']]

            hovertext = dots_filt['gene']

            fig.add_trace(
                go.Scattergl(
                    name='dots',
                    x=p_x, y=py,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=color,
                        opacity=1,
                        symbol='cross',
                    ),
                    hoverinfo='x+y+text',
                    hovertext=hovertext
                )
            )

            self.logger.info('gen_figure_2d: created Scattergl trace')

        self.logger.info('gen_figure_2d: returning updated figure')

        return fig

    def gen_figure_3d(
        self,
        selected_genes,
        active,
        color_option,
        z_step_size,
        pixel_size
    ):
        """
        gen_figure_3d:
        Given a list of selected genes and a dataset, generates a Plotly figure with
        Scatter3d and Mesh3d traces for dots and cells, respectively. Memoizes using the
        gene selection and active dataset name.

        If ACTIVE_DATA is not set, as it isn't at initialization, this function
        generates a figure with an empty Mesh3d and Scatter3d.

        Returns: plotly.graph_objects.Figure containing the selected data.
        """
        self.logger.info('Entering gen_figure_3d')

        print(active)
        dots = active.get('dots')
        mesh = active.get('mesh')

        figdata = []

        # If dots is populated, grab it.
        # Otherwise, set the coords to None to create an empty Scatter3d.
        if dots is not None:

            dots_df = pd.read_csv(dots)
            dots_filt = self.query_df(dots_df, selected_genes).copy()
            del dots_df

            self.logger.info('gen_figure_3d: read and queried dots DF')

            pz, p_x, py = dots_filt[['z', 'y', 'x']].values.T

            color = dots_filt['geneColor']
            if color_option == 'fake':
                real_fake = ('#1d4', '#22a')
                color = [real_fake[int('fake' in g)] for g in dots_filt['gene']]

            hovertext = dots_filt['gene']

            figdata.append(
                go.Scatter3d(
                    name='dots',
                    x=p_x, y=py, z=pz,
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=color,
                        opacity=1,
                        symbol='circle',
                    ),
                    hoverinfo='text',
                    hovertext=hovertext
                )
            )

            self.logger.info('gen_figure_3d: added Scatter3d trace')

        # A sensible default for aesthetic purposes (refers to the ratio between the
        # total extent in the Z dimension to that in the X or Y direction
        z_aspect = 0.07
        # If the mesh is present, populate it.
        # Else, create an empty Mesh3d.
        if mesh is not None:

            x, y, z, i, j, k = populate_mesh(mesh_from_json(mesh))

            figdata.append(
                go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    color='lightgray',
                    opacity=0.6,
                    hoverinfo='skip',
                )
            )

            self.logger.info('gen_figure_3d: Added mesh3d trace')

            if pixel_size and z_step_size:
                x_extent = pixel_size * (x.max() - x.min())
                z_extent = z_step_size * (z.max() - z.min())

                z_aspect = z_extent / x_extent

                self.logger.info(f'gen_figure_3d: px {pixel_size} z {z_step_size} '
                            f'gives x extent {x_extent} z extent {z_extent} '
                            f'ratio = {z_aspect}')

        figscene = go.layout.Scene(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=z_aspect),
        )

        figlayout = go.Layout(
            height=1000,
            width=1000,
            margin=dict(b=10, l=10, r=10, t=10),
            scene=figscene
        )

        fig = go.Figure(data=figdata, layout=figlayout)

        self.logger.info('gen_figure_3d: returning figure')

        return fig

    def gen_figure_allpos(self, active):
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=3, cols=2)

        for i, (k, v) in enumerate(active.items()):
            row = 1 + (i // 2)
            col = 1 + (i % 2)

            print(row, col)

            if v:
                fig.add_image(
                    source=base64_image(v[0]),
                    row=row,
                    col=col,

                )

        return fig


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

    z, y, x = np.array(cell_mesh['verts']).T
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

    possible_genes = ['All', 'All Real', 'All Fake'] +\
        list(np.flip(unique_genes[np.argsort(gene_counts)]))

    return possible_genes


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


def aggregate_dot_dfs(locations_csvs, hyb, position, take_column='int'):

    position_temp = []
    rename = {take_column: 'Dot Count', 'ch': 'Channel'}

    if position is None:
        groupby = ['ch']
        columns = ['Position', 'Channel', 'Dot Count']

    else:
        groupby = ['ch', 'z']
        rename['z'] = 'Z slice'
        columns = ['Position', 'Channel', 'Z slice', 'Dot Count']

    result = pd.DataFrame(columns=columns)

    for csvname in locations_csvs:
        m = re.search('MMStack_Pos(\\d+)', str(csvname))

        if len(m.groups()) > 0:
            curpos = int(m.groups()[0])
        else:
            continue

        hyb_q = int(hyb)

        dots_df = pd.read_csv(csvname).query('hyb == @hyb_q')

        dots_count = dots_df.groupby(groupby)[take_column].count().reset_index()
        dots_count.rename(columns=rename, inplace=True)

        dots_count['Channel'] = dots_count['Channel'] - 1

        if 'Z slice' in dots_count.columns:
            if dots_count['Z slice'].min() == 1:
                dots_count['Z slice'] -= 1

        dots_count['Position'] = curpos

        position_temp.append(dots_count)

        del dots_df

    if position_temp:
        result = pd.concat(position_temp)[columns].sort_values(by=columns[:-1])

    return result


