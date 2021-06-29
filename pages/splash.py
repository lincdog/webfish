import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd

example_barcode_key = pd.DataFrame({
    'gene': ['acacb', 'acer1', 'acvr2a', 'adcy1', 'adcy10'],
    'hyb1': [5, 3, 3, 7, 6],
    'hyb2': [8, 8, 6, 7, 7],
    'hyb3': [5, 3, 7, 2, 5],
    'hyb4': [2, 6, 8, 8, 2]
})


example_sm_key = pd.DataFrame({
    'gene': ['ADARB2', 'AIF1', 'APOD', 'APOE', 'AQP4'],
    'hyb':  [2, 3, 4, 5, 6],
    'channel': [3, 3, 3, 3, 3]
})

layout = [
    dbc.Col([
        dbc.Jumbotron([
            html.H1('Welcome to Webfish!'),
            dcc.Markdown('Webfish allows Cai Lab researchers to interact with their '
                         'data as well as submit new analysis runs on the Caltech HPC '
                         'analysis pipeline.')
        ], style={'margin-top': '10px'}),
        dbc.Card([
            dbc.CardHeader('Getting Started'),
            dbc.CardBody([
                html.H3('The HPC'),
                dcc.Markdown('The central repository for our data is the '
                             '[Caltech high performance computing cluster](https://www.hpc.caltech.edu/) '
                             '(HPC). You must make an account on the HPC in order to '
                             'upload your data and see it displayed here.'),
                html.Hr(),
                html.H3('Existing Data'),
                dcc.Markdown('If your data of interest is already stored on the HPC, '
                             'use the dropdown menus at the top of the page to '
                             'select a user ("personal" field) and dataset '
                             '("experiment_name" field). Then navigate using the tabs '
                             'to explore the data or submit a new analysis run.')
            ])
        ], style={'margin-top': '10px'}),

    ], width=6),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader('Uploading Data'),
            dbc.CardBody([
                html.H3('Location'),
                dcc.Markdown('To use the pipeline and this site, upload the raw data'
                             ' from each of your experiments to: \n\n**`/groups/CaiLab/personal/'
                             '<user name>/raw/<experiment name>/`**\n\n where you insert '
                             'your name and each experiment\'s name. '
                             '*Use `/central/groups/CaiLab/personal/nrezaee/raw/2020-08-08-takei` as an example.*'),
                dcc.Markdown('*Download a tool like [Cyberduck](https://cyberduck.io/) '
                             'to upload your data using an easy, graphical interface.*'),
                html.H3('Directory Structure'),
                dcc.Markdown('The required directory structure of your dataset is '
                             'essentially the same as the output from an automation '
                             'experiment. The contents of each experiment directory should be:'),
                html.Ul([
                    html.Li(dcc.Markdown('A series of folders, one for each hyb cycle, named **`HybCycle_#`**. '
                                         'Make sure you do not have any other files with `HybCycle_` in the name.')),
                    html.Li(dcc.Markdown(
                        'Within each HybCycle folder, a series of TIFF stacks, '
                        'one for each position, named **`MMStack_Pos#.ome.tif`.**. '
                        'Do not add folders that have `HybCycle_#` in the name unless they have '
                        '`MMStack_Pos#.ome.tif` files in them.')),
                    html.Li([
                        dcc.Markdown(
                            '*If barcoding*: A folder named **`barcode_key`** containing one or more files '
                            'named **`channel_#.csv`** that specify the barcode assigned '
                            'to each gene in each barcoding channel in your experiment.'
                            ' Note that **channels start from 1**. Click the arrow below '
                            'for an example of barcode key formatting, and also see the folder '
                            '`/central/groups/CaiLab/personal/nrezaee/raw/2020-08-08-takei/barcode_key`.'
                        ),
                        html.Div(html.Details([
                            html.Summary(html.B('Barcode key example')),
                            dbc.Table.from_dataframe(example_barcode_key, striped=True, size='sm')
                        ]), style={'width': '50%',
                                   'font-size': '10pt',
                                   'margin': '20px'})
                    ]),
                    html.Li([
                        dcc.Markdown(
                            '*If doing smFISH*: A folder named **`non_barcoded_key`** '
                            'with a file named'
                            ' **`sequential_key.csv`** containing the hyb and gene mapping'
                        ),
                        html.Div(html.Details([
                            html.Summary(html.B('smFISH key example')),
                            dbc.Table.from_dataframe(example_sm_key, striped=True, size='sm')
                        ]), style={'width': '50%',
                                   'font-size': '10pt',
                                   'margin': '20px'})
                    ]),
                    html.Li(dcc.Markdown(
                        '(optional) a folder named **`final_background`** '
                        'containing background images of each position '
                        '(**`MMStack_Pos#.ome.tif`**) for background subtraction'
                    )),
                    html.Li(dcc.Markdown(
                        '(optional) A folder named **`segmentation`** which contains '
                        'images to be used for segmentation - e.g. membrane stains. '
                        'These images should be named identically to the hyb round images: '
                        '`MMStack_Pos#.ome.tif` for each position.'
                    )),
                    html.Li(dcc.Markdown(
                        '(optional) A folder named **`Labeled_Images`** containing '
                        '*existing* segmentation masks produced off-line.'
                    )),
                    html.Li(dcc.Markdown(
                        'A positions file from the microscope, '
                        'ending in **`.pos`**, which is used to display the '
                        'experiment overview in the correct arrangement.'
                    ))
                ]),

            ])
        ], style={'margin-top': '10px'})
    ], width=6)
]
