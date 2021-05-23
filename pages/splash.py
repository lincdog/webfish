import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

PAGENAME = 'home'

layout = [
    html.H1('This is the home page.'),
    dcc.Markdown('Some day, it will have nice and helpful information about this app,'
                 ' and maybe a [link](https://github.com/lincdog/webfish) to the GitHub repo'
                 ' and also to the [Cai Lab homepage](https://spatial.caltech.edu).')
]
