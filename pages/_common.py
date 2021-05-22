from copy import copy
from lib.client import DataClient
from app import config, s3_client
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html


# Convenience dict with the page name (short), title (for display), and description
# TODO: Put this into DataClient
page_index = {k: {'title': v.get('title', k),
                  'description': v.get('description', '')
                  }
              for k, v in config['pages'].items()}

data_clients = {
    pagename: DataClient(pagename=pagename, config=config, s3_client=s3_client)
    for pagename in page_index.keys()
}
data_clients['__all__'] = DataClient(pagename=None, config=config, s3_client=s3_client)


def sync_with_s3():
    data_clients['__all__'].sync_with_s3(download=True)

    for cli in data_clients.values():
        cli.sync_with_s3(download=False)


sync_with_s3()

# Save the dataset record to compare to individual pages
all_datasets = data_clients['__all__'].datasets.copy()


class ComponentManager:
    """
    ComponentManager
    ----------------
    A convenience class for dynamic Dash apps that need to reset or modify
    various groups of components en masse, modifying different attributes
    on a case-by-case basis. Instead of hard-coding the "cleared" or "base"
    components, we supply them to this class as well as groups of components
    that are often manipulated together (like related fields of a form that
    all depend on some other field). Then resetting or updating any component
    is as simple as using this class to fetch it with appropriate modifications.

    Also, if we want to modify something about our base components,
    we only have to modify one definition. The specification of the
    layout variable in the page file is more compact and intuitive,
    if you use descriptive names for `component_groups.

    In the future we could read the long dict of components from a file.
    """

    def __init__(
        self,
        clear_components,
        id_prefix='',
        component_groups=None
    ):
        component_groups = component_groups or {}

        self.clear_components = copy(clear_components)
        self.component_groups = copy(component_groups)

    def component(self, id, **options):
        comp = copy(self.clear_components[id])

        for k, v in options.items():
            setattr(comp, k, v)

        return comp

    def component_group(self, name, tolist=False, options={}):
        """

        options is a dict whose keys can be a mixture of **component ids**
        and **component attributes**. If a component id, the value is expected
        to be a dict of attribute updates to apply to the component.
        If a component attribute, the value is expected to be whatever value
        that attribute of **all** components within the group should take.

        For example, if you wanted to return group 'foo', update component
        'bar's `options` attribute, and set `disabled=True` for all components in
        the group:

        cm.component_group('foo', options={
            'bar': dict(options=['baz']),
            'disabled': True
        })

        tolist specifies whether only the list of components should be returned,
        as you would return it from a Dash callback. Otherwise it is a dict of
        pairs like `component_id: component`.
        """
        ids = self.component_groups[name]

        # get the component-specific options
        id_options = {i: options.get(i, {}) for i in ids}

        # get the whole-group options
        for opt, val in options.items():
            # Any options entries directly referring to an attribute
            # get applied to all members of this component group
            if opt not in ids:
                id_options = {
                    i: oldopts | {opt: val}
                    for i, oldopts in id_options.items()
                }

        comps = {i: self.component(i, **opt) for i, opt in id_options.items()}

        if tolist:
            return list(comps.values())

        return comps

