import io
import json
import re

from lib.util import sanitize
from pages._page_util import PageHelper


# Helper functions for form item processing used by SubmissionHelper
def _one_z_process(item, current):

    if item:
        try:
            item = item[0]
        except TypeError:
            pass

        current['z slices'] = str(item)

    return current


def _position_process(positions, current):
    if not positions:
        current['positions'] = ''
    else:
        current['positions'] = ','.join([str(p) for p in positions])

    return current


def _checklist_process(checked, current):
    update = {k: "true" for k in checked}

    current.update(update)

    return current


def _decoding_channel_process(arg, current):

    cur_decoding = current.get('decoding', None)
    print(f'cd: {cur_decoding}, {arg}')

    if cur_decoding is None:
        # If there is not yet a decoding key, add it and return the dict
        current['decoding'] = str(arg)

        return current

    elif cur_decoding == 'individual':
        if isinstance(arg, list):
            # Individual is selected and we are handling the list of selected
            # channels.
            current['decoding'] = {
                'individual': [str(a) for a in arg]
            }
        elif not arg:
            raise ValueError('Must specify at least one channel to decode '
                             'if "individual" is selected')
        return current
    else:
        return current


def _dotdetection_threshold_process(arg, current):
    cur_dd = current.get('dot detection', None)

    if cur_dd != 'matlab 3d':
        current['threshold'] = str(arg)

    return current


def _dotdetection_minsigma_process(arg, current):
    cur_dd = current.get('dot detection', None)

    if cur_dd == 'biggest jump 3d':
        current['min sigma dot detection'] = str(arg)

    return current


def _dotdetection_maxsigma_process(arg, current):
    cur_dd = current.get('dot detection', None)

    if cur_dd == 'biggest jump 3d':
        current['max sigma dot detection'] = str(arg)

    return current


def _dotdetection_numsigma_process(arg, current):
    cur_dd = current.get('dot detection', None)

    if cur_dd == 'biggest jump 3d':
        current['num sigma dot detection'] = str(arg)

    return current


def _dotdetection_overlap_process(arg, current):
    cur_dd = current.get('dot detection', None)

    if cur_dd == 'biggest jump 3d':
        current['overlap'] = str(arg)

    return current


class SubmissionHelper(PageHelper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def put_analysis_request(
            self,
            user,
            dataset,
            analysis_name,
            dot_detection='biggest jump 3d'
    ):
        analysis_dict = {
            'personal': user,
            'experiment_name': dataset,
            'dot detection': dot_detection,
            'dot detection test': 'true',
            'visualize dot detection': 'true',
            'strictness': 'multiple',
            'clusters': {
                'ntasks': '1',
                'mem-per-cpu': '10G',
                'email': 'nrezaee@caltech.edu'
            }
        }

        dict_bytes = io.BytesIO(json.dumps(analysis_dict).encode())
        # These are "characters to avoid" in keys according to AWS docs
        # \, {, }, ^, [, ], %, `, <, >, ~, #, |
        # TODO: Sanitize filenames on server side before uploading too; mostly
        #   potential problem for user-defined dataset/analysis names
        analysis_sanitized = re.sub('[\\\\{^}%` \\[\\]>~<#|]', '', analysis_name)
        keyname = f'json_analyses/{analysis_sanitized}.json'

        try:
            self.data_client.client.client.upload_fileobj(
                dict_bytes,
                Bucket=self.data_client.bucket_name,
                Key=keyname
            )
        except Exception as e:
            return str(e)

        print(f'analysis_dict: {json.dumps(analysis_dict, indent=2)}')

        return analysis_sanitized

    # staticmethods are wrapped in a descriptor protocol, so we need
    # to access the underlying function in order to make this dict work
    # properly. An alternative would be to assign the dict as a class
    # variable after the class definition,
    id_to_json_key = {
        'user-select': 'personal',
        'dataset-select': 'experiment_name',

        'sb-position-select': _position_process,
        'sb-one-z-select': _one_z_process,

        'sb-alignment-select': 'alignment',

        'sb-dot detection-select': 'dot detection',
        'sb-bg-subtraction': _checklist_process,
        'sb-strictness-select': 'strictness',
        'sb-threshold-select': _dotdetection_threshold_process,

        'sb-minsigma-dotdetection': _dotdetection_minsigma_process,
        'sb-maxsigma-dotdetection': _dotdetection_maxsigma_process,
        'sb-numsigma-dotdetection': _dotdetection_numsigma_process,
        'sb-overlap-dotdetection': _dotdetection_overlap_process,

        'sb-segmentation-select': 'segmentation',
        'sb-segmentation-checklist': _checklist_process,
        'sb-edge-deletion': 'edge deletion',
        'sb-nuclei-distance': 'distance between nuclei',
        'sb-nuclei-channel': 'nuclei channel number',
        'sb-cyto-channel': 'cyto channel number',
        'sb-cyto-radius': 'cyto radius',
        'sb-cyto-cell-prob-threshold': 'cyto cell prob threshold',
        'sb-cyto-flow-threshold': 'cyto flow threshold',
        'sb-nuclei-radius': 'nuclei radius',
        'sb-cell-prob-threshold': 'cell_prob_threshold',
        'sb-flow-threshold': 'flow_threshold',

        'sb-decoding-select': _decoding_channel_process,
        'sb-individual-channel-select': _decoding_channel_process
    }

    def form_to_json_output(self, form_status, selected_stages):
        """
        form_to_json_output
        -------------------
        Takes the status of the form (a dict where the keys are the DOM
        id of each form element and the values are the value of that form
        element) and performs the necessary processing to generate a dict
        that will be written as a JSON file to submit to the pipeline.

        The id_to_json_key dict (above) is crucial because it specifies either
        * the mapping from the form element id to the pipeline JSON input key,
            leaving the value unchanged
        or:
        * a callable that takes the value of the form element and returns a dict
            that will be used to update() the JSON dict in progress.

        """

        # The clusters key is always the same (at least for now)
        out = {
            "clusters": {
                "ntasks": "1",
                "mem-per-cpu": "10G",
                "email": "nrezaee@caltech.edu"
            },
            "__ERRORS__": []
        }

        # This is special as it is not in the final dict but becoems the
        # filename of the JSON file.
        analysis_name = ''

        selected_stages.append('basic-metadata')

        if 'segmentation' in selected_stages:
            selected_stages.insert(
                selected_stages.index('segmentation')+1,
                'segmentation-advanced'
            )

        if 'dot detection' in selected_stages:
            selected_stages.insert(
                selected_stages.index('dot detection')+1,
                'dot detection-python'
            )

        selected_form_ids = ['user-select', 'dataset-select']
        for s in selected_stages:
            selected_form_ids.extend(self.cm.component_groups[s])

        # For each form-id: form-value pair
        for k in selected_form_ids:

            v = form_status.get(k, '__NONE__')

            if v == '__NONE__':
                continue

            if k == 'sb-analysis-name' and v:
                analysis_name = sanitize(v, delimiter_allowed=False)
            elif k in self.id_to_json_key.keys():
                # If the form-id is in the id_to_json_key dict, fetch
                # the corresponding value (a string or a function)
                prekey = self.id_to_json_key[k]

                if callable(prekey):
                    # If a function, directly set the dictionary to the
                    # result of calling the function on the current output dict
                    try:
                        out = prekey(v, out)
                    except ValueError as e:
                        out['__ERRORS__'].append(e)
                else:
                    # else (a string), make a one-element dict that just
                    # assigns the form value to the JSON key
                    out.update({prekey: str(v)})

        return analysis_name, out
