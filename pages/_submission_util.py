import io
import json
import re

from lib.util import sanitize, sort_as_num_or_str
from pages._page_util import PageHelper


# Helper functions for form item processing used by SubmissionHelper
def _one_z_process(form_id, form_val, current):

    if form_val:
        try:
            form_val = form_val[0]
        except TypeError:
            pass

        current['z slices'] = str(form_val)

    return current


def _position_process(form_id, form_val, current):
    if not form_val:
        current['positions'] = ''
    else:
        current['positions'] = ','.join([str(p) for p in form_val])

    return current


def _checklist_process(form_id, form_val, current):
    update = {k: "true" for k in form_val}

    current.update(update)

    return current


def _pp_checklist_process(form_id, form_val, current):
    update = {}

    tophat_raw = 'tophat raw data kernel size'
    if tophat_raw in form_val:
        update[tophat_raw] = '10'
        form_val.remove(tophat_raw)

    dilate_bg = 'dilate background kernel'
    if dilate_bg in form_val:
        update[dilate_bg] = '10'
        form_val.remove(dilate_bg)

    current.update(update)

    current = _checklist_process(form_id, form_val, current)

    return current


def _decoding_channel_process(form_id, form_val, current):

    decoding_method = current.get('decoding method')

    if decoding_method == 'syndrome' and form_val == 'non barcoded':
        raise ValueError('Syndrome decoding only works with barcoded data.')

    cur_decoding = current.get('decoding', None)

    if cur_decoding is None:
        # If there is not yet a decoding key, add it and return the dict
        current['decoding'] = str(form_val)

        return current

    elif cur_decoding == 'individual':
        if isinstance(form_val, list):
            # Individual is selected and we are handling the list of selected
            # channels.
            current['decoding'] = {
                'individual': sort_as_num_or_str(form_val)
            }
        elif not form_val:
            raise ValueError('Must specify at least one channel to decode '
                             'if "individual" is selected')
        return current
    else:
        return current


def _decoding_algorithm_process(form_id, form_val, current):

    if form_val == 'syndrome':
        current['decoding method'] = 'syndrome'

    return current


def _decoding_syndrome_process(form_id, form_val, current):
    cur_decoding = current.get('decoding method')

    form_id_to_json_syndrome = {
        'sb-syndrome-lateral-variance': 'lateral variance factor',
        'sb-syndrome-z-variance': 'z variance factor',
        'sb-syndrome-logweight-variance': 'log weight variance factor'
    }

    if cur_decoding == 'syndrome':
        json_key = form_id_to_json_syndrome[form_id]
        current[json_key] = str(form_val)

    return current


def _dotdetection_bright_dots(form_id, form_val, current):
    cur_dd = current.get('dot detection', None)

    if cur_dd == 'biggest jump 3d':
        if 'keep' in form_val:
            current['remove very bright dots'] = 'False'
        else:
            current['remove very bright dots'] = 'True'

    return current


def _dotdetection_pyparams_process(form_id, form_val, current):
    cur_dd = current.get('dot detection', None)

    form_id_to_json_pyparams = {
        'sb-threshold-select': 'threshold',
        'sb-minsigma-dotdetection': 'min sigma dot detection',
        'sb-maxsigma-dotdetection': 'max sigma dot detection',
        'sb-numsigma-dotdetection': 'num sigma dot detection',
        'sb-overlap-dotdetection': 'overlap'
    }

    if cur_dd == 'biggest jump 3d':
        json_key = form_id_to_json_pyparams[form_id]

        current[json_key] = str(form_val)

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
                'email': 'lombelets@caltech.edu'
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

        'sb-preprocessing-checklist': _pp_checklist_process,
        'sb-tophat-kernel-size': 'tophat kernel size',
        'sb-rollingball-kernel-size': 'rolling ball kernel size',
        'sb-blur-kernel-size': 'blur kernel size',

        'sb-dot detection-select': 'dot detection',
        'sb-strictness-select': 'strictness',

        'sb-remove-bright-dots': _dotdetection_bright_dots,
        'sb-threshold-select': _dotdetection_pyparams_process,
        'sb-minsigma-dotdetection': _dotdetection_pyparams_process,
        'sb-maxsigma-dotdetection': _dotdetection_pyparams_process,
        'sb-numsigma-dotdetection': _dotdetection_pyparams_process,
        'sb-overlap-dotdetection': _dotdetection_pyparams_process,

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

        'sb-decoding-algorithm': _decoding_algorithm_process,
        'sb-decoding-select': _decoding_channel_process,
        'sb-individual-channel-select': _decoding_channel_process,

        # FIXME: verify correct JSON key values with Jonathan
        'sb-syndrome-lateral-variance': _decoding_syndrome_process,
        'sb-syndrome-z-variance': _decoding_syndrome_process,
        'sb-syndrome-logweight-variance': _decoding_syndrome_process
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

        The order of the keys in form_status are determined by the dict of
        form components in submission.py, cm.clear_components. This is
        very important because the callables defined above are subject to the
        order of keys, since the JSON dict is built up sequentially.

        """

        # The clusters key is always the same (at least for now)
        out = {
            "clusters": {
                "ntasks": "1",
                "mem-per-cpu": "10G",
                "email": "lombelets@caltech.edu"
            },
            "__ERRORS__": []
        }

        # This is special as it is not in the final dict but becomes the
        # filename of the JSON file.
        analysis_name = ''

        selected_stages.append('basic-metadata')

        # Add the advanced segmentation parameters after the segmentation parameters
        if 'segmentation' in selected_stages:
            selected_stages.insert(
                selected_stages.index('segmentation')+1,
                'segmentation-advanced'
            )

        # Add the python dot detection parameters after dot detection
        if 'dot detection' in selected_stages:
            selected_stages.insert(
                selected_stages.index('dot detection')+1,
                'dot detection-python'
            )

        # Add the syndrome decoding parameters after decoding
        if 'decoding' in selected_stages:
            selected_stages.insert(
                selected_stages.index('decoding')+1,
                'decoding-syndrome'
            )

        # Begin the list of which form IDs we are going to care about.
        # We always care about the user and dataset selection.
        selected_form_ids = ['user-select', 'dataset-select']

        # Add the form IDs from all the selected stages
        for s in selected_stages:
            selected_form_ids.extend(self.cm.component_groups[s])

        # For each form-id: form-value pair
        for k in selected_form_ids:

            v = form_status.get(k, '__NONE__')

            if v == '__NONE__':
                continue

            if k == 'sb-analysis-name' and v:
                analysis_name = sanitize(v, delimiter_allowed=False)
            elif k in self.id_to_json_key:
                # If the form-id is in the id_to_json_key dict, fetch
                # the corresponding value (a string or a function)
                prekey = self.id_to_json_key[k]

                if callable(prekey):
                    # If a function, directly set the dictionary to the
                    # result of calling the function on the current output dict
                    try:
                        out = prekey(k, v, out)
                    except ValueError as e:
                        out['__ERRORS__'].append(e)
                else:
                    # else (a string), make a one-element dict that just
                    # assigns the form value to the JSON key
                    out.update({prekey: str(v)})

        return analysis_name, out
