# Webfish: in-browser datavis for seqFISH

Webfish is a Dash app intended to provide easy visualization and exploration 
of seqFISH spatial transcriptomic datasets to researchers in the [Cai Lab](https://spatial.caltech.edu) and elsewhere.

Webfish pulls relevant files from cloud storage and can also be run on Amazon EC2. 

Written by Lincoln Ombelets in collaboration with Nick Rezaee in the Cai Lab, 2021-present.

## Running

Right now, Webfish reads in several configuration values from a file called `consts.yml`. These specify the cloud storage details (AWS S3 API style), local storage path, and names of specific files to look for in each dataset. These file names are likely to change as we develop webfish.

Webfish looks for the following environment variables on startup:
* credentials (`WEBFISH_CREDS`) by default: the name of this variable is set in `consts.yml`, and it is expected to contain a path to an AWS CLI config-style file that lists an AWS Access key ID and an AWS Secret Access Key. If it doesn't find this, it will default to the `boto3` library's default settings, which looks in `~/.aws/` for credential files. See the AWS CLI docs for more details on that.
* `WEBFISH_HOST`: the host IP to run the server on. Defaults to 127.0.0.1 (localhost), which is good for local testing, but 0.0.0.0 is used to deploy on a remote server, for example.
* `WEBFISH_PORT`: the port to deploy the Dash app on. Defaults to 8050.
* 
So an example invocation would be:

```env WEBFISH_CREDS=/path/to/credentials WEBFISH_HOST=0.0.0.0 python index.py```
