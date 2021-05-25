import gc
import skimage.util as skiu
from pathlib import Path
from lib.util import ImageMeta, safe_imwrite

"""
preuploaders.py
-------------
This file contains classes and functions referred to as "preuploads":
they take raw input files that are located on the master storage (HPC),
and perform processing on them before they are uploaded to S3. This function
is analogous to `generators.py` but happens on the master storage computer,
*before* the files are uploaded to S3. 

The classes are only namespaces with class/static methods defined; we could
equally use namedtuples or similar. The `source_files` and `raw_files` sections of
the Webfish config file for each page can specify a preupload class
and a function to call for each source or raw file. When the cron job on the HPC
finds new files that haven't been uploaded to S3 yet, it first calls 
`server.DataServer.run_preuploads` on them and then only uploads the successfully
processed results of that. 

The preuploader's job is to do some processing and return the path to the
processed file. It is expected to check if the file exists already - this is not enforced
by DataServer. Because this is happening on the HPC, the output file location is specified
by the config entry `preupload_root`, currently in my (Lincoln) personal folder. DataServer
updates the filenames as the preupload jobs succeed, to point to this new location.
"""


class DotDetectionPreupload:

    @classmethod
    def compress_raw_im(
        cls,
        inrow,
        outpattern,
        savedir
    ):
        gc.enable()
        if not inrow:
            return None

        im = inrow['filename']
        outfile = Path(savedir, outpattern.format_map(inrow))

        if outfile.is_file():
            return outfile.relative_to(savedir), False

        try:
            cls.compress_8bit(im, 'DEFLATE', outfile)
            gc.collect()
        except Exception as e:
            return im, e

        return Path(outfile).relative_to(savedir), False

    @staticmethod
    def compress_8bit(
        imgfilename,
        compression='DEFLATE',
        outfile=None
    ):
        err = None
        im = None
        imarr = None

        try:
            im = ImageMeta(imgfilename)
            imarr = im.asarray()

            safe_imwrite(
                skiu.img_as_ubyte(imarr),
                outfile,
                compression=compression
            )
        except (PermissionError, IOError, OSError) as e:
            err = e
        finally:
            del imarr
            gc.collect()
            im.close()

        if err:
            raise err

        return outfile
