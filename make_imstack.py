#!/usr/bin/env python
import os, datetime, logging, h5py, contextlib
import numpy as np
from optparse import OptionParser 
from astropy.io import fits

##
## Define some necessary parameters
##----------------------------------

VERSION = "0.1"
CACHE_SIZE=8 
N_PASS=1
TIME_INTERVAL=0.5
TIME_INDEX=1
POLS = 'XX,YY'
STAMP_SIZE=16
SLICE = [0, 0, slice(None, None, None), slice(None, None, None)]
HDU = 0
PB_THRESHOLD = 0.1 # fraction of pbmax
SUFFIXES="image,model"
#N_TIMESTEPS=591
N_TIMESTEPS=50
N_CHANNELS=1
DTYPE = np.float16
FILENAME="{obsid}-t{time:04d}-{pol}-{suffix}.fits"
FILENAME_BAND="{obsid}_{band}-t{time:04d}-{pol}-{suffix}.fits"
PB_FILE="{obsid}-{pol}-beam.fits"
PB_FILE_BAND="{obsid}_{band}-{pol}-beam.fits"


##
## Define some useful commandline arguments
##------------------------------------------

parser = OptionParser(usage = "usage: obsid" +
"""
    Convert a set of wsclean images into an hdf5 image cube
""")
parser.add_option("-n", default=N_TIMESTEPS, dest="n", type="int", help="number of timesteps to process [default: %default]")
parser.add_option("--start", default=TIME_INDEX, dest="start", type="int", help="starting time index [default: %default]")
parser.add_option("--n_pass", default=N_PASS, dest="n_pass", type="int", help="number of passes [default: %default]")
parser.add_option("--step", default=TIME_INTERVAL, dest="step", type="float", help="time between timesteps [default: %default]")
parser.add_option("--outfile", default=None, dest="outfile", type="str", help="outfile [default: [obsid].hdf5]")
parser.add_option("--suffixes", default=SUFFIXES, dest="suffixes", type="str", help="comma-separated list of suffixes to store [default: %default]")
parser.add_option("--bands", default=None, dest="bands", type="str", help="comma-separated list of contiguous frequency bands [default None]")
parser.add_option("--pols", default=POLS, dest="pols", type="str", help="comma-separated list of pols [default: %default]")
parser.add_option("--pb_thresh", default=PB_THRESHOLD, dest="pb_thresh", type="float", help="flag below this threshold [default: %default]")
parser.add_option("--stamp_size", default=STAMP_SIZE, dest="stamp_size", type="int", help="hdf5 stamp size [default: %default]")
parser.add_option("--skip_check_wsc_timesteps", action="store_true", dest="skip_check_wcs_timesteps", help="don't check WSClean timesteps")
parser.add_option("-v", "--verbose", action="count", dest="verbose", help="-v info, -vv debug")


##
## Parse any commandline arguments given by the user
##---------------------------------------------------

opts, args = parser.parse_args()

if not len(args) == 1:
    parser.error("incorrect number of arguments")

obsid = int(args[0])

if opts.verbose == 1:
    logging.basicConfig(level=logging.INFO)
elif opts.verbose > 1:
    logging.basicConfig(level=logging.DEBUG)

if opts.outfile is None:
    opts.outfile = "%d.hdf5" % obsid

opts.suffixes = opts.suffixes.split(',')
opts.pols = opts.pols.split(',')
if opts.bands is None:
    opts.bands = [None]
else:
    opts.bands = opts.bands.split(',')

if opts.skip_check_wcs_timesteps:
    logging.warn("Warning: not checking timesteps. Checking verbose output carefully is recommended!")


##
## Check if all required input files are present
##-----------------------------------------------

if opts.bands is None:
   for band in opts.bands:
       for suffix in opts.suffixes:
           for t in xrange(opts.start, opts.n+opts.start):
               infile = FILENAME.format(obsid=obsid, time=t, pol=p, suffix=suffix)
               if not os.path.exists(infile):
                  raise IOError, "couldn't find file %s" % infile
               logging.debug("%s found", infile)
else:
   for band in opts.bands:
       for suffix in opts.suffixes:
           for t in xrange(opts.start, opts.n+opts.start):
               for p in opts.pols:
                   infile = FILENAME_BAND.format(obsid=obsid, band=band, time=t, pol=p, suffix=suffix)
                   if not os.path.exists(infile):
                      raise IOError, "couldn't find file %s" % infile
                      logging.debug("%s found", infile)


##
## Create a new HDF5 with a chunk cache of specified size
##--------------------------------------------------------

propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
settings = list(propfaid.get_cache())
settings[2] *= CACHE_SIZE 
propfaid.set_cache(*settings)

if os.path.exists(opts.outfile):
    logging.warn("Warning: hdf5 file exists -overwriting")

with contextlib.closing(h5py.h5f.create(opts.outfile, fapl=propfaid)) as fid:
     df = h5py.File(fid,'w')


##
## Separate each frequency band into its own group in the HDF5 file
##------------------------------------------------------------------ 
for band in opts.bands:
    if band is None:
       group = df['/']
    else:
       group = df.create_group(band)
    group.attrs['TIME_INTERVAL'] = opts.step


##
## Determine the dimensionality of the datasets and chunks.  Also determine the
## size of the datasets and chunk in each of these dimensions.
##------------------------------------------------------------------------------

    if band is None:
       image_file = FILENAME.format(obsid=obsid, time=opts.start, pol=opts.pols[0], suffix=opts.suffixes[0])
    else:
       image_file = FILENAME_BAND.format(obsid=obsid, band=band, time=opts.start, pol=opts.pols[0], suffix=opts.suffixes[0])

    hdus = fits.open(image_file, memmap=True)
    image_size = hdus[HDU].data.shape[-1]
    data_shape = [image_size, image_size, opts.n]


##
## Read in beam shape data and attributes from appropriate FITS files and write into the
## new HDF5 file.
##---------------------------------------------------------------------------------------

    beam_shape = [image_size, image_size]
    pb_sum = np.zeros( beam_shape )
    beam_dataset_name = "beam-{pol}"
    for p, pol in enumerate(opts.pols):
        dname = beam_dataset_name.format(pol=pol)
        beam = group.create_dataset(dname, beam_shape, dtype=np.float32, compression="gzip")
        if band is None:
           hdus = fits.open(PB_FILE.format(obsid=obsid, pol=pol), memmap=True)
        else:
           hdus = fits.open(PB_FILE_BAND.format(obsid=obsid, band=band, pol=pol), memmap=True)
         
        beam[:, :] = hdus[HDU].data[SLICE]
        for key, item in hdus[0].header.iteritems():
            beam.attrs[key] = item
        pb_sum = pb_sum + np.sum(beam)


##
## Check for the presence of NaNs in the beam data
##-------------------------------------------------

    pb_sum = np.sqrt(pb_sum)/len(opts.pols)
    pb_mask = pb_sum > opts.pb_thresh*np.nanmax(pb_sum)
    pb_nan = pb_sum/pb_sum
    if np.any(np.isnan(pb_sum)):
       logging.warn("NaNs in primary beam")

##
## Read in header infromation from appropriate FITS files and output to the new HDF5 file
##----------------------------------------------------------------------------------------

    timesteps = group.create_dataset("WSCTIMES", (opts.n,), dtype=np.uint16)
    timesteps2 = group.create_dataset("WSCTIMEE", (opts.n,), dtype=np.uint16)
    if band is None:
       header_file = FILENAME.format(obsid=obsid, time=opts.n//2, pol=opts.pols[0], suffix=opts.suffixes[0])
    else:
       header_file = FILENAME_BAND.format(obsid=obsid, band=band, time=opts.n//2, pol=opts.pols[0], suffix=opts.suffixes[0])
        
    hdus = fits.open(header_file, memmap=True)
    header = group.create_dataset('header', data=[], dtype=DTYPE)
    for key, item in hdus[0].header.iteritems():
        header.attrs[key] = item

##
## Record the names of the original input FITS files.
##------------------------------------------------------------------
    for s, suffix in enumerate(opts.suffixes):
        filenames = group.create_dataset("%s_filenames" % suffix, (len(opts.pols), N_CHANNELS, opts.n), dtype="S%d" % len(header_file))
        for i in range(opts.n_pass):
            for t in xrange(opts.n):
                for p, pol in enumerate(opts.pols):
                    if band is None:
                       infile = FILENAME.format(obsid=obsid, time=t+opts.start, pol=pol, suffix=suffix)
                    else:
                       infile = FILENAME_BAND.format(obsid=obsid, band=band, time=t+opts.start, pol=pol, suffix=suffix)
                    filenames[p, 0, t] = infile  
##
## Update timestep information in the new HDF5 file
##--------------------------------------------------

                    if s==0 and p==0:
                       timesteps[t] = hdus[0].header['WSCTIMES']
                       timesteps2[t] = hdus[0].header['WSCTIMEE']
                    else:
                       if not opts.skip_check_wcs_timesteps:
                          assert timesteps[t] == hdus[0].header['WSCTIMES'], "Timesteps do not match %s in %s" % (opts.suffixes[0], infile)
                          assert timesteps2[t] == hdus[0].header['WSCTIMEE'], "Timesteps do not match %s in %s" % (opts.suffixes[0], infile)
                       else:
                          logging.debug(hdus[0].header['DATE-OBS'])
                          logging.debug(hdus[0].header['DATE-OBS'])

##
## Construct the necessary datasets to hold the image data
##---------------------------------------------------------

    for s, suffix in enumerate(opts.suffixes):
        for p, pol in enumerate(opts.pols):
            dname = "{suffix}-{pol}".format(suffix=suffix,pol=pol)
            data = group.create_dataset(dname, data_shape, compression="gzip", dtype=DTYPE )

##
## Read in data from the appropriate input FITS file
##---------------------------------------------------

    for s, suffix in enumerate(opts.suffixes):
        n_rows = image_size/opts.n_pass
        for i in range(opts.n_pass):
            logging.info("processing segment %d/%d" % (i+1, opts.n_pass))
            im_slice = [slice(n_rows*i, n_rows*(i+1)), slice(None, None, None)]
            fits_slice = SLICE[:-2] + im_slice

        for p, pol in enumerate(opts.pols):
            dname = "{suffix}-{pol}".format(suffix=suffix,pol=pol)
            dset = group[dname]
            for t in xrange(opts.n):

                if band is None:
                   infile = FILENAME.format(obsid=obsid, time=t+opts.start, pol=pol, suffix=suffix)
                else:
                   infile = FILENAME_BAND.format(obsid=obsid, band=band, time=t+opts.start, pol=pol, suffix=suffix)
                logging.info(" processing %s", infile)
                hdus = fits.open(infile, memmap=True)

##
## Write data into the new HDF5 datasets.  Filter the data array to remove data points where the beam shape is a NaN
##-------------------------------------------------------------------------------------------------------------------

                dset[n_rows*i:n_rows*(i+1), :, t] = np.where( pb_mask[n_rows*i:n_rows*(i+1), :], 
                                                              hdus[0].data[fits_slice],
                                                              np.nan ) * pb_nan[n_rows*i:n_rows*(i+1), :] 

