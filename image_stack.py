import numpy as np
from math import sqrt, floor, ceil, pi, hypot
import h5py
from astropy import wcs

def pix2stamp(x, y, n):
    """
    produce slices for a square centred on x, y.
    x, y are real numbers
    return x_slice, y_slice

    >>> pix2stamp(1.6, 3.2, 1)
    (slice(2, 3, None), slice(3, 4, None))
    >>> pix2stamp(1.6, 3.2, 2)
    (slice(1, 3, None), slice(3, 5, None))
    >>> pix2stamp(1.6, 3.2, 3)
    (slice(1, 4, None), slice(2, 5, None))
    """
    if n % 2:
        return tuple(slice(int(round(i)) - (n-1)/2, int(round(i)) + (n+1)/2) for i in (x, y))
    else:
        return tuple(slice(int(ceil(i)) - n/2, int(ceil(i)) + n/2) for i in (x, y))

def semihex(data, axis=None):
    """
    Calculate standard deviation via semi-interhexile range.
    """
    h1, h5 = np.percentile(data, (100/6., 500/6.), axis=axis)
    return (h5-h1)/2.

class ImageStack(object):
    def __init__(self, h5filename, image_type='image', freq=None, steps=None, mode='r'):
        self.df = h5py.File(h5filename, mode)
        assert image_type in ('image', 'dirty', 'model', 'moment'), "Unsupported image_type %s." % image_type
        assert self.df.attrs['VERSION'] == '0.1', "Wrong version. Expected 0.1, got %s" % self.df.attrs['VERSION']
        self.image_type = image_type
        if freq is None:
            self.group = self.df['/']
        else:
            self.group = self.df[freq]
        self.data = self.group[image_type]
        self.header = self.group['header'].attrs
        self.wcs = wcs.WCS(self.header)
        self.channel = 0
        if steps is None:
            self.steps = [0, self.data.shape[-1]]
        else:
            self.steps = [steps[0], steps[1]]

    def update(self, freq=None, image_type=None, steps=None):
        if freq is not None:
            self.group = self.df[freq]
            self.data = self.group[self.image_type]
            self.header = self.group['header'].attrs
            self.wcs = wcs.WCS(self.header)

        if image_type is not None:
            assert image_type in ('image', 'dirty', 'model', 'moment'), "Unsupported image_type %s." % image_type
            self.image_type = image_type
            self.data = self.group[image_type]

        if steps is not None:
            self.steps = [steps[0], steps[1]]

    def check_slice(self, x, y, margin=0):
        """
        check slice is valid (not off the edge of the image)
        """
        sqrt_beamsize = sqrt(self.get_pixel_beam())
        x_slice, y_slice = pix2stamp(x, y, margin+int(round(10*max(1, sqrt_beamsize))))
        if not all(0 < s < self.data.shape[2] for s in (x_slice.start, x_slice.stop)):
            return 1
        if not all(0 < s < self.data.shape[1] for s in (y_slice.start, y_slice.stop)):
            return 1
        return 0

    def get_interval(self):
        """
        Get interval between neighbouring images
        """
        return self.group.attrs['TIME_INTERVAL']

    def get_intervals(self):
        """
        Get series of intervals
        """
        return self.group.attrs['TIME_INTERVAL']*np.arange(self.steps[1]-self.steps[0])

    def world2pix(self, ra, dec, floor=True):
        """
        return pixel coordinates x, y
        NB x is the fastest varying axis!
        """
        pixcoord = self.wcs.celestial.wcs_world2pix(np.array([[ra, dec]]), 0)
        if floor:
            pixcoord = np.round(pixcoord).astype(np.int)
        return pixcoord[0, 0], pixcoord[0, 1]

    def get_pixel_beam(self):
        """
        Get size of *synthesised* beam in pixels.
        NB assumes square pixels and roughly circular synth beam
        """
        return abs(pi*0.25*self.header['BMAJ']*self.header['BMIN']/(self.header['CDELT1']*self.header['CDELT2']))
    
    def get_scale(self):
        if 'SCALE' in self.group['beam'].attrs:
            return self.group['beam'].attrs['SCALE']
        return 1.0

    def pix2beam(self, x, y, avg_pol=True):
        """
        get beam corresponding to x,y
        """
        beam = self.group['beam'][:, y, x, self.channel, 0]
        beam *= self.get_scale()
        if avg_pol is True:
            if not np.any(beam):
                return 0.0
            return np.average(1/beam, weights=beam**2)**-1
        else:
            return beam

    def world2beam(self, x, y, avg_pol=True):
        """
        get beam corresponding to ra,dec
        """
        x, y = self.world2pix(ra, dec)
        return self.pix2beam(x, y, avg_pol)

    def pix2ts(self, x, y, avg_pol=True, correct=True):
        """
        Produce a timeseries for a given RA and Decl.
        avg_pol=True, correct=True Average polarisation having corrected for primary beam
        avg_pol=True, correct=False Average polarisations weighting for primary beam
        avg_pol=False, correct=True Return both polarisations corrected for primary beam
        avg_pol=False, correct=False Return both polarisations, no primary beam correction
        """
        ts = self.data[:, y, x, self.channel, self.steps[0]:self.steps[1]].astype(np.float_)
        if avg_pol is True:
            beam = self.pix2beam(x, y, False)
            ts = np.average(ts/beam[:, np.newaxis], axis=0, weights=beam**2)
            if correct is True:
                return ts
            return ts*hypot(beam[0], beam[1])
        else:
            if correct is True:
                beam = self.pix2beam(x, y, False)
                return ts/beam[:, np.newaxis]
            return ts

    def world2ts(self, ra, dec, avg_pol=True, correct=True):
        """
        Produce a timeseries for a given RA and Decl.
        avg_pol=True, correct=True Average polarisation having corrected for primary beam
        avg_pol=True, correct=False Average polarisations weighting for primary beam
        avg_pol=False, correct=True Return both polarisations corrected for primary beam
        avg_pol=False, correct=False Return both polarisations, no primary beam correction
        """
        x, y = self.world2pix(ra, dec)
        return self.pix2ts(x, y, avg_pol, correct)

    def slice2cube(self, x_slice, y_slice, avg_pol=True, correct=True):
        """
        Produce an nxm cube centred
        avg_pol=True, correct=True Average polarisation having corrected for primary beam
        avg_pol=True, correct=False Average polarisations weighting for primary beam
        avg_pol=False, correct=True Return both polarisations corrected for primary beam
        avg_pol=False, correct=False Return both polarisations, no primary beam correction
        """
        ts = self.data[:, y_slice, x_slice, self.channel, self.steps[0]:self.steps[1]].astype(np.float_)
        _, beam = np.broadcast_arrays(ts, self.pix2beam(x_slice, y_slice, False)[..., np.newaxis])
        if avg_pol is True:
            ts = np.average(ts/beam, axis=0, weights=beam**2)
            if correct is True:
                return ts
            
            return ts*np.hypot(beam[0, ...], beam[1, ...])
        else:
            if correct is True:
                return ts/beam
            return ts

    def pix2cube(self, x, y, n, m=None, avg_pol=True, correct=True):
        """
        Produce an nxm cube centred on a given RA and Decl.
        n is required
        m defaults to n
        avg_pol=True, correct=True Average polarisation having corrected for primary beam
        avg_pol=True, correct=False Average polarisations weighting for primary beam
        avg_pol=False, correct=True Return both polarisations corrected for primary beam
        avg_pol=False, correct=False Return both polarisations, no primary beam correction
        """
        if m is None:
            m = n
        x_slice, y_slice = pix2stamp(x, y, n)
        return self.slice2cube(x_slice, y_slice, avg_pol, correct)

    def pix2rms(self, x, y, avg_pol=True, correct=True):
        """
        Calculate local rms centred on x,y for each timestep
        avg_pol=True, correct=True Average polarisation having corrected for primary beam
        avg_pol=True, correct=False Average polarisations weighting for primary beam
        avg_pol=False, correct=True Return both polarisations corrected for primary beam
        avg_pol=False, correct=False Return both polarisations, no primary beam correction
        """
        cube = self.pix2cube(x, y, int(round(10*max(1, sqrt_beamsize))), avg_pol=avg_pol, correct=correct)
        if avg_pol is True:
            #cube has axes x, y, time
            return semihex(cube.reshape(cube.shape[0]*cube.shape[1], cube.shape[2]), axis=0)
        else:
            #cube has axes pol, x, y, time
            return semihex(cube.reshape(cube.shape[0], cube.shape[1]*cube.shape[2], cube.shape[3]), axis=1)


if __name__ == "__main__":
    """
    run tests
    """
    import doctest
    doctest.testmod()
