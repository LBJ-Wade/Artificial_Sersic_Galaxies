import numpy as np
from astropy.modeling.models import Sersic2D
import matplotlib.pyplot as plt
from photutils.datasets import make_noise_image
from astropy.io import fits

n_gal=30
shape = (800, 1000)
image0=np.zeros(shape)
Sersic_sim_box=100
noise_mean=5.0

x,y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

np.random.seed(50)
for i in range(n_gal):
    mod = Sersic2D(amplitude = abs(np.random.normal(loc=50, scale=20)),
                   r_eff = 5+np.random.random()*20,
                   n=abs(np.random.normal(loc=1.5, scale=0.2)),
                   x_0=np.random.randint(0, high=shape[1]-1),
                   y_0=np.random.randint(0, high=shape[0]-1),
                   ellip=abs(np.random.normal(loc=0.5, scale=0.2)),
                   theta=np.pi*np.random.random()-np.pi/2)
    img = mod(x, y)
    image0+=img


image_galaxies = image0 + make_noise_image(shape, distribution='poisson', mean=noise_mean)

hdu = fits.PrimaryHDU(image_galaxies)
hdu.writeto('image_galaxies.fits', overwrite=True)
log_img = np.log10(image_galaxies)

plt.figure()
plt.imshow(log_img, origin='lower', interpolation='nearest', cmap='gray', vmin=0, vmax=4)
plt.tight_layout()
plt.savefig('image_galaxies.jpg')
plt.show()
