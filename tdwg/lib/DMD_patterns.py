import numpy as np
from scipy.linalg import circulant
from scipy.signal import sawtooth
from PIL import Image
import matplotlib.pyplot as plt

# Vialux V7000 ("2D waveguide DMD")
resY=1024
resX=768
pixel_size = 13.7 # um
# "SHG PNN waveguide DMD"
# resX=1280
# resY=800

X,Y=np.meshgrid(np.arange(resX),np.arange(resY))
Z=np.flipud(np.round(2*np.pi/np.sqrt(2)*(X+Y)).astype(int))
lam=np.flipud(np.round(1/np.sqrt(2)*(X-Y)).astype(int))
X=X.astype(int)
Y=Y.astype(int)
lamv=np.arange(np.amin(lam),np.amax(lam))
period=128
resA = len(lamv)

def gen_bin_img(amps,X,Y,Z,lam,period=128): #need to precompute grid vectors 
    #idea is that the energy into the diffracted beam is quadratically related to its duty cycle. 
    #The DMD itself is however rotated at 45 degrees, so each line corresponds to a 45 degree line through the image
    #we assume amps is between 0 and 1
    img=np.sin(2*np.pi*Z/period)+(2*amps[lam-np.min(lam)-1]-1) #need to do offset to get full range of amplitude modulation
    img[img>0]=255
    img[img<=0]=0
    return img.astype('uint8')

Z_temp = np.sin(2*np.pi*Z/period)-1
lam_temp = lam-np.min(lam)-1

lam_temp2 = lam_temp.astype('int16')
Z_temp2 = (-127*Z_temp/2).astype('uint8')

def get_amp_img(amps):
    amps = (127*(amps) + 127.5).astype('uint8')
    img = amps[lam_temp2] 
    img -= Z_temp2 #need to do offset to get full range of amplitude modulation
    mask = img > 127
    img[mask] = 255
    img[np.bitwise_not(mask)]=0
    return img

def generate_all_off(resX = resX, resY = resY):
    return np.zeros([resX, resY]).astype('uint8')

def generate_all_on(resX = resX, resY = resY):
    return 255*(np.ones([resX, resY]).astype('uint8'))

def generate_spotty_image(size, separation, resX = resX, resY = resY):
    # e.g. for size = 2 and separation = 3:
    #
    # array([[255, 255,   0,   0,   0, 255, 255,   0,   0,   0],
    #        [255, 255,   0,   0,   0, 255, 255,   0,   0,   0],
    #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #        [255, 255,   0,   0,   0, 255, 255,   0,   0,   0],
    #        [255, 255,   0,   0,   0, 255, 255,   0,   0,   0],
    #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=uint8)
    
    lshift = int(np.floor(size/2))
    rshift = int(np.ceil(size/2))

    image = np.zeros([resY, resX], dtype = np.uint8)

    for xpos in np.arange(0 + lshift, resX, size + separation):
        for ypos in np.arange(0 + lshift, resY, size + separation):
            ymin = np.max([0, ypos-lshift])
            ymax = np.min([resY, ypos+rshift])
            xmin = np.max([0, xpos-lshift])
            xmax = np.min([resX, xpos+rshift])
            image[ymin:ymax, xmin:xmax] = 255
            
    return image

def generate_vertical_grating(len_1s, len_0s = None, phaseshift = 0, resX = resX, resY = resY):
    # e.g. for len_1s = 2 and len_0s = 3:
    #
    # array([[  0,   0,   0, 255, 255,   0,   0,   0, 255, 255],
    #        [  0,   0,   0, 255, 255,   0,   0,   0, 255, 255],
    #        [  0,   0,   0, 255, 255,   0,   0,   0, 255, 255],
    #        [  0,   0,   0, 255, 255,   0,   0,   0, 255, 255],
    #        [  0,   0,   0, 255, 255,   0,   0,   0, 255, 255],
    #        [  0,   0,   0, 255, 255,   0,   0,   0, 255, 255],
    #        [  0,   0,   0, 255, 255,   0,   0,   0, 255, 255],
    #        [  0,   0,   0, 255, 255,   0,   0,   0, 255, 255],
    #        [  0,   0,   0, 255, 255,   0,   0,   0, 255, 255],
    #        [  0,   0,   0, 255, 255,   0,   0,   0, 255, 255]], dtype=uint8)
    
    if len_0s == None: len_0s = len_1s
    if resY%(len_0s + len_1s) != 0: print('Number of rows is not divisible by len_1s + len_0s; incomplete grating')

    x = np.arange(resX)
    y = np.arange(resY)
    xx, yy = np.meshgrid(x,y)
    inds = (yy-phaseshift) % (len_0s + len_1s)
    grating = 255*(inds < len_1s).T

    return grating.astype(np.uint8)

def generate_horizontal_grating(len_1s, len_0s = None, phaseshift = 0, resX = resX, resY = resY):
    # e.g. for len_1s = 2 and len_0s = 3:
    #
    # array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    #        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    #        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255]], dtype=uint8)
    
    grating = generate_vertical_grating(len_1s, len_0s, phaseshift, resX = resY, resY = resX)
    return grating.transpose()

def generate_diagonal_grating(len_1s, len_0s = None, resX = resX, resY = resY):
    #Diagonal Grating
    # e.g. for len_1s = 5 and len_0s = 3:

    # array([[  0, 255, 255, 255, 255, 255,   0,   0,   0, 255],
    #        [  0,   0, 255, 255, 255, 255, 255,   0,   0,   0],
    #        [  0,   0,   0, 255, 255, 255, 255, 255,   0,   0],
    #        [255,   0,   0,   0, 255, 255, 255, 255, 255,   0],
    #        [255, 255,   0,   0,   0, 255, 255, 255, 255, 255],
    #        [255, 255, 255,   0,   0,   0, 255, 255, 255, 255],
    #        [255, 255, 255, 255,   0,   0,   0, 255, 255, 255],
    #        [255, 255, 255, 255, 255,   0,   0,   0, 255, 255],
    #        [  0, 255, 255, 255, 255, 255,   0,   0,   0, 255],
    #        [  0,   0, 255, 255, 255, 255, 255,   0,   0,   0]], dtype=uint8)

    if not len_0s: len_0s = len_1s
    if resY%(len_0s + len_1s) != 0: print('Number of rows is not divisible by len_1s + len_0s; incomplete grating')
    unit_array = np.ones(len_0s + len_1s)
    unit_array[0:len_0s] = 0.
    line = np.tile(unit_array, int(resY/(len_0s + len_1s))+1)
    diag_grating = circulant(line)
    rescaled_diag_grating = (255.0 / diag_grating[0:resX].max() * (diag_grating[0:resX]\
                            - diag_grating[0:resX].min()))
    return rescaled_diag_grating[:resX, :resY].astype(np.uint8)

def rotate(points, origin = complex(0, 0), angle = 0):
    # rotates complex array points by angle in degree around complex point origin
    rotated = (points - origin) * np.exp(complex(0, np.deg2rad(angle))) + origin
    return rotated.real, rotated.imag

def generate_sawtooth(period_z, angle=0, ycenter = 0, height = resY/2, phaseshift = 0, duty_cycle = 0.5, resX = resX, resY = resY):
    """
    period_z: period of the sawtooth in z direction
    phaseshift: phase shift of the sawtooth in z direction, in units of radians
    ycenter = center of the sawtooth in y direction, in units of pixels
    """
    shape = lambda t : height/2 * (sawtooth( (2 * np.pi * t - resX/2) / period_z - phaseshift, width = duty_cycle) )
    xx, yy = np.meshgrid(np.arange(-resY/2, resY/2), np.arange(-resX/2, resX/2))
    xr, yr = rotate(xx + 1j * yy, angle = angle)
    
    ycenter = ycenter - resX//2
    ythreshold = shape(xr)
    img = np.zeros([resX, resY])
    img[yr-ycenter>ythreshold] = 255
    return img.astype(np.uint8)

def generate_angled_grating(len_1s, len_0s = None, angle = 0, phaseshift = 0, resX = resX, resY = resY, origin = complex(0,0)):
    if not len_0s:
        len_0s = len_1s
        
    xx, yy = np.meshgrid(np.arange(-resY/2, resY/2), np.arange(-resX/2, resX/2))
    xr, yr = rotate(xx + 1j * yy, angle = angle, origin = origin)
    
    img = np.zeros([resX, resY])
    ind_1s = (xr + phaseshift) % (len_0s+len_1s)
    img[ind_1s >= len_0s] = 255
    return img.astype(np.uint8)

def generate_parabolic_lens(thickness, width, angle=0, xoffset = 0, yoffset = 0, resX = resX, resY = resY):
    xx, yy = np.meshgrid(np.arange(-resY/2, resY/2), np.arange(-resX/2, resX/2))
    xx -= xoffset
    yy -= yoffset
    xr, yr = rotate(xx + 1j * yy, angle = angle)
    
    img = np.zeros([resX, resY])
    side1 = 2*thickness / width**2 * yr**2 - thickness/2 < xr
    side2 = 2*thickness / width**2 * yr**2 - thickness/2 < -xr
    img[np.logical_and(side1, side2)] = 255
    return img.astype(np.uint8)

def generate_circle(r, thickness, angle = 0, center = (0,0), rotation_origin = complex(0,0), resX = resX, resY = resY):
    xx, yy = np.meshgrid(np.arange(-resY/2, resY/2) - center[0], 
                         np.arange(-resX/2, resX/2) - center[1])
    xr, yr = rotate(xx + 1j * yy, angle = angle, origin = rotation_origin)
    
    img = np.zeros([resX, resY])
    ind_1s = np.abs(xr**2 + yr**2)
    r_inner = np.abs(r) - thickness / 2
    r_outer = np.abs(r) + thickness / 2
    img[ind_1s < r_outer**2] = 255
    img[ind_1s < r_inner**2] = 0
    return img.astype(np.uint8)

def generate_curved_waveguide(width, radius_of_curvature, entrance_point = 0, entrance_angle = 0, center_y = 0, end_point = 1024, resX = resX, resY = resY):
    center = np.array([-resY/2 + center_y, np.abs(radius_of_curvature) + entrance_point - resX/2])
    rotation_origin = np.array([0, -np.abs(radius_of_curvature)])
    if radius_of_curvature < 0:
        center += np.array([0, 2*radius_of_curvature])
        rotation_origin -= np.array([0, 2*radius_of_curvature])

    return generate_circle(r = radius_of_curvature, 
                          thickness = width, 
                          center=center, 
                          angle = entrance_angle, 
                          rotation_origin=complex(*rotation_origin),
                          resX = resX, 
                          resY = resY)


def generate_beamsteering_edge(angle, entrance_point):
    phaseshift = 386 - entrance_point
    origin = complex(-512,-phaseshift)
    len_1s = 2048
    len_0s = len_1s

    xx, yy = np.meshgrid(np.arange(-resY/2, resY/2), np.arange(-resX/2, resX/2))
    xr, yr = rotate(xx + 1j * yy, angle = angle, origin = origin)

    img = np.zeros([resX, resY])
    ind_1s = (yr + phaseshift) % (len_0s+len_1s)
    img[ind_1s >= len_0s] = 255
    return img.astype(np.uint8)

def array2image(array):
    return Image.fromarray(array.astype(np.uint8))

def save_image(image, path):
    if type(image) == np.ndarray: image = array2image(image)
    assert type(image) == Image.Image, 'Unsupported data type: image needs to be PIL.Image.Image or numpy.ndarray'
    image.save(path)

def add_patterns(p1, p2):
    # adds pattern p1 to p2 so that pixels 
    # 0 + 0 = 0 
    # 0 + 1 = 1 
    # 1 + 0 = 1 
    # 1 + 1 = 1
    return np.uint8(np.clip(np.uint16(p1)+np.uint16(p2), 0, 255))

def substract_patterns(p1, p2):
    # substracts patterns p2 from p1, so that pixels 
    # 0 - 0 = 0
    # 1 - 0 = 1
    # 0 - 1 = 0
    # 1 - 1 = 0
    return np.uint8(np.clip(np.uint16(p1)-np.uint16(p2), 0, 255))

def AND_patterns(p1, p2):
    # combines patterns p1 and p2 such that 
    # 0 + 0 = 0
    # 0 + 1 = 0
    # 1 + 1 = 0
    # 1 + 1 = 1
    
    # check if patterns are binary
    assert np.sum(p1==255) + np.sum(p1==0) == resX * resY, 'p1 not binary'
    assert np.sum(p2==255) + np.sum(p2==0) == resX * resY, 'p2 not binary'
    return np.logical_and(p1 == 255, p2 == 255).astype(np.uint8)*255

def OR_patterns(p1, p2):
    # combines patterns p1 and p2 such that 
    # 0 + 0 = 0
    # 0 + 1 = 1
    # 1 + 1 = 1
    # 1 + 1 = 1
    
    # check if patterns are binary
    assert np.sum(p1==255) + np.sum(p1==0) == resX * resY, 'p1 not binary'
    assert np.sum(p2==255) + np.sum(p2==0) == resX * resY, 'p2 not binary'
    return np.logical_or(p1 == 255, p2 == 255).astype(np.uint8)*255

def invert_pattern(p):
    # 0 -> 1
    # 1 -> 0
    return substract_patterns(generate_all_on(), p)

def generate_checkerboard_pattern(field_size, resX = resX, resY = resY):
    gratingh1 = generate_horizontal_grating(field_size, resX = resX, resY = resY)
    gratingv1 = generate_vertical_grating(field_size, resX = resX, resY = resY)
    gratingh2 = generate_horizontal_grating(field_size, phaseshift=field_size, resX = resX, resY = resY)
    gratingv2 = generate_vertical_grating(field_size, phaseshift=field_size, resX = resX, resY = resY)
    return add_patterns(gratingh1 * gratingv1, gratingh2 * gratingv2)

def generate_fill_factor_mask(fill_factor, scale_factor_y, scale_factor_z, shape):
    fill_factor = 0.5
    len1s = (scale_factor_y * np.sqrt(fill_factor))
    len0s = (scale_factor_y - len1s)
    mask_y = generate_horizontal_grating(len1s, len0s, phaseshift = len0s/2, resX = shape[0], resY = shape[1]) 

    len1s = (scale_factor_z * np.sqrt(fill_factor))
    len0s = (scale_factor_z - len1s)
    mask_z = generate_vertical_grating(len1s, len0s, phaseshift = len0s/2, resX = shape[0], resY = shape[1]) 
    return mask_y * mask_z

def plot_DMD_img(img):
    """
    Plots a DMD image with the correct orientation.
    Corresponds to the orientation with the light coming from the right, exactly how it would look if you looked down at the setup!
    """
    plt.imshow(img, aspect="auto", origin="lower", cmap="Greens", vmin=0)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.grid(alpha=0.4)

def create_grin_beamsteerer_slice(cut_ind, width_one_side, slope_positive=True):
    levels = 2*width_one_side
    inds_list = np.arange(resX)
    if slope_positive:
        i_list = inds_list - cut_ind + width_one_side
        slice = i_list/levels
        slice = 255*np.clip(slice, 0, 1)
    else:
        i_list = -inds_list + cut_ind + width_one_side
        slice = i_list/levels
        slice = 255*np.clip(slice, 0, 1)
    return slice

def generate_grin_beamsteerer(z_inds, cut_ind, width_one_side, slope_positive=True):
    img = generate_all_off()
    slice = create_grin_beamsteerer_slice(cut_ind, width_one_side, slope_positive=slope_positive)
    for ind in z_inds:
        img[:, ind] = slice
    return img

def generate_grin_y_splitter(z_inds, cut_ind, width_one_side):
    img = generate_all_off()
    slice_left = create_grin_beamsteerer_slice(cut_ind-width_one_side, width_one_side, slope_positive=True)
    slice_right = create_grin_beamsteerer_slice(cut_ind+width_one_side, width_one_side, slope_positive=False)
    slice_total = slice_left + slice_right - 255
    for ind in z_inds:
        img[:, ind] = slice_total
    return img