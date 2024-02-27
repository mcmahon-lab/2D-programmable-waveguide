import numpy.fft as fft

def ft_t_axis(N, dt):
    return fft.fftshift(fft.fftfreq(N))*N*dt
    
def ft_f_axis(N, dt):
    return fft.fftshift(fft.fftfreq(N))/dt

### 1D fourier transform functions ###

def fft_centered(y):
    y = fft.ifftshift(y)
    y = fft.fft(y)
    y = fft.fftshift(y)
    return y

def ifft_centered(Y):
    Y = fft.ifftshift(Y)
    Y = fft.ifft(Y)
    Y = fft.fftshift(Y)
    return Y

def fft_iso(y, dt):
    """
    The fft that is isometric, aka it preserves the integral
    """
    return fft.fft(y)*dt

def ifft_iso(Y, dt):
    """
    The ifft that is isometric, aka it preserves the integral
    """
    return fft.ifft(Y)/dt

def fft_centered_iso(y, dt):
    y = fft.ifftshift(y)
    y = fft_iso(y, dt)
    y = fft.fftshift(y)
    return y

def ifft_centered_iso(Y, dt):
    Y = fft.ifftshift(Y)
    Y = ifft_iso(Y, dt)
    Y = fft.fftshift(Y)
    return Y

def fft_centered_ortho(y):
    """
    The fft that is orthogonal/unitary, i.e. preserves the l2 norm of y.
    """
    y = fft.ifftshift(y)
    y = fft.fft(y, norm = 'ortho')
    y = fft.fftshift(y)
    return y

def ifft_centered_ortho(y):
    """
    The ifft that is orthogonal/unitary, i.e. preserves the l2 norm of y.
    """
    y = fft.ifftshift(y)
    y = fft.ifft(y, norm = 'ortho')
    y = fft.fftshift(y)
    return y

### 2D fourier transform functions ###

def fft2_centered(y):
    y = fft.ifftshift(y)
    y = fft.fft2(y)
    y = fft.fftshift(y)
    return y

def ifft2_centered(Y):
    Y = fft.ifftshift(Y)
    Y = fft.ifft2(Y)
    Y = fft.fftshift(Y)
    return Y

def fft2_iso(y, dx, dy):
    """
    The fft2 that is isometric, aka it preserves the integral
    """
    return fft.fft2(y)*dx*dy

def ifft2_iso(Y, dx, dy):
    """
    The ifft2 that is isometric, aka it preserves the integral
    """
    return fft.ifft2(Y)/(dx*dy)

def fft2_centered_iso(y, dx, dy):
    y = fft.ifftshift(y)
    y = fft2_iso(y, dx, dy)
    y = fft.fftshift(y)
    return y

def ifft2_centered_iso(Y, dx, dy):
    Y = fft.ifftshift(Y)
    Y = ifft2_iso(Y, dx, dy)
    Y = fft.fftshift(Y)
    return Y

def fft2_centered_ortho(y):
    """
    The fft that is orthogonal/unitary, i.e. preserves the l2 norm of y.
    """
    y = fft.ifftshift(y)
    y = fft.fft2(y, norm = 'ortho')
    y = fft.fftshift(y)
    return y

def ifft2_centered_ortho(y):
    """
    The ifft that is orthogonal/unitary, i.e. preserves the l2 norm of y.
    """
    y = fft.ifftshift(y)
    y = fft.ifft2(y, norm = 'ortho')
    y = fft.fftshift(y)
    return y