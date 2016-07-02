

def Hessian2D(I, Sigma=1):
    """
    This function Hessian2 filters the image with 2nd derivatives of a
    Gaussian with parameter Sigma.
    :param I: image, in flotaing point precision (float64)
    :param Sigma: sigma of the gaussian kernel used
    :return: the 2nd derivatives
    """
    # Make kernel coordinates
    X, Y = np.meshgrid(np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1),
                       np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), indexing='ij')

    # Build the gaussian 2nd derivatives filters
    DGaussxx = 1/(2*np.pi*Sigma**4)*(X**2/Sigma**2 - 1)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
    DGaussxy = (1/(2*np.pi*Sigma**6))*(X*Y)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
    DGaussyy = DGaussxx.conj().T

    Dxx = convolve(I, DGaussxx, mode='constant', cval=0.0)
    Dxy = convolve(I, DGaussxy, mode='constant', cval=0.0)
    Dyy = convolve(I, DGaussyy, mode='constant', cval=0.0)

    return Dxx, Dxy, Dyy
