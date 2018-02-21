#!/usr/bin/env python

"""
Module for handling vorticity tendency diffusion in a barotropic model.
"""

import numpy as np

#====================================================================================
#==== N. Weber's new hyperdiffusion scheme ==========================================
#====================================================================================

def del4_filter(zeta, lats, lons, k=2.338e16):
    """
    Calculates/returns the diffusion term of the barotropic vorticity equation,
    which is subtracted from the overall vorticity tendency.
    
    Diffusion is calculated with the following formula:
    D = k * del^4(vorticity)
    (k value taken from Sardeshmukh and Hoskins 1988)
    
    Requires:
    data --> 2D numpy array of data (e.g., global vorticity field)
             shape(data) = (nlats, nlons)
    lats --> 1D array/list of the corresponding latitudes (in degrees)
    lons --> 1D array/list of the corresponding longitudes (in degrees)
    k -----> diffusion coefficient (float)
    
    Returns:
    2-dimensional numpy array (same shape as <data>) 
    """
    del4vort = del4(zeta, lats, lons)
    return k * del4vort


def del4(data, lats, lons):
    """
    Computes the del^4 operator on 2D global data in x-y space.
    Uses the given lats/lons to compute spatial derivatives in meters.
    
    Requires:
    data --> 2D numpy array of data (e.g., global vorticity field)
             shape(data) = (nlats, nlons)
    lats --> 1D array/list of the corresponding latitudes (in degrees)
    lons --> 1D array/list of the corresponding longitudes (in degrees)
    
    Returns:
    2-dimensional numpy array (same shape as <data>) 
    """
    # Get the data resolution (in degrees) -- assumes uniform lat/lon spacing
    res = lons[1] - lons[0]
    _, la = np.meshgrid(lons * np.pi / 180., lats * np.pi / 180.)

    # Use <res> to calculate the distance (in meters) between each point
    dy = 111000. * res
    dx = np.cos(la[:,1:]) * 111000. * res
    
    # Calculate 2nd and 4th derivatives in x and y directions
    d2data_dy2 = second_derivative(data, dy, axis=0)
    d2data_dx2 = second_derivative(data, dx, axis=1)
    d4data_dy4 = fourth_derivative(data, dy, axis=0)
    d4data_dx4 = fourth_derivative(data, dx, axis=1)

    # Use the above to calculate/return del^4(data)
    return d4data_dy4 + d4data_dx4 + (2 * d2data_dy2 * d2data_dx2)


def second_derivative(data, delta, axis=0):
    """
    Computes the second derivative of Nd-array <data> along the desired axis.
    
    Requires:
    data ---> N-dimensional numpy array
    delta --> float or 1-dimensional array/list (same length as desired <data> axis) 
              indicating the distance between data points
    axis ---> desired axis to take the derivative along
    
    Returns:
    N-dimensional numpy array (same shape as <data>) 
    """
    n = len(data.shape)

    # If <delta> is not a list/array (i.e., if the mesh is uniform), 
    # create a list of deltas that is the same length as the desired axis
    deltashape = list(data.shape)
    deltashape[axis] -= 1
    if type(delta) in [float, int, np.float64]:
        delta = np.ones(deltashape) * delta
    elif type(delta) in [list, np.ndarray]:
        shp = list(data.shape)
        shp[axis] -= 1
        if np.shape(delta) != tuple(shp):
            print(np.shape(delta), tuple(shp))
            raise ValueError('input <delta> should has invalid shape')
    else:
        raise ValueError('input <delta> should be value or array')

    # create slice objects --- initially all are [:, :, ..., :]
    slice0 = [slice(None)] * n
    slice1 = [slice(None)] * n
    slice2 = [slice(None)] * n
    delta_slice0 = [slice(None)] * n
    delta_slice1 = [slice(None)] * n

    # First handle centered case
    slice0[axis] = slice(None, -2)
    slice1[axis] = slice(1, -1)
    slice2[axis] = slice(2, None)
    delta_slice0[axis] = slice(None, -1)
    delta_slice1[axis] = slice(1, None)
    combined_delta = delta[delta_slice0] + delta[delta_slice1]
    center = 2 * (data[slice0] / (combined_delta * delta[delta_slice0]) -
                  data[slice1] / (delta[delta_slice0] * delta[delta_slice1]) +
                  data[slice2] / (combined_delta * delta[delta_slice1]))
    
    # Fill the left boundary (pad it with the edge value)
    slice0[axis] = slice(None,1)
    left = center[slice0].repeat(1, axis=axis)

    # Fill the right boundary (pad it with the edge value)
    slice0[axis] = slice(-1, None)
    right = center[slice0].repeat(1, axis=axis)

    return np.concatenate((left, center, right), axis=axis)




def fourth_derivative(data, delta, axis=0):
    """
    Computes the fourth derivative of Nd-array <data> along the desired axis.
    
    Requires:
    data ---> N-dimensional numpy array
    delta --> float or 1-dimensional array/list (same length as desired <data> axis) 
              indicating the distance between data points
    axis ---> desired axis to take the derivative along
    
    Returns:
    N-dimensional numpy array (same shape as <data>) 
    """
    n = len(data.shape)

    # If <delta> is not a list (i.e., if the mesh is uniform), create a list
    # of deltas that is the same length as the desired axis
    deltashape = list(data.shape)
    deltashape[axis] -= 1
    if type(delta) in [float, int, np.float64]:
        delta = np.ones(deltashape) * delta
    elif type(delta) in [list, np.ndarray]:
        shp = list(data.shape)
        shp[axis] -= 1
        if np.shape(delta) != tuple(shp):
            print(np.shape(delta), tuple(shp))
            raise ValueError('input <delta> should has invalid shape')
            raise ValueError('input <delta> should has invalid shape')
    else:
        raise ValueError('input <delta> should be value or array')


    # create slice objects --- initially all are [:, :, ..., :]
    slice0 = [slice(None)] * n
    slice1 = [slice(None)] * n
    slice2 = [slice(None)] * n
    slice3 = [slice(None)] * n
    slice4 = [slice(None)] * n
    delta_slice0 = [slice(None)] * n
    delta_slice1 = [slice(None)] * n
    delta_slice2 = [slice(None)] * n
    delta_slice3 = [slice(None)] * n


    # First handle centered case
    slice0[axis] = slice(None, -4)
    slice1[axis] = slice(1, -3)
    slice2[axis] = slice(2, -2)
    slice3[axis] = slice(3, -1)
    slice4[axis] = slice(4, None)
    delta_slice0[axis] = slice(None, -3)
    delta_slice1[axis] = slice(1, -2)
    delta_slice2[axis] = slice(2, -1)
    delta_slice3[axis] = slice(3, None)
    center = f4(data[slice0], data[slice1], data[slice2], data[slice3], data[slice4],
                delta[delta_slice0], delta[delta_slice1], delta[delta_slice2], delta[delta_slice3])

    # Fill the left boundary (pad it with the edge value)
    slice0[axis] = slice(None,1)
    left = center[slice0].repeat(2, axis=axis)

    # Fill the right boundary (pad it with the edge value)
    slice0[axis] = slice(-1, None)
    right = center[slice0].repeat(2, axis=axis)

    return np.concatenate((left, center, right), axis=axis)




def f4(f0, f1, f2, f3, f4, d0, d1, d2, d3):
    """
    Computes the fourth derivative with the approximation:
    f^4(x) = [f(x-2) - 4*f(x-1) + 6*f(x) - 4*f(x+1) + f(x+2)] / hx^4
    (modified for non-uniform grid spacing)
    
    Requires:
    f0,f1,f2,f3,f4 -> values at the staggered locations (numerator)
    d0,d1,d2,d3 ----> distances (deltas) between the staggered locations (denominator)
    """
    d01 = d0 + d1
    d02 = d0 + d1 + d2
    d03 = d0 + d1 + d2 + d3
    d12 = d1 + d2
    d13 = d1 + d2 + d3
    d23 = d2 + d3
    return 24 * (f0/(d0*d01*d02*d03) - f1/(d0*d1*d12*d13) + f2/(d01*d1*d2*d23) -
                 f3/(d02*d12*d2*d3) + f4/(d03*d13*d23*d3))



#====================================================================================
#==== L. Madaus's original hyperdiffusion scheme ====================================
#====================================================================================

def apply_des_filter(s, cur_vort, vort_tend, ntrunc, t=0):
    """ Add spectral hyperdiffusion and return a new
    vort_tend """
    # Convert to spectral grids
    print('vorticity:', np.shape(cur_vort))
    vort_spec = s.grdtospec(cur_vort)
    print('spectral vorticity:', np.shape(vort_spec))
    vort_tend_spec = s.grdtospec(vort_tend)
    total_length = vort_spec.shape[0]

    # Reshape to 2-d array
    vort_spec = np.reshape(vort_spec,(ntrunc,-1))
    print('reshaped:', np.shape(vort_spec))
    vort_tend_spec = np.reshape(vort_tend_spec,(ntrunc,-1))
    new_vort_tend_spec = np.array(vort_tend_spec,dtype=np.complex)

    DES = compute_dampening_eddy_sponge(vort_tend_spec.shape)

    num = vort_tend_spec - DES * vort_spec
    den = np.ones(np.shape(DES), dtype=np.complex) + DES * np.complex(NL.dt,0.)
    new_vort_tend_spec[:,:] = num / den
    

    # Reshape the new vorticity tendency and convert back to grid
    new_vort_tend_spec = np.reshape(new_vort_tend_spec, (total_length,-1))

    new_vort_tend = s.spectogrd(new_vort_tend_spec)

    return new_vort_tend


def compute_dampening_eddy_sponge(fieldshape):
    """ Computes the eddy sponge by getting the eigenvalues 
    of the Laplacian for each spectral coefficient and 
    multiplying them by a dampening factor nu 
    (specified in the namelist) 
    From Held and Suarez
    """
    
    # Need some arrays
    m_vals = np.arange(fieldshape[0])
    n_vals = np.arange(fieldshape[1])

    spherical_wave = np.zeros(fieldshape)

    fourier_wave = m_vals * NL.fourier_inc
    for n in n_vals:
        spherical_wave[:,n] = fourier_wave + n

    # Now for the laplacian
    eigen_laplacian = np.divide(np.multiply(spherical_wave,np.add(spherical_wave,1.)),NL.Re**2)
    
    # Dampening Eddy Sponge values
    DES = np.multiply(eigen_laplacian, NL.nu) 
    DES_cpx = np.array(DES, dtype=np.complex)

    return DES_cpx