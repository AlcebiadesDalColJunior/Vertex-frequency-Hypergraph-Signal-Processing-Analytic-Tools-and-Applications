import numpy as np

import scipy.optimize as opt

from math import pi, cos, sqrt


#%% Spectral wavelet

def spectral(num_filters, lmax, lpfactor=20, a=2, b=2, t1=1, t2=2):
    lmin = lmax/lpfactor
    t = log_scales(lmin,lmax,num_filters-1)
    
    gb = lambda x: kernel_abspline3(x, a, b, t1, t2)
    g = [None]*(num_filters)
    for j in range(num_filters-1):
        g[j+1] = lambda x, t=t[j]: gb(t * x)
        
    f = lambda x: - gb(x)
    xstar = opt.fminbound(f, 1, 2)

    gamma_l = - f(xstar)
    lminfac = 0.6 * lmin
    
    gl = lambda x: np.exp(- x ** 4)
    g[0] = lambda x: gamma_l * gl(x / lminfac)
    return g

def log_scales(lmin,lmax,nScales,t1=1,t2=2):
  smin = t1/lmax
  smax = t2/lmin
  return np.exp(np.linspace(np.log(smax),np.log(smin),nScales))
  
def kernel_abspline3(x, alpha, beta, t1, t2):
    x = np.array(x)
    r = np.zeros(x.shape) 
    a = np.array([-5, 11, -6, 1])
    
    r1 = (x>=0) & (x<t1)
    r2 = (x>=t1) & (x<t2)
    r3 = x>=t2
    
    r[r1] = x[r1]**alpha * t1**(-alpha)
    r[r2] = a[0] + a[1]*x[r2] + a[2]*x[r2]**2 + a[3]*x[r2]**3
    r[r3] = x[r3]**(-beta) * t2**(beta)
    
    return r

#%% Uniform translates

def uniform_translates(num_filters, upper_bound_translates):
    
    dilation_factor = upper_bound_translates*(3/(num_filters-2))
    main_window = lambda x: .5+.5*cos(2*pi*(x/dilation_factor-1/2)) if (x>=0) and (x<=dilation_factor) else 0#.*(x>=0).*(x<=dilation_factor)
    
    filters = [None]*(num_filters)
    for j in range(num_filters):
        filters[j] = lambda x, t=j: main_window(x-dilation_factor/3*(t+1-3))
    
    return(filters)

#%% Spectrum adapted

def spectrum_adapted(num_filters, lmax, approx_spectrum):
    
    warp_function = lambda s: mono_cubic_warp_fn(approx_spectrum['x'], approx_spectrum['y'], [s])
    upper_bound_translates = max(approx_spectrum['y'])
    uniform_filters = uniform_translates(num_filters, upper_bound_translates)
    
    filters = [None]*(num_filters)
    for j in range(num_filters):
        filters[j] = lambda x, j=j: uniform_filters[j](warp_function(x))
    
    return(filters)
    
def mono_cubic_warp_fn(x,y,x0):
    
    cut = 1e-4
    num_pts = len(x)
    
    # 1. Compute slopes of secant lines
    Delta = np.true_divide(y[1:]-y[0:num_pts-1],x[1:]-x[0:num_pts-1])
    
    # 2. Initialize tangents m at every data point
    m = (Delta[0:num_pts-2]+Delta[1:num_pts-1])/2
    m = np.concatenate((np.array([Delta[0]]),m,np.array([Delta[-1]])))    
    
    # 3. Check for equal y's to set slopes equal to zero
    for k in range(num_pts-1):
        if Delta[k] == 0:
            m[k]   = 0
            m[k+1] = 0

    # 4. Initialize alpha and beta
    alpha = m[0:num_pts-1]/Delta
    beta = m[1:num_pts]/Delta 
    
    # 5. Make monotonic
    for k in range(num_pts-1):
        if alpha[k]**2+beta[k]**2 > 9:
            tau = 3/float(sqrt(alpha[k]**2+beta[k]**2))
            m[k]   = tau*alpha[k]*Delta[k]
            m[k+1] = tau*beta[k]*Delta[k]
    
    # 6. Cubic interpolation
    num_pts_to_interpolate = len(x0)
    interpolated_values = np.zeros((num_pts_to_interpolate,1))
    
    for i in range(num_pts_to_interpolate):
        closest_ind = np.argmin(np.abs(x-x0[i]))
        
        if (x[closest_ind]-x0[i])<(-cut) or (abs(x[closest_ind]-x0[i])<cut and closest_ind < num_pts-1):
            lower_ind = closest_ind
        else:
            lower_ind = closest_ind-1

        h = x[lower_ind+1] - x[lower_ind]
        t = (x0[i]-x[lower_ind])/float(h)
          
        interpolated_values[i] = y[lower_ind]*(2*t**3-3*t**2+1) + h*m[lower_ind]*(t**3-2*t**2+t) + y[lower_ind+1]*(-2*t**3+3*t**2) + h*m[lower_ind+1]*(t**3-t**2)
    
    return(interpolated_values)


#%% Spectrum approximation

from scipy.sparse import csc_matrix
from scipy.sparse import identity

from cvxopt import cholmod, matrix, spmatrix
from cvxopt.cholmod import options

# Compute an approximation of the cumulative density function of the Laplacian eigenvalues
def spectrum_cdf_approx(n, lmax, A, num_pts=8):
    
    counts = np.zeros((num_pts,))
    counts[-1] = n - 1
    
    interp_x = np.arange(num_pts)*lmax/(num_pts-1)
    
    I = identity(n)
    
    A = csc_matrix(A)
    #A = csc_matrix(G['L'])
    
    #Ap = A.indptr
    #Ai = A.indices
    
    #Acoo = A.tocoo()
    #As = spmatrix(Acoo.data, Acoo.row.tolist(), Acoo.col.tolist())
    
    #P = adapted_amd(n, A)
    
    #Lp,Parent,Pinv = ldl_symbolic(n, Ap, Ai, P)
    
    options['supernodal'] = 0
    
    for i in range(1,num_pts-1):
        
        shift_matrix = csc_matrix(interp_x[i]*I)
        
        mat = A - shift_matrix
        
        #mat_Ap = mat.indptr
        #mat_Ai = mat.indices
        #mat_Ax = mat.data
        
        #D = ldl_numeric(n, mat_Ap, mat_Ai, mat_Ax, Lp, Parent, P, Pinv)
        
        #D2 = np.linalg.eigvalsh(mat.toarray())
        
        #D3 = np.linalg.eigvalsh(A.toarray())
        
        Acoo = mat.tocoo()
        mats = spmatrix(Acoo.data, Acoo.row.tolist(), Acoo.col.tolist())
        F = cholmod.symbolic(mats)
        cholmod.numeric(mats,F)
        
        Di = matrix(1.0, (n,1))
        cholmod.solve(F, Di, sys=6)
        
        D = np.zeros((n,))
        for ii in range(n):
            D[ii] = Di[ii]
        
        #print 'D: ', np.sum(D<0), 'D2: ', np.sum(D2<0), 'D3: ', np.sum(D3<interp_x[i])
        
        counts[i] = np.sum(D<0)
        #counts[i] = np.sum(D3<interp_x[i])
    
    interp_y = counts/(n-1)
    
    approx_spectrum = dict()
    approx_spectrum['x'] = interp_x
    approx_spectrum['y'] = interp_y
    
    return(approx_spectrum)




