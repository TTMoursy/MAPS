# This is mostly plagiarized from the rest of MAPS which was built by Kyle and Nihan
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
import jaxopt # Can also move to Optimistix if ever preferred - JAXopt is a bit faster in my testing and more intuitive (and more stylish in my opinion)
import numpy as np
import healpy as hp
from . import clebschGordan as CG
from enterprise.signals import anis_coefficients as ac

# This file has the anis_pta class at the top and then the JAX functions at the bottom

class anis_pta():
    """A class to perform anisotropic GW searches using PTA data.

    This class can be used to perform anisotropic GW searches using PTA data by supplying
    it with the outputs of the PTA optimal statistic which can be found in
    (enterprise_extensions.frequentist.optimal_statistic or defiant/optimal_statistic).
    While you can include the OS values upon construction (xi, rho, sig, os),
    you can also use the method set_data() to set these values after construction.

    Attributes:
        psrs_theta (np.ndarray): An array of pulsar position theta [npsr].
        psrs_phi (np.ndarray): An array of pulsar position phi [npsr].
        npsr (int): The number of pulsars in the PTA.
        npair (int): The number of pulsar pairs.
        pair_idx (np.ndarray): An array of pulsar indices for each pair [npair x 2].
        xi (np.ndarray, optional): A list of pulsar pair separations from the OS [npair].
        rho (np.ndarray, optional): A list of pulsar pair correlations [npair].
            NOTE: rho is normalized by the OS value, making this slightly different from 
            what the OS uses. (i.e. OS calculates hat{A}^2 * ORF while this uses ORF).
        sig (np.ndarray, optional): A list of 1-sigma uncertainties on rho [npair].
        os (float, optional): The optimal statistic's best-fit A^2 value.
        pair_cov (np.ndarray, optional): The pair covariance matrix [npair x npair].
        pair_ind_N_inv (np.ndarray): The inverse of the pair independent covariance matrix.
        pair_cov_N_inv (np.ndarray): The inverse of the pair covariance matrix.
        l_max (int): The maximum l value for spherical harmonics.
        nside (int): The nside of the healpix sky pixelization.
        npix (int): The number of pixels in the healpix pixelization.
        blmax (int): The maximum l value for the sqrt power basis.
        clm_size (int): The number of spherical harmonic modes.
        blm_size (int): The number of spherical harmonic modes for the sqrt power basis.
        gw_theta (np.ndarray): An array of source GW theta positions [npix].
        gw_phi (np.ndarray): An array of source GW phi positions [npix].
        use_physical_prior (bool): Whether to use physical priors or not.
        include_pta_monopole (bool): Whether to include the monopole term in the search.
        mode (str): The mode of the spherical harmonic decomposition to use.
            Must be 'power_basis', 'sqrt_power_basis', or 'hybrid'.
        sqrt_basis_helper (CG.clebschGordan): A helper object for the sqrt power basis.
        ndim (int): The number of dimensions for the search.
        F_mat (np.ndarray): The antenna response matrix [npair x npix].
        Gamma_lm (np.ndarray): The spherical harmonic basis [npair x ndim].
    """

    def __init__(self, psrs_theta, psrs_phi, xi = None, rho = None, sig = None, 
                 os = None, pair_cov = None, l_max = 6, nside = 2, 
                 pair_idx = None): # always use physical prior when possible; always use forward modeling when possible; no support yet for monopole
        """Constructor for the anis_pta class.

        This function will construct an instance of the anis_pta class. This class
        can be used to perform anisotropic GW searches using PTA data by supplying
        it with the outputs of the PTA optimal statistic which can be found in 
        (enterprise_extensions.frequentist.optimal_statistic or defiant/optimal_statistic).
        While you can include the OS values upon construction (xi, rho, sig, os),
        you can also use the method set_data() to set these values after construction.
        
        Args:
            psrs_theta (np.ndarray): An array of pulsar position theta [npsr].
            psrs_phi (np.ndarray): An array of pulsar position phi [npsr].
            xi (np.ndarray, optional): A list of pulsar pair separations from the OS [npair].
            rho (np.ndarray, optional): A list of pulsar pair correlations [npair].
            sig (np.ndarray, optional): A list of 1-sigma uncertainties on rho [npair].
            os (float, optional): The optimal statistic's best-fit A^2 value.
            pair_cov (np.ndarray, optional): The pair covariance matrix [npair x npair].
            l_max (int): The maximum l value for spherical harmonics.
            nside (int): The nside of the healpix sky pixelization.
            mode (str): The mode of the spherical harmonic decomposition to use.
                Must be 'power_basis', 'sqrt_power_basis', or 'hybrid'.
            use_physical_prior (bool): Whether to use physical priors or not.
            include_pta_monopole (bool): Whether to include the monopole term in the search.
            pair_idx (np.ndarray, optional): An array of pulsar indices for each pair [npair x 2].

        Raises:
            ValueError: If the lengths of psrs_theta and psrs_phi are not equal.
            ValueError: If the length of pair_idx is not equal to the number of pulsar pairs.            
        """
        # Pulsar positions
        self.psrs_theta = psrs_theta if type(psrs_theta) is np.ndarray else np.array(psrs_theta)
        self.psrs_phi = psrs_phi if type(psrs_phi) is np.ndarray else np.array(psrs_phi)
        if len(psrs_theta) != len(psrs_phi):
            raise ValueError("Pulsar theta and phi arrays must have the same length")
        self.npsr = len(psrs_theta)
        self.npairs = int( (self.npsr * (self.npsr - 1)) / 2)

        # OS values
        if pair_idx is None:
            self.pair_idx = np.array([(a,b) for a in range(self.npsr) for b in range(a+1,self.npsr)])
        else:
            self.pair_idx = pair_idx
        
        if xi is not None:
            if type(xi) is not np.ndarray:
                self.xi = np.array(xi)
            else:
                self.xi = xi
        else:
            self.xi = self._get_xi()

        self.rho, self.sig, self.os, self.pair_cov = None, None, None, None
        self.pair_ind_N_inv, self.pair_cov_N_inv = None, None

        self.set_data(rho, sig, os, pair_cov)
        
        # Check if pair_idx is valid
        if len(self.pair_idx) != self.npairs:
            raise ValueError("pair_idx must have length equal to the number of pulsar pairs")
        
        # Pixel decomposition and Spherical harmonic parameters
        self.l_max = int(l_max)
        self.nside = int(nside)
        self.npix = hp.nside2npix(self.nside)

        # Some configuration for spherical harmonic basis runs
        # clm refers to normal spherical harmonic basis
        # blm refers to sqrt power spherical harmonic basis
        self.blmax = int(self.l_max / 2.)
        self.clm_size = (self.l_max + 1) ** 2
        self.blm_size = hp.Alm.getsize(self.blmax)

        self.gw_theta, self.gw_phi = hp.pix2ang(nside=self.nside, ipix=np.arange(self.npix))
        
        self.sqrt_basis_helper = CG.clebschGordan(l_max = self.l_max)

        self.F_mat = jnp.array(self.antenna_response())

        # The spherical harmonic basis for \Gamma_lm_mat shape (nclm, npsrs, npsrs)
        Gamma_lm_mat = ac.anis_basis(np.dstack((self.psrs_phi, self.psrs_theta))[0], 
                                 lmax = self.l_max, nside = self.nside) # This takes a few seconds and could maybe be vectorized
        
        # We need to reorder Gamma_lm_mat to shape (nclm, npairs)
        self.Gamma_lm = jnp.array(Gamma_lm_mat[:, self.pair_idx[:,0], self.pair_idx[:,1]])

        self._make_cache()
                     
        return None
    
    
    def set_data(self, rho=None, sig=None, os=None, covariance=None):
        """Set the data for the anis_pta object.

        This function allows you to set the data for the anis_pta object 
        after construction. This allows users to use the same anis_pta object
        with different draws of the data. This is especially helpful when combined
        with the noise marginalized optimal statistic or per-frequency optimal statistic 
        analyses. This function will normalize the rho, sig, and covariance by the 
        OS (A^2) value, making self.rho, self.sig, and self.pair_cov represent only 
        the correlations. 
        NOTE: If using pair covariance you still need to supply this function
        with the pairwise uncertainties as well!

        Args:
            rho (list, optional): A list of pulsar pair correlated amplitudes (<rho> = <A^2*ORF>).
            sig (list, optional): A list of 1-sigma uncertaintties on rho.
            os (float, optional): The OS' fit A^2 value.
            covariance (np.ndarray, optional): The pair covariance matrix [npair x npair].
        """
        # Read in OS and normalize cross-correlations by OS. 
        # (i.e. get <rho/OS> = <ORF>)
        self._Lt_pc, self._Lt_nopc = None, None # Reset the cholesky decompositions
        self.pair_cov_N_inv, self.pair_ind_N_inv = None, None
        self._snr_norm = None

        if (rho is not None) and (sig is not None) and (os is not None):
            self.os = jnp.array(os)
            self.rho = jnp.array(rho) / self.os
            self.sig = jnp.array(sig) / self.os
            self.pair_ind_N_inv = _get_N_inv_nopc(self.sig)
            self._Lt_nopc = _get_Lt_nopc(self.sig)

        else:
            self.rho = None
            self.sig = None
            self.os = None

        if covariance is not None:
            self.pair_cov = jnp.array(covariance) / self.os**2
            self.pair_cov_N_inv = _get_N_inv(self.sig, self.pair_cov)
            self._Lt_pc = _get_Lt(self.pair_cov_N_inv)
        else:
            self.pair_cov = None

    def _get_radec(self):
        """Get the pulsar positions in RA and DEC."""
        psr_ra = self.psrs_phi
        psr_dec = (np.pi/2) - self.psrs_theta
        return psr_ra, psr_dec


    def _get_xi(self):
        """Calculate the angular separation between pulsar pairs.

        A function to compute the angular separation between pulsar pairs. This
        function will use a pair_idx array which is assigned upon construction 
        which ensures that the ordering of the pairs is consistent with the OS.

        Returns:
            np.ndarray: An array of pair separations.
        """
        psrs_ra, psrs_dec = self._get_radec()

        x = np.cos(psrs_ra)*np.cos(psrs_dec)
        y = np.sin(psrs_ra)*np.cos(psrs_dec)
        z = np.sin(psrs_dec)
        
        pos_vectors = np.array([x,y,z])

        a,b = self.pair_idx[:,0], self.pair_idx[:,1]

        xi = np.zeros( len(a) )
        # This effectively does a dot product of pulsar position vectors for all pairs a,b
        pos_dot = np.einsum('ij,ij->j', pos_vectors[:,a], pos_vectors[:, b])
        xi = np.arccos( pos_dot )

        return np.squeeze(xi)

    def antenna_response(self):
        """A function to compute the antenna response matrix R_{ab,k}.

        This function computes the antenna response matrix R_{ab,k} where ab 
        represents the pulsar pair made of pulsars a and b, and k represents
        the pixel index. 
        NOTE: This function uses the GW propogation direction for gwtheta and gwphi
        rather than the source direction (i.e. this method uses the vector from the
        source to the observer)

        Returns:
            np.ndarray: An array of shape (npairs, npix) containing the antenna
                pattern response matrix.
        """
        npix = hp.nside2npix(self.nside)
        gwtheta,gwphi = hp.pix2ang(self.nside,np.arange(npix))

        FpFc = ac.signalResponse_fast(self.psrs_theta, self.psrs_phi, gwtheta, gwphi)
        Fp,Fc = FpFc[:,0::2], FpFc[:,1::2] 

        R_abk = Fp[self.pair_idx[:,0]]*Fp[self.pair_idx[:,1]] + Fc[self.pair_idx[:,0]]*Fc[self.pair_idx[:,1]]

        return R_abk
    

    def get_pure_HD(self):
        """Calculate the Hellings and Downs correlation for each pulsar pair.

        This function calculates the Hellings and Downs correlation for each pulsar
        pair. This is done by using the values of xi potentially supplied upon 
        construction. 

        Returns:
            np.ndarray: An array of HD correlation values for each pulsar pair.
        """
        #Return the theoretical HD curve given xi

        xx = (1 - np.cos(self.xi)) / 2.
        hd_curve = 1.5 * xx * np.log(xx) - xx / 4 + 0.5

        return hd_curve 

    def fisher_matrix_sph(self, pair_cov=False):
        """A method to calculate the Fisher matrix for the spherical harmonic basis.

        Args:
            pair_cov (bool): A flag to use the pair covariance matrix if it was supplied

        Raises:
            ValueError: If pair_cov is True and no pair covariance matrix is supplied.

        Returns:
            np.array: The Fisher matrix for the spherical harmonic basis. [n_clm x n_clm]
        """
        F_mat_clm = self.Gamma_lm.T

        if pair_cov and self.pair_cov is None:
            raise ValueError("No pair covariance matrix supplied! Set it with set_data()")
        elif pair_cov:
            fisher_mat = F_mat_clm.T @ self.pair_cov_N_inv @ F_mat_clm
        else:
            fisher_mat = F_mat_clm.T @ self.pair_ind_N_inv @ F_mat_clm 

        return fisher_mat

    def anisotropy_recovery(self, basis, pair_cov):
        if pair_cov: 
            N_inv = self.pair_cov_N_inv
            Lt = self._Lt_pc
        else: 
            N_inv = self.pair_ind_N_inv
            Lt = self._Lt_nopc
            
        if basis == 'sqrt':
            recovery = _sqrt_basis(self.rho, self.Gamma_lm, Lt,
                                   self._bmvals_zero, self._blm_mask, self._blm_vals_float_power, self._bmvals_negative, self._beta_vals,
                                   self._clm_mask, self._mvals_float_power, self._mvals_positive, self._mvals_negative, self._SQRT2,
                                   jnp.array(np.random.rand(2*(self.blmax+1)**2 - 1)), self._b00) # "prior" on blm components is [0,1] (temporarily)
        elif basis == 'pixel':
            recovery = _pixel_basis(self.rho, self.F_mat, Lt, 2*jnp.array(np.random.rand(self.npix))) # "prior" on pixel amplitudes is [0,2] (maybe temporarily)
        elif basis == 'radiometer':
            recovery = _radiometer_basis(self.rho, self.pix_area, N_inv, self.F_mat)
        elif basis == 'spherical':
            recovery = _sph_basis(self.rho, self.Gamma_lm, Lt, jnp.array(np.random.rand(self.clm_size)), self._c00) # "prior" on clms is [0,1] (temporarily)
        elif basis == 'eigenmaps':
            raise NotImplementedError('Basis not implemented yet!')
        else:
            raise ValueError('Basis not recognized!')
        return recovery

    def get_snrs_squared(self, basis_decomposition, basis, pair_cov):
        # basis_decomposition is a pixel map if basis is 'radiometer' or 'pixel'
        # basis_decomposition is a tuple, array, or list with the first element A2 and the rest the clms if basis is 'sqrt' or 'spherical'
        # basis is 'radiometer', 'pixel', 'spherical', or 'sqrt'
        # pair_cov is True or False
        # returns the square of the total snr, iso snr, and anis snr
        
        if basis in ['pixel', 'radiometer']:
            model_orf = _map2orf(basis_decomposition, self.F_mat)
        elif basis in ['sqrt', 'spherical']:
            model_orf = basis_decomposition[0]*_clm2orf(basis_decomposition[1:], self.Gamma_lm)       
        
        if pair_cov: 
            Lt = self._Lt_pc
        else:
            Lt = self._Lt_nopc
        
        iso_orf, state = _iso_fit(self.rho, Lt, self._HD, 2*jnp.array(np.random.rand(1))) # "prior" on A2 is [0,2] (temporarily)
        
        if pair_cov:
            if self._snr_norm is None:
                self._snr_norm = _get_snr_norm(self.sig, self.pair_cov)
            return _orf2snr(model_orf, iso_orf, self.rho, self.pair_cov_N_inv, self.pair_ind_N_inv, self._snr_norm)
        else:
            return _orf2snr_nopc(model_orf, iso_orf, self.rho, self.pair_ind_N_inv)

    def _make_cache(self):
        self._SQRT2 = jnp.sqrt(2)
        self._HD = jnp.array(self.get_pure_HD())
        self._c00 = jnp.array([jnp.sqrt(4*jnp.pi)])
        self._b00 = jnp.array([1])
        self.pix_area = hp.nside2pixarea(self.nside)
        # Sqrt basis cache
        self._beta_vals = jnp.array(self.sqrt_basis_helper.beta_vals)
        self._blvals = jnp.array(self.sqrt_basis_helper.bl_idx)
        self._bmvals = jnp.array(self.sqrt_basis_helper.bm_idx)
        self._abs_bmvals = jnp.abs(self._bmvals)
        self._lvals = jnp.concatenate([jnp.repeat(jnp.arange(0, self.l_max+1), jnp.array([2*ll + 1 for ll in range(0, self.l_max+1)]))])
        self._mvals = jnp.concatenate([jnp.arange(-ll, ll+1) for ll in range(0, self.l_max+1)])
        self._abs_mvals = jnp.abs(self._mvals)
        self._clm_mask = self._abs_mvals * (2 * self.l_max + 1 - self._abs_mvals) // 2 + self._lvals
        self._mvals_positive = self._mvals > 0
        self._mvals_negative = self._mvals < 0
        self._mvals_float_power = jnp.float_power(-1, self._mvals)
        self._blm_mask = self._abs_bmvals * (2*self.blmax+1 - self._abs_bmvals) // 2 + self._blvals
        self._blm_vals_float_power = jnp.float_power(-1, self._bmvals)
        self._bmvals_negative = self._bmvals < 0
        self._bmvals_zero = self._bmvals == 0

# JAX functions - anisotropy bases at the top and utils at the bottom

# Anisotropy bases

@jax.jit
def _radiometer_basis(rho, pix_area, N_inv, F_mat):
    fisher_matrix_radiometer = jnp.diag( F_mat.T @ N_inv @ F_mat )
    dirty_map = F_mat.T @ N_inv @ rho
    fisher_diag_inv = jnp.diag( 1/fisher_matrix_radiometer )
    radio_map = fisher_diag_inv @ dirty_map
    norm = 4 * jnp.pi / trapezoid(radio_map, dx = pix_area)
    radio_map_n = radio_map * norm
    radio_map_err = jnp.sqrt(jnp.diag(fisher_diag_inv)) * norm
    return radio_map_n, radio_map_err

@jax.jit
def _pixel_basis(rho, F_mat, Lt, params):
    def residuals(params):
        power_map = jnp.sqrt(params**2 + 1) - 1 # Apply non-negativity constraint using LMFit's convention, which uses the MINUIT convention
        model_orf = F_mat @ power_map
        r = rho - model_orf
        return Lt @ r
    opt_params, state = jax.jit(jaxopt.LevenbergMarquardt(residuals, materialize_jac=True, jit=True).run)(params)
    opt_power_map = jnp.sqrt(opt_params**2 + 1) - 1
    return opt_power_map, state

@jax.jit
def _sph_basis(rho, Gamma_lm, Lt, params, c00):
    def residuals(params):
        A2 = 10**(params[0])
        clm = jnp.concatenate( (c00,params[1:]) )
        orf = clm @ Gamma_lm
        model_orf = A2*orf
        r = rho - model_orf
        return Lt @ r
    opt_params, state = jax.jit(jaxopt.LevenbergMarquardt(residuals, materialize_jac=True, jit=True).run)(params)
    opt_A2 = 10**(opt_params[0])
    opt_clm = jnp.concatenate( (c00,opt_params[1:]) )
    return opt_A2, opt_clm, state

# almost unreadable from how much caching I'm using
@jax.jit
def _sqrt_basis(rho, Gamma_lm, Lt,
                cache_bmvals_zero, cache_blm_mask, cache_blm_vals_float_power, cache_bmvals_negative, cache_beta_vals,
                cache_clm_mask, cache_m_vals_float_power, cache_m_vals_positive, cache_m_vals_negative, SQRT2,
                params, b00):
    def residuals(params):
        A2 = 10**(2*jnp.sin(params[0]))
        blm = jnp.concatenate( (b00, params[1::2]+1j*params[2::2]) )
        real_blm = jnp.real(blm)
        blm = jnp.where(cache_bmvals_zero, real_blm, blm)
        alm = _blm2alm(blm, cache_beta_vals,
                       cache_blm_mask, cache_blm_vals_float_power, cache_bmvals_negative)
        clm = _alm2clm(alm,
                       cache_clm_mask, cache_m_vals_float_power, cache_m_vals_positive, cache_m_vals_negative, SQRT2)
        orf = clm @ Gamma_lm
        model_orf = A2*orf
        r = rho - model_orf
        return Lt @ r
    opt_params, state = jax.jit(jaxopt.LevenbergMarquardt(residuals, materialize_jac=True, jit=True).run)(params)
    opt_A2 = 10**(2*jnp.sin(opt_params[0]))
    opt_blm = jnp.concatenate( (b00, opt_params[1::2]+1j*opt_params[2::2]) )
    real_opt_blm = jnp.real(opt_blm)
    opt_blm = jnp.where(cache_bmvals_zero, real_opt_blm, opt_blm)
    opt_alm = _blm2alm(opt_blm, cache_beta_vals,
                       cache_blm_mask, cache_blm_vals_float_power, cache_bmvals_negative)
    opt_clm = _alm2clm(opt_alm,
                       cache_clm_mask, cache_m_vals_float_power, cache_m_vals_positive, cache_m_vals_negative, SQRT2)
    return opt_A2, opt_clm, state

@jax.jit
def _iso_fit(rho, Lt, HD_curve, A2):
    def residuals(A2):
        model_orf = (jnp.sqrt(A2**2 + 1) - 1) * HD_curve # following LMFit for lower-bounded parameters; LMFit in turn follows MINUIT convention
        r = rho - model_orf
        return Lt @ r
    opt_A2, state = jax.jit(jaxopt.LevenbergMarquardt(residuals, materialize_jac=True, jit=True).run)(A2)
    return (jnp.sqrt(opt_A2**2 + 1) - 1)*HD_curve, state

@jax.jit
def _map2orf(pixel_map, F_mat):
        return F_mat @ pixel_map

@jax.jit
def _clm2orf(clm, Gamma_lm):
        return clm @ Gamma_lm

@jax.jit
def _orf2snr_nopc(ani_orf, iso_orf, rho, N_inv_nopc):
    ani_res = rho - ani_orf
    iso_res = rho - iso_orf
    snm = (-1/2)*((ani_res).T @ N_inv_nopc @ (ani_res)) # Anisotropy chi-square
    hdnm = (-1/2)*((iso_res).T @ N_inv_nopc @ (iso_res)) # Isotropy chi-square
    nm = (-1/2)*((rho).T @ N_inv_nopc @ (rho)) # Null chi-square (Not pair covariant)
    total_sn = 2 * (snm - nm)
    iso_sn = 2 * (hdnm - nm)
    anis_sn = 2 * (snm - hdnm)
    return total_sn, iso_sn, anis_sn

@jax.jit
def _get_snr_norm(sig, pair_cov_matrix):
    det_sig = jnp.linalg.slogdet(jnp.diag(sig))[1]
    det_paircov = jnp.linalg.slogdet(pair_cov_matrix)[1]
    return 0.5*(det_sig - det_paircov)

@jax.jit
def _orf2snr(ani_orf, iso_orf, rho, N_inv, N_inv_nopc, snr_norm):
    ani_res = rho - ani_orf
    iso_res = rho - iso_orf
    snm = (-1/2)*((ani_res).T @ N_inv @ (ani_res)) # Anisotropy chi-square
    hdnm = (-1/2)*((iso_res).T @ N_inv @ (iso_res)) # Isotropy chi-square
    nm = (-1/2)*((rho).T @ N_inv_nopc @ (rho)) # Null chi-square (Not pair covariant)
    total_sn = 2 * (snm - nm + snr_norm)
    iso_sn = 2 * (hdnm - nm + snr_norm)
    anis_sn = 2 * (snm - hdnm)
    return total_sn, iso_sn, anis_sn

# utils functions
@jax.jit
def _alm2clm(alm,
             cache_clm_mask, cache_m_vals_float_power, cache_m_vals_positive, cache_m_vals_negative, SQRT2):
    clm = alm[cache_clm_mask]
    cache_positive_m = cache_m_vals_float_power * jnp.real(clm) * SQRT2
    cache_negative_m = cache_m_vals_float_power * jnp.imag(clm) * SQRT2
    clm = jnp.where(cache_m_vals_positive, cache_positive_m, clm)
    clm = jnp.where(cache_m_vals_negative, cache_negative_m, clm)
    clm = jnp.real(clm)
    return clm

@jax.jit
def _blm2alm(blms_in, beta_vals,
             cache_blm_mask, cache_blm_vals_float_power, cache_bmvals_negative):
    blm_full = blms_in[cache_blm_mask]
    cache_blm_full = cache_blm_vals_float_power*jnp.conj(blm_full)
    blm_full = jnp.where(cache_bmvals_negative, cache_blm_full, blm_full)
    alm_vals = jnp.einsum('ijk,j,k', beta_vals, blm_full, blm_full)
    return alm_vals

@jax.jit
def _get_N_inv_nopc(sig):
    return jnp.diag(1/sig ** 2)

@jax.jit
def _get_N_inv(sig, C):
    A = jnp.diag(sig ** 2)
    K = C - A
    In = jnp.eye(A.shape[0])
    return _woodbury_inverse(A, In, In, K)

@jax.jit
def _woodbury_inverse(A, U, C, V):
    Ainv = jnp.diag( 1/jnp.diag(A) )
    Cinv = jnp.linalg.pinv(C)
    CVAU = Cinv + V @ Ainv @ U
    tot_inv = Ainv - Ainv @ U @ jnp.linalg.solve(CVAU, V @ Ainv)
    return tot_inv

@jax.jit
def _get_Lt(N_inv):
    return jnp.linalg.cholesky(N_inv).T

@jax.jit
def _get_Lt_nopc(sig):
    return jnp.diag(1/sig)

@jax.jit
def _get_fisher_radiometer(F_mat, N_inv):
    return jnp.diag( F_mat.T @ N_inv @ F_mat )
