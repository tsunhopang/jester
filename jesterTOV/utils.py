# this script contain utility functions
import jax
from jax import vmap
import jax.numpy as jnp
from functools import partial
from jaxtyping import Array, Float
from interpax import interp1d as interpax_interp1d
from unxt import Quantity

jax.config.update("jax_enable_x64", True)

#################################
### CONSTANTS AND CONVERSIONS ###
#################################

# to avoid additional dependecy on scipy
# in conventional unit system
c = Quantity(299792458.0, 'm s-1')
G = Quantity(6.6743e-11, 'kg-1 m3 s-2')
Msun = Quantity(1.988409870698051e30, 'kg')
hbar = Quantity(197.3269804593025, 'MeV fm')
# Used under c = 1
hbarc = hbar
m_p = Quantity(938.2720881604904, 'MeV')
m_n = Quantity(939.5654205203889, 'MeV')
m_e = Quantity(0.510998, 'MeV')
m = (m_p + m_n) / 2.0  # Average nucleonic mass defined by Margueron et al

# convension factor from conventional unit to geometric unit
number_density_to_geometric = 1
pressure_to_geometric = G / c**4
energy_density_to_geometric = pressure_to_geometric
mass_to_geometric = G / c**2

# specific values
soloar_mass_in_length = Msun * G / c / c 

#########################
### UTILITY FUNCTIONS ###
#########################

# vmapped jnp.roots function
roots_vmap = vmap(partial(jnp.roots, strip_zeros=False), in_axes=0, out_axes=0)


@vmap
def cubic_root_for_proton_fraction(coefficients):

    a, b, c, d = coefficients
    
    f = ((3.0 * c / a) - ((b ** 2) / (a ** 2))) / 3.0
    g = (
        ((2.0 * (b ** 3)) / (a ** 3)) - ((9.0 * b * c) / (a ** 2)) + (27.0 * d / a)
    ) / 27.0
    g_squared = g ** 2
    f_cubed = f ** 3
    h = g_squared / 4.0 + f_cubed / 27.0

    R = -(g / 2.0) + jnp.sqrt(h)
    S = jnp.cbrt(R)
    T = -(g / 2.0) - jnp.sqrt(h)
    U = jnp.cbrt(T)

    x1 = (S + U) - (b / (3.0 * a))
    x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * jnp.sqrt(3.0) * 0.5j
    x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * jnp.sqrt(3.0) * 0.5j

    return jnp.array([x1, x2, x3])


def cumtrapz(y, x):
    """
    Cumulatively integrate y(x) using the composite trapezoidal rule.

    Parameters
    ----------
    y : jax.numpy.ndarray
        Values to integrate.
    x : jax.numpy.ndarray
        The coordinate to integrate along.

    Returns
    -------
    res : jax.numpy.ndarray
        The result of cumulative integration of `y` along `x`.
    """
    # check the shape of y and x
    assert y.shape == x.shape, "Not matching shape between y and x"
    assert len(y.shape) == 1, "y is expected to be one-dimensional array"
    assert len(x.shape) == 1, "x is expected to be one-dimensional array"

    # get the step size of x
    dx = jnp.diff(x)
    res = jnp.cumsum(dx * (y[1::] + y[:-1:]) / 2.0)
    res = jnp.concatenate(
        (
            jnp.array(
                [
                    1e-30,
                ]
            ),
            res,
        )
    )

    return res


def interp_in_logspace(x, xs, ys):
    logx = jnp.log(x)
    logxs = jnp.log(xs)
    logys = jnp.log(ys)
    return jnp.exp(jnp.interp(logx, logxs, logys))

def limit_by_MTOV(m: Array, 
                  r: Array, 
                  l: Array) -> tuple[Array, Array, Array]:
    """
    Limits the M, R and Lambda curves to be below MTOV in a jit-friendly manner (i.e., static shape sizes).
    The idea now is to feed this into some routine that creates an interpolation out of this, which then uses jnp.unique to get rid of these duplicates
    NOTE: this assumes that the M curve increases up to a point and potentially decreases after that point. In case the EOS has some weird features and the M curve increases again, this function will return weird results.
    TODO: generalize this for weird EOS or check if we do not have those weird EOS when sampling NEPs.
    
    Args:
        m (Array["npoints"]): Original mass curve
        r (Array["npoints"]): Original radius curve
        l (Array["npoints"]): Original lambdas curve
        
    Returns:
        tuple[Array["npoints"], Array["npoints"], Array["npoints"]]: Tuple of new mass, radius and lambdas curves, where the part of the curves where mass decreases is replaced with duplication of the first entry of the M, R and Lambda arrays.
    """
    
    # Separate head and tail of m, r and l arrays    
    m_first = m.at[0].get()
    r_first = r.at[0].get()
    l_first = l.at[0].get()
    
    m_first_array = jnp.array([m_first])
    r_first_array = jnp.array([r_first])
    l_first_array = jnp.array([l_first])
    
    m_remove_first = m[1:]
    r_remove_first = r[1:]
    l_remove_first = l[1:]
    
    # Where m is increasing, save array, otherwise repeat the first element (discard that part of the curve)
    m_is_increasing = jnp.diff(m) > 0
    
    m_new = jnp.where(m_is_increasing, m_remove_first, m_first)
    r_new = jnp.where(m_is_increasing, r_remove_first, r_first)
    l_new = jnp.where(m_is_increasing, l_remove_first, l_first)
    
    # Because of diff dropping an element, add the first element back
    m_new = jnp.concatenate([m_first_array, m_new])
    r_new = jnp.concatenate([r_first_array, r_new])
    l_new = jnp.concatenate([l_first_array, l_new])
    
    # Sort in increasing values of M for plotting etc
    sort_idx = jnp.argsort(m_new)
    
    m_new = m_new[sort_idx]
    r_new = r_new[sort_idx]
    l_new = l_new[sort_idx]
    
    return m_new, r_new, l_new


###############
### SPLINES ###
###############

def cubic_spline(xq: Float[Array, "n"],
                 xp: Float[Array, "n"],
                 fp: Float[Array, "n"]):
    """
    Create a cubic spline interpolating function through (xp, fp) with interpax (https://github.com/f0uriest/interpax)
    Args:
        xq (Float[Array, "n"]): x values at which we are going to evaluate the spline interpolator
        xp (Float[Array, "n"]): x values of the data points
        fp (Float[Array, "n"]): y values of the data points, i.e. fp = f(xp)
    """
    return interpax_interp1d(xq, xp, fp, method = "cubic")
