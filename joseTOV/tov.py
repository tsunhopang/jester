# this whole script is written in geometric unit
from . import utils
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, Dopri8, Tsit5, SaveAt, PIDController
from typing import Callable


def tov_ode(h, y, eos):
    # fetch the eos arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    # actual equations
    r, m, H, b = y
    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    dedp = e / p * jnp.interp(h, hs, dloge_dlogps)

    A = 1.0 / (1.0 - 2.0 * m / r)
    C1 = 2.0 / r + A * (2.0 * m / (r * r) + 4.0 * jnp.pi * r * (p - e))
    C0 = A * (
        -6 / (r * r)
        + 4.0 * jnp.pi * (e + p) * dedp
        + 4.0 * jnp.pi * (5.0 * e + 9.0 * p)
    ) - jnp.power(2.0 * (m + 4.0 * jnp.pi * r * r * r * p) / (r * (r - 2.0 * m)), 2.0)

    drdh = -r * (r - 2.0 * m) / (m + 4.0 * jnp.pi * r * r * r * p)
    dmdh = 4.0 * jnp.pi * r * r * e * drdh
    dHdh = b * drdh
    dbdh = -(C0 * H + C1 * b) * drdh

    dydt = drdh, dmdh, dHdh, dbdh

    return dydt


def calc_k2(R, M, H, b):

    y = R * b / H
    C = M / R

    num = (
        (8.0 / 5.0)
        * jnp.power(1 - 2 * C, 2.0)
        * jnp.power(C, 5.0)
        * (2 * C * (y - 1) - y + 2)
    )
    den = (
        2
        * C
        * (
            4 * (y + 1) * jnp.power(C, 4)
            + (6 * y - 4) * jnp.power(C, 3)
            + (26 - 22 * y) * C * C
            + 3 * (5 * y - 8) * C
            - 3 * y
            + 6
        )
    )
    den -= (
        3
        * jnp.power(1 - 2 * C, 2)
        * (2 * C * (y - 1) - y + 2)
        * jnp.log(1.0 / (1 - 2 * C))
    )

    return num / den


def tov_solver(eos, pc):
    # fetch the eos arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    # central values
    hc = utils.interp_in_logspace(pc, ps, hs)
    ec = utils.interp_in_logspace(hc, hs, es)
    dedp_c = ec / pc * jnp.interp(hc, hs, dloge_dlogps)
    dhdp_c = 1.0 / (ec + pc)
    dedh_c = dedp_c / dhdp_c

    # initial values
    # TODO: how to choose this value?
    dh = -1e-3 * hc
    h0 = hc + dh
    r0 = jnp.sqrt(3.0 * (-dh) / 2.0 / jnp.pi / (ec + 3.0 * pc))
    r0 *= 1.0 - 0.25 * (ec - 3.0 * pc - 0.6 * dedh_c) * (-dh) / (ec + 3.0 * pc)
    m0 = 4.0 * jnp.pi * ec * jnp.power(r0, 3.0) / 3.0
    m0 *= 1.0 - 0.6 * dedh_c * (-dh) / ec
    H0 = r0 * r0
    b0 = 2.0 * r0

    y0 = (r0, m0, H0, b0)

    sol = diffeqsolve(
        ODETerm(tov_ode),
        Dopri5(),
        t0=h0,
        t1=0,
        dt0=dh,
        y0=y0,
        args=eos,
        saveat=SaveAt(t1=True),
        stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
    )

    R = sol.ys[0][-1]
    M = sol.ys[1][-1]
    H = sol.ys[2][-1]
    b = sol.ys[3][-1]

    k2 = calc_k2(R, M, H, b)
    
    return M, R, k2

def tovrhs_rahul(t: jnp.array,
                 yp: tuple[jnp.array, jnp.array, jnp.array],
                 interpolations = tuple[Callable]):
    
    """
    Right hand side of TOV equations.
    TODO: type hinting.

    Returns:
        _type_: _description_
    """
  
    eos1, eos2, eos3 = interpolations
  
    r, m, y = yp
    
    eps  = eos1(t)
    p    = eos2(t)
    Ga   = eos3(t)
    
    dr_dh = -r*(r-2*m)/(m + 4* jnp.pi * r**3 * p )
    
    dm_dh = 4 * jnp.pi * r**2 * eps * dr_dh
        
    l1 = (r-2*m)*(y+1)*y/(m+4*jnp.pi*r**3*p) + y
    l2 = (m-4*jnp.pi*r**3*eps)*y/(m+4*jnp.pi*r**3*p) + (4*jnp.pi*r**3*(5*eps+9*p)-6*r)/(m+4*jnp.pi*r**3*p)
    l3 = 4*jnp.pi*r**3*(eps+p)**2/(p*Ga*(m+4*jnp.pi*r**3*p)) - 4*(m+4*jnp.pi*r**3*p)/(r-2*m)
  
    dy_dh = l1+l2+l3
    
    return dr_dh, dm_dh, dy_dh

def tov_solver_rahul(p: jnp.array,
                     e: jnp.array,
                     cs2: jnp.array,
                     h_c_array: jnp.array,
                     rtol: float=1.e-6,
                     atol: float=1.e-5):
    
    # TODO: clean this up if OK?
    CONV_MeV_fm3_to_g_cm3   = 1 / utils.MeV_fm_inv3_to_geometric * 1.78266181e-36 * 1e48
    CONV_MeV_fm3_to_dyn_cm2 = 1 / utils.MeV_fm_inv3_to_geometric * 1.78266181e-36 * 2.99792458e8**2 * 1e52

    G     = 6.6743     * 10**(-8)      # Newton's gravitational constant in cgs units
    c     = 2.99792458 * 10**10        # Speed of light in cgs units
    M_sun = 1.476      * 10**(5)       # Mass of the sun in geometrized units
    
    e = e *   CONV_MeV_fm3_to_g_cm3       * G * M_sun**2 / c**2
    p = p *   CONV_MeV_fm3_to_dyn_cm2     * G * M_sun**2 / c**4
    # cs2 = cs2 * c**2
    
    gamma    = (e + p) / p * cs2
    enthalpy = jnp.sort(utils.cumtrapz(1/(e + p), p))
                
    def epsilon_from_enthalpy(h):
        interpolator = lambda x: jnp.interp(x, enthalpy, jnp.log(e))
        # interpolator = lambda x: utils.cubic_spline(x, enthalpy, jnp.log(e))
        return jnp.exp(interpolator(h))
    
    def pressure_from_enthalpy(h):
        interpolator = lambda x: jnp.interp(x, enthalpy, jnp.log(p))
        # interpolator = lambda x: utils.cubic_spline(x, enthalpy, jnp.log(p))
        return jnp.exp(interpolator(h))
    
    def gamma_from_enthalpy(h):
        interpolator = lambda x: jnp.interp(x, enthalpy, gamma)
        # interpolator = lambda x: utils.cubic_spline(h, enthalpy, gamma)
        return interpolator(h)
    
    interpolations = (epsilon_from_enthalpy, pressure_from_enthalpy, gamma_from_enthalpy)
    
    def single_call(h_c):
        r0 = 1.e-3 
        m0 = 4/3 * jnp.pi * r0**3 * epsilon_from_enthalpy(h_c)
        y0 = 2.0
        
        initial = r0, m0, y0
        # TODO: how to choose this value?
        dh = -1e-3 * h_c
                 
        sol = diffeqsolve(
            ODETerm(tovrhs_rahul),
            Dopri5(),
            t0=h_c,
            t1=0,
            dt0=dh,
            y0=initial,
            args=interpolations,
            saveat=SaveAt(t1=True),
            stepsize_controller=PIDController(rtol=rtol, atol=atol),
        )
        
        R = sol.ys[0][-1]
        M = sol.ys[1][-1]
        
        C = M/R
        Y = sol.ys[2][-1]

        Xi = 4*C**3*(13-11*Y+C*(3*Y-2)+2*C**2*(1+Y)) + 3*(1-2*C)**2*(2-Y+2*C*(Y-1))*jnp.log(1-2*C)+2*C*(6-3*Y+3*C*(5*Y-8))
        
        Lambda = 16/(15*Xi) * (1-2*C)**2*(2+2*C*(Y-1)-Y)
        Radius = R * M_sun * 10**(-5)
        
        return M, Radius, Lambda
    
    pc = pressure_from_enthalpy(h_c_array)
    m, r, l = jax.vmap(single_call)(h_c_array)
    
    return pc, m, r, l