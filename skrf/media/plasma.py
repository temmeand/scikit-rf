

'''
.. module:: skrf.media.plasma

========================================
plasma (:mod:`skrf.media.plasma`)
========================================

A Plane-wave in a non-magnetized plasma.
'''
from scipy.constants import epsilon_0, mu_0, m_e, e, pi
from numpy import real, imag
from .distributedCircuit import DistributedCircuit


class Plasma(DistributedCircuit):
    '''
    Represents a plane-wave in a non-magnetized plasma, defined by the electron
    density and collision frequency. The permittivity is calculated using

    .. math::

        \\epsilon (\\omega) = \\epsilon_0 \\left( 1-
        \\frac{\\omega_p^2}{\\omega^2+\\nu^2} \\right)
        -j \\epsilon_0 \\frac{\\omega_p^2\\nu}{\\omega(\\omega^2+\\nu^2)}\\\\
        \\omega_p=\\sqrt{\\frac{n_e e^2}{\\epsilon_0 m_e}}

    where :math:`\\omega=2\\pi f`.

    The field properties are related to a distributed circuit transmission line
    model given in circuit theory by:

    ===============================  ==============================
    Circuit Property                 Field Property
    ===============================  ==============================
    distributed_capacitance          real(ep_0*ep_r)
    distributed_resistance           imag(ep_0*ep_r)
    distributed_inductance           real(mu_0*mu_r)
    distributed_conductance          imag(mu_0*mu_r)
    ===============================  ==============================

    This class's inheritence is;
            :class:`~skrf.media.media.Media`->
            :class:`~skrf.media.distributedCircuit.DistributedCircuit`->
            :class:`~skrf.media.plasma.Plasma`

    '''
    def __init__(self, frequency,  ne=0, nu=0, *args, **kwargs):
        '''
        Plasma initializer

        Parameters
        -----------

        frequency : :class:`~skrf.frequency.Frequency` object
                frequency band of this transmission line medium
        ne : number
            Electron density per cubic meter
        nu : number
            Collision frequency in Hz
        \*args, \*\*kwargs : arguments and keyword arguments

        Notes
        ------
        The permittivity is calculated using

        .. math::

            \\epsilon (\\omega) = \\epsilon_0 \\left( 1-
            \\frac{\\omega_p^2}{\\omega^2+\\nu^2} \\right)
            -j\\epsilon_0 \\frac{\\omega_p^2\\nu}{\\omega(\\omega^2+\\nu^2)}\\\\
            \\omega_p=\\sqrt{\\frac{n_e e^2}{\\epsilon_0 m_e}}

        where :math:`\\omega=2\\pi f`.

        The distributed circuit parameters are related to a space's
        field properties by

        ===============================  ==============================
        Circuit Property                 Field Property
        ===============================  ==============================
        distributed_capacitance          real(ep_0*ep_r)
        distributed_resistance           :math:`\\omega  imag(ep_0*ep_r)
        distributed_inductance           real(mu_0*mu_r)
        distributed_conductance          :math:`\\omega  imag(mu_0*mu_r)
        ===============================  ==============================
        '''
        omega = 2 * pi * frequency.f

        wp2 = (ne*e**2)/(epsilon_0*m_e)
        ep_r = ((1-wp2/(omega**2+nu**2))
                - 1j * wp2*nu/(omega*(omega**2+nu**2)))
        mu_r = 1

        DistributedCircuit.__init__(self,
                                    frequency=frequency,
                                    C=real(epsilon_0*ep_r),
                                    G=frequency.w * imag(epsilon_0*ep_r),
                                    I=real(mu_0*mu_r),
                                    R=frequency.w * imag(mu_0*mu_r),
                                    *args, **kwargs
                                    )

    def __str__(self):
        f = self.frequency
        output = ('Plasma  Media.  %i-%i %s.  %i points' %
                  (f.f_scaled[0], f.f_scaled[-1], f.unit, f.npoints))
        return output

    def __repr__(self):
        return self.__str__()
