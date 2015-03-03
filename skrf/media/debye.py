

'''
.. module:: skrf.media.debye

========================================
debye (:mod:`skrf.media.debye`)
========================================

A media with parameters governed by the Debye equation.
'''
from scipy.constants import mu_0, pi
from numpy import real, imag
from .distributedCircuit import DistributedCircuit


class Debye(DistributedCircuit):
    '''
    A media with parameters governed by the Debye equation.

    .. math::
        \\epsilon = \\epsilon^\\prime + j\\epsilon^{\\prime\\prime}\\\\
        \\epsilon^\\prime = \\epsilon_\\infty+
        \\frac{\\epsilon_s-\\epsilon_\\infty}{1+\\omega^2\\tau^2}\\\\
        \\epsilon^{\\prime\\prime} =
        \\frac{(\\epsilon_s-\\epsilon_\\infty)\\omega\\tau}{1+\\omega^2\\tau^2}

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
            :class:`~skrf.media.debye.Debye`

    '''
    def __init__(self, frequency,  es=0, einf=0, tau=0, *args, **kwargs):
        '''
        Debye material initializer

        Parameters
        -----------
        frequency : :class:`~skrf.frequency.Frequency` object
                frequency band of this transmission line medium
        es : number
            static permittivity
        einf : number
            high frequency limit permittivity
        tau : number
            Relaxation time, in seconds
        \*args, \*\*kwargs : arguments and keyword arguments

        Notes
        ------
        The Debye equation is

        .. math::
            \\epsilon = \\epsilon^\\prime + j\\epsilon^{\\prime\\prime}\\\\
            \\epsilon^\\prime = \\epsilon_\\infty+
            \\frac{\\epsilon_s-\\epsilon_\\infty}{1+\\omega^2\\tau^2}\\\\
            \\epsilon^{\\prime\\prime} =
            \\frac{(\\epsilon_s-\\epsilon_\\infty)\\omega\\tau}{1+\\omega^2\\tau^2}

        For water at 25degC $\\epsilon_s=78.408$, $\\epsilon_\\infty=5.2$ and
        $\\tau=8.27$ ps. Reference: CRC Handbook of Chemistry and Physics, 95th
        Ed., 2014-2015.

        The distributed circuit parameters are related to a space's
        field properties by

        ===============================  ==============================
        Circuit Property                 Field Property
        ===============================  ==============================
        distributed_capacitance          real(eps)
        distributed_resistance           :math:`\\omega  imag(eps)
        distributed_inductance           real(mu_0)
        distributed_conductance          :math:`\\omega  imag(mu_0)
        ===============================  ==============================
        '''

        omega = 2 * pi * frequency.f

        ep = einf + (es - einf)/(1+omega**2*tau**2)
        edp = ((es-einf)*omega*tau)/(1+omega**2*tau**2)
        eps = ep+1j*edp

        DistributedCircuit.__init__(self,
                                    frequency=frequency,
                                    C=real(eps),
                                    G=frequency.w * imag(eps),
                                    I=real(mu_0),
                                    R=frequency.w * imag(mu_0),
                                    *args, **kwargs
                                    )

    def __str__(self):
        f = self.frequency
        output = ('Debye  Media.  %i-%i %s.  %i points' %
                  (f.f_scaled[0], f.f_scaled[-1], f.unit, f.npoints))
        return output

    def __repr__(self):
        return self.__str__()
