'''
.. module:: skrf.media.twowire

========================================
twowire (:mod:`skrf.media.twowire`)
========================================

A two-wire transmission line defined by wire dimensions and surrounding media
'''
from scipy.constants import epsilon_0, mu_0, pi
from numpy import arccosh, sqrt
from .distributedCircuit import DistributedCircuit
from .media import to_meters


class TwoWire(DistributedCircuit):
    '''
    A two-wire transmission line defined by wire dimensions and surrounding media

    Calculates and returns the resistance, impedance, conductance, and
    capacitance for a two-wire transmission line. Geometric and electrical
    properties are given for the wire and the media in which the wires are
    located.

    This class's inheritence is;
            :class:`~skrf.media.media.Media`->
            :class:`~skrf.media.distributedCircuit.DistributedCircuit`->
            :class:`~skrf.media.twowire.TwoWire`

    '''
    def __init__(self, frequency, a, D, sigma_c=inf, mu_c=1, eps_r=1,
                 tanDelta=0, mu_r=1, unit='m', *args, **kwargs):
        '''
        Two wire transmission line initializer

        Parameters
        -----------

        frequency : :class:`~skrf.frequency.Frequency` object
                frequency band of this transmission line medium
        ne : number
            Electron density per cubic meter
        nu : number
            Collision frequency in Hz


        Parameters
        ----------
        frequency : :class:`~skrf.frequency.Frequency` object
                frequency band of this transmission line medium
        a : scalar
            Wire radius
        D : scalar
            center-to-center distance of the wires
        sigma_c : scalar
            Conductivity of the wires
        mu_c : scalar, optional
            Relative permeability of the wires. Default is 1.
        eps_r : scalar, optional
            Relative permittivity of the environment. Used as
            epsilon = eps_r*eps_0*(1-j*tanDelta). Default is 1.
        tanDelta : scalar, optional
            Loss tangent of the environment. Used as
            epsilon = eps_r*eps_0*(1-j*tanDelta). Default is 0.
        mu_r : scalar, optional
            Relative permeability of the environment. Default is 1
        unit : ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']
                the units of d.  See :func:`to_meters`, for details
        \*args, \*\*kwargs : arguments and keyword arguments

        '''

        # Scale dimensions
        a = to_meters(a, unit)
        D = to_meters(D, unit)
        eps_sgl = epsilon_0*eps_r
        eps_dbl = epsilon_0*eps_r*tanDelta

        omega = 2 * pi * frequency.f
        # delta_cond = 1/sqrt(pi*freq*sigma_c)

        invCosh = arccosh(D/(2*a))

        # R = 1/(pi*a*sigma_c*delta_cond)
        R = sqrt(omega * mu_c /
                 (2 * (pi * a)**2 * sigma_c
                  * (1-(2 * a/D)**2)))
        G = (pi*omega*eps_dbl)/invCosh
        L = mu_0/pi*invCosh
        C = pi*eps_sgl/invCosh

        DistributedCircuit.__init__(self, frequency=frequency, C=C, G=G,
                                    I=L, R=R, *args, **kwargs)

    def __str__(self):
        f = self.frequency
        output = ('Two Wire T-line.  %i-%i %s.  %i points' %
                  (f.f_scaled[0], f.f_scaled[-1], f.unit, f.npoints))
        return output

    def __repr__(self):
        return self.__str__()
