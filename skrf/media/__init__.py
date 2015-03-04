
'''
.. module:: skrf.media
========================================
media (:mod:`skrf.media`)
========================================

This package provides objects representing transmission line mediums.

The :class:`~media.Media` object is the base-class that is inherited
by specific transmission line instances, such as
:class:`~freespace.Freespace`, or
:class:`~rectangularWaveguide.RectangularWaveguide`. The
:class:`~media.Media` object provides generic methods to produce
:class:`~skrf.network.Network`'s for any transmission line medium, such
as :func:`~media.Media.line` and :func:`~media.Media.delay_short`. These
methods are inherited by the specific transmission line classes,
which interally define relevant quantities such as propagation constant,
and characteristic impedance. This allows the specific transmission line
mediums to produce networks without re-implementing methods for
each specific media instance.

Network components specific to an given transmission line medium
such as :func:`~media.cpw.CPW.cpw_short` and
:func:`~media.microstrip.Microstrip.microstrip_bend`, are implemented
in those object




Media base-class
-------------------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~media.Media

Transmission Line Classes
-------------------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~distributedCircuit.DistributedCircuit
    ~rectangularWaveguide.RectangularWaveguide
    ~cpw.CPW
    ~freespace.Freespace
    ~plasma.Plasma
    ~debye.Debye
    ~colecole.ColeCole
    ~coaxial.Coaxial
    ~twowire.TwoWire


.. _DistributedCircuit: :class:`~skrf.media.distributedCircuit.DistributedCircuit`

'''

from coaxial import Coaxial
from colecole import ColeCole
from cpw import CPW
from debye import Debye
from distributedCircuit import DistributedCircuit
from freespace import Freespace
from media import Media
from plasma import Plasma
from rectangularWaveguide import RectangularWaveguide
from twowire import TwoWire
