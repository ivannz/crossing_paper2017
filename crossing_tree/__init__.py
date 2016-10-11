""""""
import numpy as np

from ._crossing import crossings, _align_crossing_times, _get_statistics

from .processes import FractionalBrownianMotion, HermiteProcess, WeierstrassFunction

def crossing_tree(X, T, scale, origin=0):
    """Return the crossing tree for thr process :math:`(X_t)_{t\\in T}`, with
    the finest grid originating from `orgigin` and its resolution set to `scale`.
    """
    offspring, durations = list(), list()
    subcrossings, excursions = list(), list()

    # Get crossings of the finest grid.
    xi1, ti1 = crossings(X, T, scale, origin)
    xi, ti = xi1, ti1

    # The alignment vector always has a zero-th crossing
    while len(ti1) > 1:
        xi0, ti0 = xi1, ti1
        scale *= 2
        xi1, ti1 = crossings(X, T, scale, origin)
        index1 = _align_crossing_times(ti0, ti1).base
        # Keep the tree structure
        offspring.append(index1)
        # Compute the duration of the crossing and the number of sub-crossings
        subcrossings.append(np.diff(index1))
        durations.append(np.diff(ti1))
        # Get statistics: excursions within each crossing and its direction
        excursions.append(_get_statistics(index1, xi0).base)
    return xi, ti, offspring, excursions, subcrossings, durations
