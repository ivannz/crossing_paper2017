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

def structural_statistics(X, T, scale=1.0, percentiles=(0.1, 0.5, 1.0, 2.5, 5.0,
                                                        10, 25, 50, 75, 90, 95,
                                                        97.5, 99, 99.5, 99.9)):
    """Computes the structural statistics of the crossing tree for the sample path
    of a process :math:`(X_t)_{t\\in T}`. The crossing tree is constructed over 
    nested lattices :math:`\\delta 2^n \\mathbb{Z}` rooted at :math:`X_{t_0}`.

    The `scale` parameter determines the resolution of the finest spatial lattice,
    :math:`\\delta`.

    The `percentiles` array determines which quantiles of the crossing durations to track.
    """
    xi, ti, offspring, Vnk, Znk, Wnk = crossing_tree(X, T, scale=scale, origin=X[0])
    # Sanity check.
    # for j in xrange(len(Znk)):
    #     assert np.allclose(2 * Vnk[j][:, :2].sum(axis=1) + 2, Znk[j])

    # Nn[n] -- the total number of crossings of grid with spacing \delta 2^n
    Nn = np.r_[len(xi), [len(index_) for index_ in offspring]] - 1

    # Dnk[n, k] -- the number of crossings of grid \delta 2^{n+1}
    #  with exactly 2(k+1) subcrossings of grid \delta 2^n.
    freq = [np.bincount(Zk)[2::2] for Zk in Znk]
    Dnk = np.zeros((len(Znk), max(len(f) for f in freq)), np.int)
    for l, f in enumerate(freq):
        Dnk[l, :len(f)] = f

    # Vnde[n, d, e] -- the total number of up-down(e=0) and down-up(e=1)
    #  excursions in a downward (d=0) or upward (d=1) crossing of level
    #  n+1
    Vnde = np.array([(Vk[Vk[:, 2] < 0, :2].sum(axis=0),
                      Vk[Vk[:, 2] > 0, :2].sum(axis=0))
                     for Vk in Vnk], dtype=np.int)

    # Wnp[n, p] -- the p-th empirical quantile of the n-th level crossing
    #  durations.
    empty_ = np.full_like(percentiles, np.nan)
    Wnp = np.stack([np.percentile(Wk, percentiles) if len(Wk) > 0 else empty_ for Wk in Wnk])

    # The average crossing duration and its standard deviation
    Wavgn = np.array([np.mean(Wk) if len(Wk) > 0 else np.nan for Wk in Wnk])
    Wstdn = np.array([np.std(Wk) if len(Wk) > 0 else np.nan for Wk in Wnk])
    return scale, Nn, Dnk, Vnde, Wnp, Wavgn, Wstdn

def collect_structural_statistics(tree_statistics):
    """Collects the structural statistics of the crossing trees .
    """
    # materialize the iterable of structural statistics data
    tree_statistics = list(tree_statistics)

    # scale_m is an array of length M of $\delta_m$.
    scale_m = np.array([scale for scale, Nn, Dnk, Vnde, Wnp, Wavgn, Wstdn in tree_statistics])

    # Nmn is an MxL matrix of numbers of complete crossings.
    Nmn = [Nn for scale, Nn, Dnk, Vnde, Wnp, Wavgn, Wstdn in tree_statistics]
    n_levels = max(Nn.shape[0] for Nn in Nmn)

    Nmn = np.stack([np.pad(Nn, (0, n_levels - Nn.shape[0]), mode="constant").astype(np.float)
                    for Nn in Nmn])

    # Dmnk is an MxLxK tensor of offspring frequencies (# of crossings with the
    # specified number of subcrossings).
    Dmnk = [Dnk for scale, Nn, Dnk, Vnde, Wnp, Wavgn, Wstdn in tree_statistics]
    n_pairs = max(Dnk.shape[1] for Dnk in Dmnk)

    Dmnk = np.stack([np.pad(Dnk, ((0, n_levels - 1 - Dnk.shape[0]), (0, n_pairs - Dnk.shape[1])),
                            mode="constant").astype(np.float)
                     for Dnk in Dmnk])

    # Wmnp is an MxLxP tensor of empirical quantiles of the crossing durations.
    Wmnp = np.stack([np.pad(Wnp.astype(np.float), ((0, n_levels - 1 - Wnp.shape[0]), (0, 0)),
                            mode="constant", constant_values=np.nan)
                     for scale, Nn, Dnk, Vnde, Wnp, Wavgn, Wstdn in tree_statistics])

    # Vmnde is an MxLx2x2 tensor of excursion frequencies.
    Vmnde = np.stack([np.pad(Vnde.astype(np.float), ((0, n_levels - 1 - Vnde.shape[0]),
                                                     (0, 0), (0, 0)),
                             mode="constant", constant_values=np.nan)
                      for scale, Nn, Dnk, Vnde, Wnp, Wavgn, Wstdn in tree_statistics])

    # Wavgmn is an MxL matrix of within level average of the crossing durations.
    Wavgmn = np.stack([np.pad(Wavgn.astype(np.float), (0, n_levels - 1 - Wavgn.shape[0]),
                              mode="constant", constant_values=np.nan)
                       for scale, Nn, Dnk, Vnde, Wnp, Wavgn, Wstdn in tree_statistics])

    # Wstdmn is an MxL matrix of within level standard devision of the crossing durations.
    Wstdmn = np.stack([np.pad(Wstdn.astype(np.float), (0, n_levels - 1 - Wstdn.shape[0]),
                              mode="constant", constant_values=np.nan)
                       for scale, Nn, Dnk, Vnde, Wnp, Wavgn, Wstdn in tree_statistics])

    return scale_m, Nmn, Dmnk, Vmnde, Wmnp, Wavgmn, Wstdmn
