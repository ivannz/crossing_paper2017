# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

# Author: Ivan Nazarov <ivannnnz@gmail.com>

import numpy as _np
cimport numpy as np
cimport cython

from libc.math cimport fabs, floor, ceil, NAN #, isnan

np.import_array()

ctypedef fused real:
    cython.floating

def crossings(real[:] x, real[:] t, real origin, real scale):
    """Compute crossings of an integer grid `scale`*Z with specified `origin`
    by the given sample path of the stochastic process (t, x).
    """
    cdef np.intp_t n_samples = x.shape[0]
    cdef np.intp_t[::1] size = _np.empty(n_samples - 1, dtype=_np.int)
    cdef real[::1] first = _np.empty(n_samples - 1, dtype=_np.float)
    
    cdef np.intp_t i
    cdef double first_, last_, direction, prev_last = NAN
    cdef np.intp_t total = 0, size_
    with nogil:
        # Detect integer-level crossings, ignoring re-crossings of the same level
        for i in range(n_samples - 1):
            direction, size_ = 0.0, 0
            if x[i] < x[i+1]:
                first_, last_ = ceil((x[i] - origin) / scale), floor((x[i+1] - origin) / scale)
                direction = +1.0
            elif x[i] > x[i+1]:
                first_, last_ = floor((x[i] - origin) / scale), ceil((x[i+1] - origin) / scale)
                direction = -1.0
            if direction != 0.0:
                size_ = <int>fabs(last_ + direction - first_)
                if size_ > 0 and prev_last == first_:
                    first_ += direction
                    size_ -= 1
                if size_ > 0:
                    prev_last = last_
            first[i], size[i] = first_, size_
            total += size_

    cdef real[::1] xi = _np.empty(total, dtype=_np.float)
    cdef real[::1] ti = _np.empty(total, dtype=_np.float)

    cdef np.int_t k, j = 0
    cdef double x_slope_, t_slope_, first_xi_, first_ti_
    with nogil:
        # Interpolate the crossing times and crossing levels
        for i in range(n_samples-1):
            size_ = size[i]
            if size_ > 0:
                x_slope_ = +scale if x[i+1] > x[i] else -scale
                t_slope_ = (t[i+1] - t[i]) / (x[i+1] - x[i])
                first_ = first[i] * scale + origin
                for k in range(size_):
                    xi[j] = first_ + x_slope_ * k
                    ti[j] = t[i] + t_slope_ * (xi[j] - x[i])
                    j += 1
    return xi, ti

## Marginally slower
#             size_ = size[i]
#             if size_ > 0:
#                 t_slope_ = (t[i+1] - t[i]) / (x[i+1] - x[i])
#                 xi[j] = first[i] * scale
#                 ti[j] = t[i] + t_slope_ * (xi[j] - x[i])
#                 j += 1
#             if size_ > 1:
#                 x_slope_ = +scale if x[i+1] > x[i] else -scale
#                 for k in range(size_ - 1):
#                     xi[j] = xi[j-1] + x_slope_
#                     ti[j] = ti[j-1] + t_slope_ * x_slope_
#                     j += 1
#    return xi, ti

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.nonecheck(False)
def _align_crossing_times(real[:] t0, real[:] t1):
    # Find the alignment vector
    cdef np.intp_t n_samples = t1.shape[0]
    cdef np.intp_t[::1] index = _np.empty(n_samples, dtype=_np.int)
    cdef np.intp_t i0 = 0, i1
    with nogil:
        for i1 in range(n_samples):
            ## If i1-th hit exists then there must be the corresponding i0-th hit.
            while t0[i0] < t1[i1]:
                i0 += 1
            ## Correct the index if we overshot due to numerical accuracy.
            if i0 > 0:
                if t0[i0] - t1[i1] > t1[i1] - t0[i0-1]:
                    i0 -= 1
            index[i1] = i0
            i0 += 1
    return index

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.nonecheck(False)
def _get_statistics(np.intp_t[:] index1, real[:] x0):
    # Compute the level statistics
    cdef np.intp_t n_samples = index1.shape[0]

    cdef np.int_t[::1] ud = _np.empty(n_samples - 1, dtype=_np.int)
    cdef np.int_t[::1] du = _np.empty(n_samples - 1, dtype=_np.int)
    cdef np.int8_t[::1] direction = _np.empty(n_samples - 1, dtype=_np.int8)

    cdef np.intp_t ud_, du_, i1
    cdef np.intp_t i0 = index1[0] + 1
    with nogil:
        for i1 in range(n_samples-1):
            # Count \/ and /\ excursions
            du_, ud_ = 0, 0
            # All pairs up to the one before last are necessarily excursions.
            while i0 < index1[i1+1] - 2:
                # check the direction of the last traversal in the pair
                if x0[i0] > x0[i0-1]:
                    du_ += 1
                else:
                    ud_ += 1
                i0 += 2
            du[i1], ud[i1] = du_, ud_

            # Get the crossing direction
            direction[i1] = +1 if x0[i0] > x0[i0-1] else -1

            # Advance to the next pair
            i0 += 2

    return direction, ud, du
