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
    cdef real first_, last_, direction, prev_last = NAN
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
    cdef real x_slope_, t_slope_, first_xi_, first_ti_
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
