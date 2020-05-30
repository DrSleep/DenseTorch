from __future__ import division

cimport cython
cimport numpy as np
import numpy as np


def fast_cm(unsigned char[::1] preds, unsigned char[::1] gt,
            int n_classes):
    """Computing confusion matrix faster.

    Args:
      preds (Tensor) : predictions (either flatten or of size (len(gt), top-N)).
      gt (Tensor) : flatten gt.
      n_classes (int) : number of classes.

    Returns:

      Confusion matrix (Tensor of size (n_classes, n_classes)).

    """
    cdef np.ndarray[np.int_t, ndim=2] cm = np.zeros((n_classes, n_classes),
                                                    dtype=np.int_)
    cdef np.intp_t i,a,p, n = gt.shape[0]

    for i in xrange(n):
        a = gt[i]
        p = preds[i]
        cm[a, p] += 1
    return cm

def compute_iu(np.ndarray[np.int_t, ndim=2] cm):
    """Compute IU from confusion matrix.

    Args:
      cm (Tensor) : square confusion matrix.

    Returns:
      IU vector (Tensor).

    """
    cdef unsigned int pi = 0
    cdef unsigned int gi = 0
    cdef unsigned int ii = 0
    cdef unsigned int denom = 0
    cdef unsigned int n_classes = cm.shape[0]
    # IU is between 0 and 1, hence any value larger than that can be safely ignored
    cdef unsigned int default_value = 2
    cdef np.ndarray[np.float_t, ndim=1] IU = np.ones(n_classes) * default_value
    cdef np.intp_t i
    for i in xrange(n_classes):
        pi = sum(cm[:, i])
        gi = sum(cm[i, :])
        ii = cm[i, i]
        denom = pi + gi - ii
        if denom > 0:
            IU[i] = ii / denom
    return IU
