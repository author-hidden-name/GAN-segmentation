import numpy as np
import mxnet as mx


class MultiPad(object):
    def __init__(self, axis=(0,), pad_val=0, ret_length=False):
        self._axis = axis

        if isinstance(axis, int):
            axis = (axis,)

        assert isinstance(axis, tuple), 'axis must be an tuple! ' \
                                      'Received axis=%s, type=%s.' % (str(axis),
                                                                      str(type(axis)))
        self._pad_val = pad_val
        self._ret_length = ret_length

    def __call__(self, data):
        """Batchify the input data.
        Parameters
        ----------
        data : list
            A list of N samples. Each sample can be 1) ndarray or
             2) a list/tuple of ndarrays
        Returns
        -------
        batch_data: NDArray
            Data in the minibatch. Shape is (N, ...)
        valid_length: NDArray, optional
            The sequences' original lengths at the padded axis. Shape is (N,). This will only be
            returned in `ret_length` is True.
        """
        if isinstance(data[0], (mx.nd.NDArray, np.ndarray, list)):
            padded_arr, original_length = _pad_arrs_to_max_length(data, self._axis,
                                                                  self._pad_val, True)
            if self._ret_length:
                return padded_arr, original_length
            else:
                return padded_arr
        else:
            raise NotImplementedError


def _pad_arrs_to_max_length(arrs, pad_axes, pad_val, use_shared_mem=False):
    """Inner Implementation of the Pad batchify
    Parameters
    ----------
    arrs : list
    pad_axes : tuple
    pad_val : number
    use_shared_mem : bool, default False
    Returns
    -------
    ret : NDArray
    original_length : NDArray
    """
    if not isinstance(arrs[0], (mx.nd.NDArray, np.ndarray)):
        arrs = [np.asarray(ele) for ele in arrs]
    original_length = np.array([[ele.shape[pad_axis] for pad_axis in pad_axes] for ele in arrs])
    max_size = np.max(original_length, axis=0)

    ret_shape = list(arrs[0].shape)
    for pad_axis, axis_max_size in zip(pad_axes, max_size):
        ret_shape[pad_axis] = axis_max_size

    ret_shape = (len(arrs), ) + tuple(ret_shape)
    if use_shared_mem:
        ret = mx.nd.full(shape=ret_shape, val=pad_val, ctx=mx.Context('cpu_shared', 0),
                         dtype=arrs[0].dtype)
        original_length = mx.nd.array(original_length, ctx=mx.Context('cpu_shared', 0),
                                      dtype=np.int32)
    else:
        ret = mx.nd.full(shape=ret_shape, val=pad_val, dtype=arrs[0].dtype)
        original_length = mx.nd.array(original_length, dtype=np.int32)

    for i, arr in enumerate(arrs):
        slices = [slice(None) for _ in range(arr.ndim)]
        for pad_axis in pad_axes:
            slices[pad_axis] = slice(0, arr.shape[pad_axis])
        slices = [slice(i, i + 1)] + slices
        ret[tuple(slices)] = arr

    return ret, original_length