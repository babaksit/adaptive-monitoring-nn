from numbers import Integral
from typing import Union, Optional
import pandas as pd
import numpy as np
import torch
from fastcore.xtras import is_listy


def sliding_window(window_len:int, stride:Union[None, int]=1, start:int=0, pad_remainder:bool=False, padding:str="post", padding_value:float=np.nan,
                  add_padding_feature:bool=True, get_x:Union[None, int, list]=None, get_y:Union[None, int, list]=None, y_func:Optional[callable]=None,
                  output_processor:Optional[callable]=None, copy:bool=False, horizon:Union[int, list]=1, seq_first:bool=True, sort_by:Optional[list]=None,
                  ascending:bool=True, check_leakage:bool=True):

    """
    Applies a sliding window to a 1d or 2d input (np.ndarray, torch.Tensor or pd.DataFrame)
    Args:
        window_len          = length of lookback window
        stride              = n datapoints the window is moved ahead along the sequence. Default: 1. If None, stride=window_len (no overlap)
        start               = determines the step where the first window is applied: 0 (default) or a given step (int). Previous steps will be discarded.
        pad_remainder       = allows to pad remainder subsequences when the sliding window is applied and get_y == [] (unlabeled data).
        padding             = 'pre' or 'post' (optional, defaults to 'pre'): pad either before or after each sequence. If pad_remainder == False, it indicates
                              the starting point to create the sequence ('pre' from the end, and 'post' from the beginning)
        padding_value       = value (float) that will be used for padding. Default: np.nan
        add_padding_feature = add an additional feature indicating whether each timestep is padded (1) or not (0).
        horizon             = number of future datapoints to predict (y). If get_y is [] horizon will be set to 0.
                            * 0 for last step in each sub-window.
                            * n > 0 for a range of n future steps (1 to n).
                            * n < 0 for a range of n past steps (-n + 1 to 0).
                            * list : for those exact timesteps.
        get_x               = indices of columns that contain the independent variable (xs). If None, all data will be used as x.
        get_y               = indices of columns that contain the target (ys). If None, all data will be used as y.
                              [] means no y data is created (unlabeled data).
        y_func              = optional function to calculate the ys based on the get_y col/s and each y sub-window. y_func must be a function applied to axis=1!
        output_processor    = optional function to process the final output (X (and y if available)). This is useful when some values need to be removed.
                              The function should take X and y (even if it's None) as arguments.
        copy                = copy the original object to avoid changes in it.
        seq_first           = True if input shape (seq_len, n_vars), False if input shape (n_vars, seq_len)
        sort_by             = column/s used for sorting the array in ascending order
        ascending           = used in sorting
        check_leakage       = checks if there's leakage in the output between X and y
    Input:
        You can use np.ndarray, pd.DataFrame or torch.Tensor as input
        shape: (seq_len, ) or (seq_len, n_vars) if seq_first=True else (n_vars, seq_len)
    """

    if get_y == []: horizon = 0
    if horizon == 0: horizon_rng = np.array([0])
    elif is_listy(horizon): horizon_rng = np.array(horizon)
    elif isinstance(horizon, Integral): horizon_rng = np.arange(1, horizon + 1) if horizon > 0 else np.arange(horizon + 1, 1)
    min_horizon = min(horizon_rng)
    max_horizon = max(horizon_rng)
    _get_x = slice(None) if get_x is None else get_x.tolist() if isinstance(get_x, pd.core.indexes.base.Index) else [get_x] if not is_listy(get_x) else get_x
    _get_y = slice(None) if get_y is None else get_y.tolist() if isinstance(get_y, pd.core.indexes.base.Index) else [get_y] if not is_listy(get_y) else get_y
    if min_horizon <= 0 and y_func is None and get_y != [] and check_leakage:
        assert get_x is not None and  get_y is not None and len([y for y in _get_y if y in _get_x]) == 0,  \
        'you need to change either horizon, get_x, get_y or use a y_func to avoid leakage'
    if stride == 0 or stride is None:
        stride = window_len
    if pad_remainder: assert padding in ["pre", "post"]

    def _inner(o):
        if copy:
            if isinstance(o, torch.Tensor):  o = o.clone()
            else: o = o.copy()
        if not seq_first: o = o.T
        if isinstance(o, pd.DataFrame):
            if sort_by is not None: o.sort_values(by=sort_by, axis=0, ascending=ascending, inplace=True, ignore_index=True)
            if get_x is None: X = o.values
            elif isinstance(_get_x, str) or (is_listy(_get_x) and isinstance(_get_x[0], str)): X = o.loc[:, _get_x].values
            else: X = o.iloc[:, _get_x].values
            if get_y == []: y = None
            elif get_y is None: y = o.values
            elif isinstance(_get_y, str) or (is_listy(_get_y) and isinstance(_get_y[0], str)): y = o.loc[:, _get_y].values
            else: y = o.iloc[:, _get_y].values
        else:
            if isinstance(o, torch.Tensor): o = o.numpy()
            if o.ndim < 2: o = o[:, None]
            if get_x is None: X = o
            else: X = o[:, _get_x]
            if get_y == []: y = None
            elif get_y is None: y = o
            else: y = o[:, _get_y]

        # X
        if start != 0:
            X = X[start:]
        X_len = len(X)
        if not pad_remainder:
            if X_len < window_len + max_horizon:
                return None, None
            else:
                n_windows = 1 + (X_len - max_horizon - window_len) // stride
        else:
            n_windows = 1 + max(0, np.ceil((X_len - max_horizon - window_len) / stride).astype(int))
        X_max_len = window_len + max_horizon + (n_windows - 1) * stride # total length required (including y)
        X_seq_len = X_max_len - max_horizon

        if pad_remainder and X_len < X_max_len:
            if add_padding_feature:
                X = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
            _X = np.empty((X_max_len - X_len, *X.shape[1:]))
            _X[:] = padding_value
            if add_padding_feature:
                _X[:, -1] = 1
            if padding == "pre":
                X = np.concatenate((_X, X))
            elif padding == "post":
                X = np.concatenate((X, _X))
        if padding == "pre":
            X_start = X_len - X_max_len
            X = X[-X_max_len:-X_max_len + X_seq_len]
        elif padding == "post":
            X_start = 0
            X = X[:X_seq_len]

        X_sub_windows = (np.expand_dims(np.arange(window_len), 0) +
                         np.expand_dims(np.arange(n_windows * stride, step=stride), 0).T)
        X = np.transpose(X[X_sub_windows], (0, 2, 1))

        # y
        if get_y != [] and y is not None:
            y_start = start + X_start + window_len + min_horizon - 1
            y_max_len = max_horizon - min_horizon + 1 + (n_windows - 1) * stride
            y = y[y_start:y_start + y_max_len]
            y_len = len(y)
            y_seq_len = y_max_len

            if pad_remainder and y_len < y_max_len:
                _y = np.empty((y_max_len - y_len, *y.shape[1:]))
                _y[:] = padding_value
                if padding == "pre":
                    y = np.concatenate((_y, y))
                elif padding == "post":
                    y = np.concatenate((y, _y))

            y_sub_windows = (np.expand_dims(horizon_rng - min_horizon, 0)+
                             np.expand_dims(np.arange(n_windows * stride, step=stride), 0).T)
            y = y[y_sub_windows]

            if y_func is not None and len(y) > 0:
                y = y_func(y)
            if y.ndim >= 2:
                for d in np.arange(1, y.ndim)[::-1]:
                    if y.shape[d] == 1: y = np.squeeze(y, axis=d)
            if y.ndim == 3:
                y = y.transpose(0, 2, 1)
        if output_processor is not None:
            X, y = output_processor(X, y)
        return X, y
    return _inner
