from typing import Dict, Optional, Set, Tuple, Union

import numba
import numpy as np

# ATTENTION: when Numba JIT compilation is active. Some functions may crash silently
# without raising the expected errors! (and stack traces, e.g. just SIGSEGV)
# -> perhaps deactivate Numba for DEBUGGING!
DEBUG = True
# DEBUG = False

# Custom typing
SCALAR = Union[float, int, np.integer, np.floating]
INT_DTYPE = np.int_
FLOAT_DTYPE = np.float64  # NOTE: numpy.float_ is deprecated in NumPy v2.0
B_DTYPE = np.bool_

DEFAULT_DOMAIN = np.array([-1, 1])

# --- Numba types. Must match the Numpy dtypes

# Floats
FLOAT = numba.from_dtype(FLOAT_DTYPE)
# F_TYPE = f8
F_1D = FLOAT[:]
F_2D = FLOAT[:, :]
F_3D = FLOAT[:, :, :]

# Integers
INT = numba.from_dtype(INT_DTYPE)
I_1D = INT[:]
I_2D = INT[:, :]
UINT32 = numba.from_dtype(np.uint32)
UINT64 = numba.from_dtype(np.uint64)

# Boolean
B_TYPE = numba.b1
B_1D = B_TYPE[:]

DEFAULT_LP_DEG = 2.0

NOT_FOUND = -1  # meaning: exponent vector is not contained

ARRAY = np.ndarray  # type: ignore
# TYPED_LIST = List[ARRAY]
TYPED_LIST = numba.typed.List
INT_TUPLE = Tuple[int, int]
ARRAY_DICT = Dict[Tuple[int, int, int], Optional[ARRAY]]  # type: ignore
TRAFO_DICT = Dict[Tuple[int, int], ARRAY]  # type: ignore
INT_SET = Set[int]
DICT_TRAFO_TYPE = Tuple[TRAFO_DICT, ARRAY]  # type: ignore
FACTORISED_TRAFO_TYPE = Tuple[ARRAY, ARRAY, ARRAY, ARRAY]  # type: ignore
