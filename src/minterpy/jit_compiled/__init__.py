"""
This subpackage collects all just-in-time compiled codes for Minterpy.

Minterpy accelerates numerous performance-critical internal functions by
just-in-time (JIT) compilation using the `Numba <http://numba.pydata.org>`_
package.

Many if not all JIT-compiled codes are decorated by ``njit``
instead of ``jit``.
This means we already knows and specify in advance the input and output
requirements of the compiled codes.
``njit`` is less flexible than ``jit``, but would in most cases be faster and
memory-efficient.
"""