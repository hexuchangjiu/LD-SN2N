"""Selected LoopSN2N diagnostics and illustrative training objectives.

This is an algorithm interface, not the historical experiment/training package.
Torch objectives live in their own modules; NumPy diagnostics remain importable
without importing a network or loading any checkpoint.
"""

from .loop import evaluate_loop, loop_signature, signature_map_from_Y, render_2d_colormap

__all__ = ["evaluate_loop", "loop_signature", "signature_map_from_Y", "render_2d_colormap"]

