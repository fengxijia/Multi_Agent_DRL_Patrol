"""Small rendering helpers shared by both patrol environments."""

import numpy as np


def grab_rgb_frame(canvas):
    """Return the current Matplotlib canvas as an ``(H, W, 3)`` uint8 array.

    Uses ``buffer_rgba`` (the modern, supported API) and falls back to the
    removed ``tostring_rgb`` for very old Matplotlib versions. This keeps the
    GIF export working on both legacy and current Matplotlib.
    """
    canvas.draw()
    if hasattr(canvas, "buffer_rgba"):
        rgba = np.asarray(canvas.buffer_rgba())
        return rgba[..., :3].copy()
    # Legacy fallback (Matplotlib < 3.8, where tostring_rgb still exists).
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(canvas.get_width_height()[::-1] + (3,))
