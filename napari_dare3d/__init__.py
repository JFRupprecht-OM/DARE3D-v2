"""napari-dare3d: napari plugin for DARE3D division-axis inference.

Kept intentionally light — it does NOT import :mod:`napari_dare3d._api` (which
pulls torch + dare3d) nor the Qt widget, so importing the package for the npe2
manifest stays cheap.

Reference
---------
Method paper — cite as **Karpinski et al.** (bioRxiv 2024):
https://www.biorxiv.org/content/10.1101/2024.02.05.578987v2

License: MIT — see the ``LICENSE`` file in the plugin folder.
"""

__version__ = "0.0.1"
__license__ = "MIT"
__citation__ = (
    "Karpinski et al., bioRxiv 2024. "
    "https://www.biorxiv.org/content/10.1101/2024.02.05.578987v2"
)
