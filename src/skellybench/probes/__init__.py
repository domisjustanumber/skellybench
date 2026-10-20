"""Standalone environment probes.

``local`` reads this machine and its Python environment. ``docker`` reads the container that the
process runs in. Each file uses the standard library only and imports no other skellybench module,
so another project can copy or import it alone.
"""

from . import docker, local

__all__ = ["docker", "local"]
