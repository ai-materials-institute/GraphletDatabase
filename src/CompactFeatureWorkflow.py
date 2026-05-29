#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backwards-compatible wrapper for the centralized graphlet workflow module.

All implementation now lives in ``graphlet_core.py``. This file is retained as
an import/CLI compatibility layer.
"""

from __future__ import annotations

from graphlet_core import *  # noqa: F401,F403
from graphlet_core import compact_workflow_cli_main
from graphlet_core import __all__  # re-export public API


def main() -> None:
    compact_workflow_cli_main()


if __name__ == "__main__":
    main()
