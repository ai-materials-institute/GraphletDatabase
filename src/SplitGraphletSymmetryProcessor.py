#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backwards-compatible wrapper for symmetry extraction utilities.

All implementation now lives in ``graphlet_core.py``. This module preserves
legacy imports.
"""

from __future__ import annotations

from graphlet_core import Config, SymmetryFeatureExtractor, build_symmetry_feature_payload_from_cif

__all__ = ["Config", "SymmetryFeatureExtractor", "build_symmetry_feature_payload_from_cif"]
__version__ = "1.0.0"
