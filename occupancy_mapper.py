"""
occupancy_mapper.py
Range-aware online log-odds occupancy mapping for navigate.py.

Improvements over the original scattered implementation in navigate.py:
  - σ_Z(Z) = Z² · σ_d / (f·B): far-field obstacle weight drops quadratically
  - WLS per-pixel confidence weighting
  - Radial Gaussian smearing: prevents endpoint grid-hopping at range
  - Motion gating: skip / taper log-odds writes during fast turns
  - Free-space hysteresis: require free_confirm_frames before clearing confirmed obstacles
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from stereo_depth import FOCAL_PX, BASELINE_M


@dataclass
class DepthFrame:
    depth: np.ndarray       # H×W float32, metres, 0 = invalid
    confidence: np.ndarray  # H×W float32, [0, 1]
    timestamp: float


@dataclass
class MotionState:
    yaw_rate_deg_s: float = 0.0   # smoothed angular velocity (deg/s)
    speed_cm_s: float = 0.0       # smoothed linear speed (cm/s)


@dataclass
class MapStats:
    frames_used: int = 0
    frames_gated_motion: int = 0
    far_field_rejected_cols: int = 0
    total_cols_processed: int = 0
    last_avg_pixel_conf: float = 1.0

    @property
    def far_rej_pct(self) -> float:
        if self.total_cols_processed == 0:
            return 0.0
        return 100.0 * self.far_field_rejected_cols / self.total_cols_processed

    @property
    def gate_pct(self) -> float:
        total = self.frames_used + self.frames_gated_motion
        if total == 0:
            return 0.0
        return 100.0 * self.frames_gated_motion / total


@dataclass
class MapViews:
    combined_raw: np.ndarray         # bool, True = traversable (for global planner)
    combined_safe: np.ndarray        # bool, True = traversable (inflated, for viz/local)
    combined_safe_local: np.ndarray  # combined_safe & (observed | auv_disk)
    L: np.ndarray                    # float32 log-odds grid (read-only reference)
    stable_occ: np.ndarray           # bool, hysteresis-confirmed occupancy
    observed_mask: np.ndarray        # bool, cells seen by depth rays


def _bresenham(u0: int, v0: int, u1: int, v1: int):
    cells = []
    dx, dy = abs(u1 - u0), abs(v1 - v0)
    sx = 1 if u1 >= u0 else -1
    sy = 1 if v1 >= v0 else -1
    err = dx - dy
    u, v = u0, v0
    while True:
        cells.append((u, v))
        if u == u1 and v == v1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            u += sx
        if e2 < dx:
            err += dx
            v += sy
    return cells


class OccupancyMapper:
    """
    Maintains L, observed_mask, occ_consistency, free_consistency, and produces
    combined_raw / combined_safe / combined_safe_local on demand.
    """

    def __init__(self, grid_config: dict, nav_cfg: dict, transformer):
        img_cfg = grid_config['image_size']
        self._H = img_cfg['height']
        self._W = img_cfg['width']
        self._transformer = transformer
        self._cm_per_px = float(grid_config['resolution']['cm_per_pixel'])

        cfg_log = nav_cfg['log_odds']
        self._l_free    = float(cfg_log['l_free'])
        self._l_occ     = float(cfg_log['l_occ'])
        self._l_min     = float(cfg_log['l_min'])
        self._l_max     = float(cfg_log['l_max'])
        self._l_thresh  = float(cfg_log['l_thresh'])
        self._occ_confirm = max(1, int(cfg_log.get('occ_confirm_frames', 3)))
        # decay=1.0 → no decay (correct for static scenes with ground-truth pose)
        self._decay     = float(cfg_log.get('decay', 1.0))

        self._cfg_depth = nav_cfg['depth']

        sm = nav_cfg.get('sensor_model', {})
        self._sigma_disp      = float(sm.get('stereo_sigma_disp_px', 0.5))
        self._max_sigma_cells = float(sm.get('max_sigma_cells', 4.0))
        self._use_wls_conf    = bool(sm.get('use_wls_confidence', True))
        self._min_px_conf     = float(sm.get('min_pixel_confidence', 0.2))
        self._free_w_scale    = float(sm.get('free_weight_scale', 0.6))
        self._free_confirm    = max(1, int(sm.get('free_confirm_frames', 2)))
        self._omega_taper     = float(sm.get('motion_omega_taper_deg_s', 15.0))
        self._omega_skip      = float(sm.get('motion_omega_skip_deg_s', 30.0))
        self._speed_skip      = float(sm.get('motion_speed_skip_cm_s', 280.0))
        self._taper_scale     = float(sm.get('motion_taper_scale', 0.3))

        plan_cfg = nav_cfg['planner']
        safe_scale = max(0.1, float(plan_cfg.get('safe_inflation_scale', 1.0)))
        self._inflation_px = max(1, int(math.ceil(
            (grid_config['planning']['inflation_radius_m']
             + grid_config['planning'].get('safety_margin_m', 0.0))
            * 100.0 / self._cm_per_px * safe_scale)))
        self._observe_plan_r_px = max(self._inflation_px * 3, 30)
        self._clear_r  = max(2, self._inflation_px // 2)
        self._replan_r = max(3, self._inflation_px // 2)

        # σ_cells(Z) = Z² * _range_denom
        # where Z in metres, result in grid cells
        self._range_denom = self._sigma_disp / (
            FOCAL_PX * BASELINE_M * self._cm_per_px / 100.0)

        H, W = self._H, self._W
        self._static_np = np.ones((H, W), dtype=bool)
        self._L         = np.zeros((H, W), dtype=np.float32)
        self._occ_cons  = np.zeros((H, W), dtype=np.uint8)
        self._free_cons = np.zeros((H, W), dtype=np.uint8)
        self._obs_mask  = np.zeros((H, W), dtype=bool)

        self._stats = MapStats()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, frame: DepthFrame, pose: tuple,
               motion: MotionState) -> MapStats:
        """Ingest one depth frame and update the log-odds grid."""
        auv_x, auv_y, _auv_z, auv_yaw = pose
        auv_u, auv_v = self._transformer.world_to_pixel(auv_x, auv_y)
        H, W = self._H, self._W

        # Time decay (skip multiply when decay=1.0 to avoid touching the whole array)
        if self._decay < 1.0:
            self._L *= self._decay

        # AUV footprint: clear weak marks, mark as observed
        self._clear_patch(auv_u, auv_v, self._clear_r, only_below_thresh=True)
        self._obs_mask[max(0, auv_v - 2):min(H, auv_v + 3),
                       max(0, auv_u - 2):min(W, auv_u + 3)] = True

        # Motion gating
        abs_omega = abs(motion.yaw_rate_deg_s)
        full_gate = (abs_omega > self._omega_skip
                     or motion.speed_cm_s > self._speed_skip)
        if full_gate:
            self._stats.frames_gated_motion += 1
            return self._stats

        taper = 1.0
        if abs_omega > self._omega_taper:
            t = (abs_omega - self._omega_taper) / max(
                self._omega_skip - self._omega_taper, 1e-6)
            taper = max(self._taper_scale,
                        min(1.0, 1.0 - t * (1.0 - self._taper_scale)))

        self._project_and_update(frame, auv_x, auv_y, auv_yaw,
                                 auv_u, auv_v, taper)
        self._stats.frames_used += 1
        return self._stats

    def get_views(self, auv_uv: tuple) -> MapViews:
        """Build and return all derived maps for this frame."""
        auv_u, auv_v = auv_uv
        H, W = self._H, self._W

        raw_occ = self._L > self._l_thresh

        # Occupancy hysteresis (count-up / count-down)
        self._occ_cons[raw_occ] = np.minimum(
            self._occ_cons[raw_occ] + 1, self._occ_confirm)
        not_raw = ~raw_occ
        cv = self._occ_cons[not_raw]
        self._occ_cons[not_raw] = np.where(cv > 0, cv - 1, 0)
        stable_occ = self._occ_cons >= self._occ_confirm

        # Free-space hysteresis: prevent a single clear frame from erasing
        # a confirmed obstacle — require free_confirm_frames consecutive
        # observed-free readings before removing stable_occ.
        if self._free_confirm > 1:
            clearly_free = self._obs_mask & ~raw_occ
            self._free_cons[clearly_free] = np.minimum(
                self._free_cons[clearly_free] + 1, self._free_confirm)
            not_clearly_free = ~clearly_free
            fv = self._free_cons[not_clearly_free]
            self._free_cons[not_clearly_free] = np.where(fv > 0, fv - 1, 0)
            stable_occ = stable_occ & ~(self._free_cons >= self._free_confirm)

        combined_raw = self._static_np & ~stable_occ

        safe_r = max(1, self._inflation_px)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * safe_r + 1, 2 * safe_r + 1))
        occ_safe = cv2.dilate(stable_occ.astype(np.uint8), kernel) > 0
        combined_safe = self._static_np & ~occ_safe

        # Local map: unobserved cells are obstacles, except AUV neighbourhood
        yy, xx = np.ogrid[:H, :W]
        auv_disk = ((xx - auv_u) ** 2 + (yy - auv_v) ** 2
                    <= self._observe_plan_r_px ** 2)
        combined_safe_local = combined_safe & (self._obs_mask | auv_disk)

        return MapViews(
            combined_raw=combined_raw,
            combined_safe=combined_safe,
            combined_safe_local=combined_safe_local,
            L=self._L,
            stable_occ=stable_occ,
            observed_mask=self._obs_mask,
        )

    def clear_around_auv(self, auv_uv: tuple,
                         only_below_thresh: bool = True):
        """Clear log-odds patch around AUV (call before replanning)."""
        self._clear_patch(auv_uv[0], auv_uv[1], self._replan_r,
                          only_below_thresh)

    def hard_clear_around_auv(self, auv_uv: tuple, radius_px: int = 0):
        """Force-reset ALL obstacle data within radius (incl. confirmed obstacles).

        Used as a last-resort recovery when the planner cannot find a path.
        Default radius is 2 × inflation_px + 3 (large enough to break through
        the obstacle shell that accumulated around the AUV near pillars).
        """
        r = radius_px if radius_px > 0 else self._inflation_px * 2 + 3
        auv_u, auv_v = auv_uv
        H, W = self._H, self._W
        v0, v1 = max(0, auv_v - r), min(H, auv_v + r + 1)
        u0, u1 = max(0, auv_u - r), min(W, auv_u + r + 1)
        self._L[v0:v1, u0:u1] = 0.0
        self._occ_cons[v0:v1, u0:u1] = 0
        self._free_cons[v0:v1, u0:u1] = 0

    @property
    def stats(self) -> MapStats:
        return self._stats

    @property
    def inflation_px(self) -> int:
        return self._inflation_px

    @property
    def observe_plan_r_px(self) -> int:
        return self._observe_plan_r_px

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _clear_patch(self, cu: int, cv: int, r: int,
                     only_below_thresh: bool):
        H, W = self._H, self._W
        v0, v1 = max(0, cv - r), min(H, cv + r + 1)
        u0, u1 = max(0, cu - r), min(W, cu + r + 1)
        patch = self._L[v0:v1, u0:u1]
        if only_below_thresh:
            patch[patch < self._l_thresh] = 0.0
        else:
            patch[:] = 0.0
        if self._occ_confirm > 1:
            op = self._occ_cons[v0:v1, u0:u1]
            op[self._L[v0:v1, u0:u1] < self._l_thresh] = 0
            self._occ_cons[v0:v1, u0:u1] = op

    def _project_and_update(self, frame: DepthFrame,
                             auv_x: float, auv_y: float, auv_yaw: float,
                             auv_u: int, auv_v: int,
                             taper: float):
        depth = frame.depth
        conf  = frame.confidence
        cfg   = self._cfg_depth
        H, W  = self._H, self._W
        cm_per_px = self._cm_per_px

        # Morphological opening: remove isolated single-pixel depth hits
        kernel = np.ones((3, 3), np.uint8)
        valid  = (depth > 0).astype(np.uint8)
        valid  = cv2.morphologyEx(valid, cv2.MORPH_OPEN, kernel)
        depth  = depth * valid.astype(np.float32)

        r_min  = float(cfg['min_range_m'])
        r_max  = float(cfg['max_range_m'])
        dH, dW = depth.shape
        v_lo   = int(cfg['row_band_frac'][0] * dH)
        v_hi   = int(cfg['row_band_frac'][1] * dH)
        c_band = cfg.get('col_band_frac', [0.0, 1.0])
        c_lo   = int(c_band[0] * dW)
        c_hi   = int(c_band[1] * dW)
        stride      = int(cfg['col_stride'])
        free_stride = max(stride * 4, 32)
        cx     = dW / 2.0
        yaw_r  = math.radians(auv_yaw)

        min_hits      = int(cfg.get('min_valid_per_col', 4))
        hit_pct       = float(cfg.get('hit_percentile', 20.0))
        near_r        = float(cfg.get('near_obstacle_m', 2.0))
        near_min_hits = int(cfg.get('near_min_valid_per_col', 2))

        col_slice  = depth[v_lo:v_hi, :]
        conf_slice = conf[v_lo:v_hi, :]
        row_band_h = max(v_hi - v_lo, 1)

        l_free_eff = self._l_free * taper
        l_occ_base = self._l_occ * taper

        pixel_conf_accum = []

        for u in range(c_lo, c_hi, stride):
            self._stats.total_cols_processed += 1
            col = col_slice[:, u]
            valid_col = col[(col > r_min) & (col < r_max)]
            hit_pct_frac = float(min(valid_col.size / row_band_h, 1.0))

            # Determine near-field range r ──────────────────────────────────
            if valid_col.size >= min_hits:
                r = float(np.percentile(valid_col, hit_pct))
            else:
                near_valid = valid_col[valid_col < near_r]
                if near_valid.size >= near_min_hits:
                    r = float(np.percentile(near_valid, min(hit_pct, 25.0)))
                else:
                    # No obstacle: optionally clear the ray to r_max
                    if cfg.get('clear_unknown_as_free', False):
                        self._draw_free_ray(
                            auv_x, auv_y, auv_u, auv_v,
                            u, cx, yaw_r, r_max,
                            l_free_eff * self._free_w_scale, H, W)
                    continue

            # Range-aware reliability ────────────────────────────────────────
            sigma_cells = r * r * self._range_denom
            if sigma_cells > self._max_sigma_cells:
                self._stats.far_field_rejected_cols += 1
                # Still mark the ray as free (reduced weight for far field)
                self._draw_free_ray(
                    auv_x, auv_y, auv_u, auv_v,
                    u, cx, yaw_r, r_max,
                    l_free_eff * self._free_w_scale, H, W)
                continue

            w_range = float(
                np.clip(1.0 - sigma_cells / self._max_sigma_cells, 0.2, 1.0))

            # Pixel confidence (from WLS) ────────────────────────────────────
            w_pixel = 1.0
            if self._use_wls_conf:
                valid_flag = (col > r_min) & (col < r_max)
                col_conf_vals = conf_slice[:, u][valid_flag]
                if col_conf_vals.size > 0:
                    w_pixel = float(np.mean(col_conf_vals))
                    pixel_conf_accum.append(w_pixel)
                if w_pixel < self._min_px_conf:
                    # Low confidence: mark free up to r but no obstacle write
                    theta = math.atan2(u - cx, FOCAL_PX)
                    phi   = yaw_r + theta
                    ex = auv_x + r * 100.0 * math.cos(phi)
                    ey = auv_y + r * 100.0 * math.sin(phi)
                    eu, ev = self._transformer.world_to_pixel(ex, ey)
                    for ru, rv in _bresenham(auv_u, auv_v, eu, ev)[:-1]:
                        if 0 <= rv < H and 0 <= ru < W:
                            self._L[rv, ru] = np.clip(
                                self._L[rv, ru] + l_free_eff,
                                self._l_min, self._l_max)
                            self._obs_mask[rv, ru] = True
                    continue

            l_occ_eff = l_occ_base * w_range * w_pixel * (
                0.5 + 0.5 * hit_pct_frac)

            theta = math.atan2(u - cx, FOCAL_PX)
            phi   = yaw_r + theta
            ex = auv_x + r * 100.0 * math.cos(phi)
            ey = auv_y + r * 100.0 * math.sin(phi)
            eu, ev = self._transformer.world_to_pixel(ex, ey)

            # Free ray up to obstacle ────────────────────────────────────────
            for ru, rv in _bresenham(auv_u, auv_v, eu, ev)[:-1]:
                if 0 <= rv < H and 0 <= ru < W:
                    self._L[rv, ru] = np.clip(
                        self._L[rv, ru] + l_free_eff,
                        self._l_min, self._l_max)
                    self._obs_mask[rv, ru] = True

            # Obstacle endpoint: radial Gaussian smearing ────────────────────
            if sigma_cells >= 0.8:
                K = min(int(math.ceil(sigma_cells)), 4)
                ray_du = eu - auv_u
                ray_dv = ev - auv_v
                ray_len = max(math.hypot(ray_du, ray_dv), 1e-6)
                dir_u = ray_du / ray_len
                dir_v = ray_dv / ray_len

                ks = list(range(-K, K + 1))
                raw_w = np.array(
                    [math.exp(-0.5 * (k / sigma_cells) ** 2) for k in ks])
                raw_w /= raw_w.sum()

                for idx, k in enumerate(ks):
                    pu = int(round(eu + k * dir_u))
                    pv = int(round(ev + k * dir_v))
                    if 0 <= pv < H and 0 <= pu < W:
                        self._L[pv, pu] = np.clip(
                            self._L[pv, pu] + float(raw_w[idx]) * l_occ_eff,
                            self._l_min, self._l_max)
                        self._obs_mask[pv, pu] = True
            else:
                if 0 <= ev < H and 0 <= eu < W:
                    self._L[ev, eu] = np.clip(
                        self._L[ev, eu] + l_occ_eff,
                        self._l_min, self._l_max)
                    self._obs_mask[ev, eu] = True

        # Free rays for no-obstacle columns (clear_unknown_as_free) ─────────
        if cfg.get('clear_unknown_as_free', False):
            for u in range(c_lo, c_hi, free_stride):
                col = col_slice[:, u]
                if not np.any((col > r_min) & (col < r_max)):
                    self._draw_free_ray(
                        auv_x, auv_y, auv_u, auv_v,
                        u, cx, yaw_r, r_max,
                        l_free_eff * self._free_w_scale, H, W)

        if pixel_conf_accum:
            self._stats.last_avg_pixel_conf = float(np.mean(pixel_conf_accum))

    def _draw_free_ray(self, auv_x, auv_y, auv_u, auv_v,
                       u, cx, yaw_r, r_max, l_free_val, H, W):
        theta = math.atan2(u - cx, FOCAL_PX)
        phi   = yaw_r + theta
        ex = auv_x + r_max * 100.0 * math.cos(phi)
        ey = auv_y + r_max * 100.0 * math.sin(phi)
        eu, ev = self._transformer.world_to_pixel(ex, ey)
        for ru, rv in _bresenham(auv_u, auv_v, eu, ev):
            if 0 <= rv < H and 0 <= ru < W:
                self._L[rv, ru] = np.clip(
                    self._L[rv, ru] + l_free_val,
                    self._l_min, self._l_max)
                self._obs_mask[rv, ru] = True
