"""V11 Pattern Detection and Tracking.

Identifies coherent structures in the continuous Lenia field
and tracks them across timesteps with persistent identity.

A "pattern" is a spatially connected region of active cells
with internal correlation structure distinct from background.
This is the emergent boundary from Part 1, Definition 2.12.
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class Pattern:
    """A detected pattern (emergent entity) in the substrate."""
    id: int
    cells: np.ndarray         # (N, 2) array of (row, col) coordinates
    values: np.ndarray        # (N,) cell intensities
    center: np.ndarray        # (2,) mass-weighted centroid
    mass: float               # sum of cell values
    size: int                 # number of active cells
    bbox: Tuple[int, int, int, int]  # (r_min, r_max, c_min, c_max)
    age: int = 0              # timesteps since first detection


def detect_patterns(grid_np, threshold=0.1, min_size=8, max_size=8000):
    """Find coherent patterns via thresholding + connected components.

    This is the minimal pattern detection: cells above threshold
    that form spatially connected groups. The threshold defines
    the "boundary" -- the edge of the information structure.

    Returns list of Pattern objects (without persistent IDs).
    """
    binary = (grid_np > threshold).astype(np.int32)
    labeled, n_features = ndimage.label(binary)

    patterns = []
    for label_id in range(1, n_features + 1):
        cells = np.argwhere(labeled == label_id)
        if not (min_size <= len(cells) <= max_size):
            continue

        values = grid_np[cells[:, 0], cells[:, 1]]
        weights = values / (values.sum() + 1e-10)
        center = np.average(cells.astype(float), axis=0, weights=weights)
        mass = float(values.sum())
        r_min, c_min = cells.min(axis=0)
        r_max, c_max = cells.max(axis=0)

        patterns.append(Pattern(
            id=-1,
            cells=cells,
            values=values,
            center=center,
            mass=mass,
            size=len(cells),
            bbox=(int(r_min), int(r_max), int(c_min), int(c_max)),
        ))

    return patterns


def detect_patterns_mc(grid_mc_np, threshold=0.2, min_size=8, max_size=15000):
    """Find coherent patterns in multi-channel Lenia field.

    Thresholds on the mean across channels (per-channel activity).
    Pattern `values` becomes (N_cells, C) — per-channel values at each cell.
    Pattern `mass` = sum across all channels and cells.

    Higher max_size than single-channel because multi-channel patterns
    tend to be larger (3x the active mass).

    Args:
        grid_mc_np: (C, N, N) numpy array of channel states
    """
    C = grid_mc_np.shape[0]
    aggregate = grid_mc_np.mean(axis=0)  # (N, N) — mean per channel

    binary = (aggregate > threshold).astype(np.int32)
    labeled, n_features = ndimage.label(binary)

    patterns = []
    for label_id in range(1, n_features + 1):
        cells = np.argwhere(labeled == label_id)
        if not (min_size <= len(cells) <= max_size):
            continue

        # Per-channel values at pattern cells
        values_mc = grid_mc_np[:, cells[:, 0], cells[:, 1]].T  # (N_cells, C)
        values_agg = aggregate[cells[:, 0], cells[:, 1]]  # (N_cells,) for weighting

        weights = values_agg / (values_agg.sum() + 1e-10)
        center = np.average(cells.astype(float), axis=0, weights=weights)
        mass = float(values_mc.sum())
        r_min, c_min = cells.min(axis=0)
        r_max, c_max = cells.max(axis=0)

        patterns.append(Pattern(
            id=-1,
            cells=cells,
            values=values_mc,  # (N_cells, C) instead of (N_cells,)
            center=center,
            mass=mass,
            size=len(cells),
            bbox=(int(r_min), int(r_max), int(c_min), int(c_max)),
        ))

    return patterns


class PatternTracker:
    """Maintains persistent identity for patterns across timesteps.

    Uses centroid distance + mass similarity for matching.
    Records history for trajectory-based measurements.
    """

    def __init__(self, max_match_dist=30.0, mass_weight=0.3):
        self.next_id = 0
        self.active: Dict[int, Pattern] = {}
        self.max_match_dist = max_match_dist
        self.mass_weight = mass_weight
        # History: id -> list of (center, mass, size, step)
        self.history: Dict[int, List[dict]] = {}
        # Track births and deaths for analysis
        self.births: List[Tuple[int, int]] = []   # (id, step)
        self.deaths: List[Tuple[int, int]] = []   # (id, step)
        self.current_step = 0

    def update(self, new_patterns, step=None):
        """Match new detections to existing patterns. Returns updated patterns."""
        if step is not None:
            self.current_step = step

        if not self.active:
            for p in new_patterns:
                self._assign_new_id(p)
            return new_patterns

        # Compute cost matrix: centroid distance + mass difference
        old_ids = list(self.active.keys())
        old_list = [self.active[i] for i in old_ids]

        if not old_list or not new_patterns:
            # All old patterns died
            for oid in old_ids:
                self.deaths.append((oid, self.current_step))
            self.active.clear()
            for p in new_patterns:
                self._assign_new_id(p)
            return new_patterns

        n_old, n_new = len(old_list), len(new_patterns)
        costs = np.full((n_old, n_new), np.inf)

        for i, op in enumerate(old_list):
            for j, np_ in enumerate(new_patterns):
                dist = np.linalg.norm(op.center - np_.center)
                mass_diff = abs(op.mass - np_.mass) / (op.mass + 1e-10)
                costs[i, j] = dist + self.mass_weight * mass_diff * 100

        # Greedy matching (lowest cost first)
        matched_old = set()
        matched_new = set()
        flat_order = np.argsort(costs.ravel())

        for flat_idx in flat_order:
            i, j = divmod(int(flat_idx), n_new)
            if costs[i, j] > self.max_match_dist:
                break
            if i in matched_old or j in matched_new:
                continue
            # Match
            old_id = old_ids[i]
            new_patterns[j].id = old_id
            new_patterns[j].age = self.active[old_id].age + 1
            matched_old.add(i)
            matched_new.add(j)

        # Deaths: unmatched old
        for i, oid in enumerate(old_ids):
            if i not in matched_old:
                self.deaths.append((oid, self.current_step))

        # Births: unmatched new
        for j, p in enumerate(new_patterns):
            if j not in matched_new:
                self._assign_new_id(p)

        # Update active set and record history
        self.active = {}
        for p in new_patterns:
            self.active[p.id] = p
            self._record(p)

        return new_patterns

    def _assign_new_id(self, p):
        p.id = self.next_id
        p.age = 0
        self.next_id += 1
        self.active[p.id] = p
        self.history[p.id] = []
        self.births.append((p.id, self.current_step))
        self._record(p)

    def _record(self, p):
        if p.id not in self.history:
            self.history[p.id] = []
        self.history[p.id].append({
            'step': self.current_step,
            'center': p.center.copy(),
            'mass': p.mass,
            'size': p.size,
            'values': p.values.copy(),
            'cells': p.cells.copy(),
        })

    def get_mass_trajectory(self, pid):
        """Get mass over time for a pattern."""
        if pid not in self.history:
            return np.array([])
        return np.array([h['mass'] for h in self.history[pid]])

    def get_center_trajectory(self, pid):
        """Get centroid over time for a pattern."""
        if pid not in self.history:
            return np.array([]).reshape(0, 2)
        return np.array([h['center'] for h in self.history[pid]])

    def get_longest_lived(self, n=5):
        """Return IDs of the n longest-lived patterns."""
        lifetimes = {pid: len(hist) for pid, hist in self.history.items()}
        sorted_ids = sorted(lifetimes, key=lifetimes.get, reverse=True)
        return sorted_ids[:n]

    def survival_stats(self):
        """Summary statistics on pattern lifetimes."""
        lifetimes = np.array([len(h) for h in self.history.values()])
        if len(lifetimes) == 0:
            return {'n_total': 0}
        return {
            'n_total': len(lifetimes),
            'n_active': len(self.active),
            'mean_lifetime': float(lifetimes.mean()),
            'max_lifetime': int(lifetimes.max()),
            'median_lifetime': float(np.median(lifetimes)),
        }
