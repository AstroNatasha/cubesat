"""Mission constraint utilities for CubeSat federated learning simulation.

All physical quantities follow SI conventions:
  link_rate_kbps           → kilobits per second
  contact_window_minutes   → minutes per ground-station pass
  contacts_per_day         → passes per day
  *_MB                     → megabytes (1 MB = 1 000 000 bytes)
"""

from __future__ import annotations


def daily_link_capacity_mb(
    link_rate_kbps: float,
    contact_window_minutes: float,
    contacts_per_day: int,
) -> float:
    """Total downlink capacity per day in MB."""
    bytes_per_sec    = link_rate_kbps * 1_000 / 8
    secs_per_contact = contact_window_minutes * 60
    return bytes_per_sec * secs_per_contact * contacts_per_day / 1e6


def downlink_time_minutes(payload_mb: float, link_rate_kbps: float) -> float:
    """Minutes required to transmit payload_mb at link_rate_kbps."""
    bits = payload_mb * 1e6 * 8
    return bits / (link_rate_kbps * 1_000) / 60


def days_to_transmit(payload_mb: float, capacity_mb_per_day: float) -> float:
    """Days to transmit payload_mb at a given daily downlink capacity."""
    return payload_mb / capacity_mb_per_day if capacity_mb_per_day > 0 else float("inf")


def max_rounds_from_budget(budget_mb: float, comm_per_round_mb: float) -> int:
    """Maximum complete FL rounds achievable within budget."""
    if comm_per_round_mb <= 0:
        return 0
    return int(budget_mb / comm_per_round_mb)


def compute_mission_metrics(
    *,
    used_comm_mb: float,
    comm_budget_mb: float,
    link_rate_kbps: float,
    contact_window_minutes: float,
    contacts_per_day: int,
    rounds_completed: int,
    rounds_requested: int,
) -> dict:
    """Compute all mission-derived quantities for logging."""
    cap_mb  = daily_link_capacity_mb(link_rate_kbps, contact_window_minutes, contacts_per_day)
    dl_min  = downlink_time_minutes(used_comm_mb, link_rate_kbps)
    days_tx = days_to_transmit(used_comm_mb, cap_mb)

    return {
        "used_communication_MB":           round(used_comm_mb, 4),
        "remaining_communication_MB":      round(max(0.0, comm_budget_mb - used_comm_mb), 4),
        "budget_exhausted":                used_comm_mb >= comm_budget_mb,
        "feasible_under_budget":           used_comm_mb <= comm_budget_mb,
        "rounds_completed":                rounds_completed,
        "rounds_requested":                rounds_requested,
        "daily_link_capacity_MB":          round(cap_mb, 4),
        "estimated_downlink_time_minutes": round(dl_min, 2),
        "estimated_days_to_transmit":      round(days_tx, 2),
    }
