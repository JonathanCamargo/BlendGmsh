"""
Unit tests for CoverageReport and build_coverage_report.

Pure Python tests -- no Gmsh required.
"""

import io
import sys
import pytest

from matching_library.coverage import CoverageReport, build_coverage_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_match_results(assignments: dict) -> dict:
    """
    Build a synthetic match_results dict.

    assignments: {surface_tag: group_name_or_None_or_list}
    Accepts a single group name, None, or a list of group names for
    multi-group surfaces.

    All matched surfaces get 8 facets per group, unmatched get 0.
    """
    results = {}
    for tag, grp in assignments.items():
        if grp is None:
            results[tag] = {
                "groups": {},
                "total_facets": 10,
                "unmatched_facets": 10,
            }
        elif isinstance(grp, list):
            results[tag] = {
                "groups": {g: 8 for g in grp},
                "total_facets": 10,
                "unmatched_facets": 2,
            }
        else:
            results[tag] = {
                "groups": {grp: 8},
                "total_facets": 10,
                "unmatched_facets": 2,
            }
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_coverage_report_group_stats():
    """build_coverage_report returns correct surface counts and facet totals per group."""
    match_results = _make_match_results({
        1: "inlet",
        2: "inlet",
        3: "outlet",
        4: None,
    })
    report = build_coverage_report(match_results, ["inlet", "outlet"])

    assert "inlet" in report.group_stats, "Expected 'inlet' in group_stats"
    assert "outlet" in report.group_stats, "Expected 'outlet' in group_stats"

    inlet_stats = report.group_stats["inlet"]
    assert len(inlet_stats["surfaces"]) == 2, (
        f"Expected 2 surfaces for inlet, got {len(inlet_stats['surfaces'])}"
    )
    assert inlet_stats["matched_facets"] == 16, (
        f"Expected 16 matched facets for inlet (2 * 8), got {inlet_stats['matched_facets']}"
    )
    assert inlet_stats["total_facets"] == 20, (
        f"Expected 20 total facets for inlet (2 * 10), got {inlet_stats['total_facets']}"
    )

    outlet_stats = report.group_stats["outlet"]
    assert len(outlet_stats["surfaces"]) == 1, (
        f"Expected 1 surface for outlet, got {len(outlet_stats['surfaces'])}"
    )


def test_coverage_report_unmatched_surfaces():
    """match_results with unmatched surfaces lists them in unmatched_surfaces."""
    match_results = _make_match_results({
        1: "inlet",
        2: None,
        3: None,
        4: None,
        5: None,
    })
    report = build_coverage_report(match_results, ["inlet"])

    assert len(report.unmatched_surfaces) == 4, (
        f"Expected 4 unmatched surfaces, got {len(report.unmatched_surfaces)}"
    )
    for tag in [2, 3, 4, 5]:
        assert tag in report.unmatched_surfaces, (
            f"Surface {tag} should be in unmatched_surfaces"
        )


def test_zero_coverage_group_detected():
    """Requested group 'missing_group' not matched to any surface -> check_zero_coverage_groups returns it."""
    match_results = _make_match_results({
        1: "inlet",
        2: "outlet",
    })
    report = build_coverage_report(match_results, ["inlet", "outlet", "missing_group"])

    zeros = report.check_zero_coverage_groups(["inlet", "outlet", "missing_group"])
    assert "missing_group" in zeros, (
        f"Expected 'missing_group' in zero-coverage groups, got: {zeros}"
    )
    assert "inlet" not in zeros, "inlet has surfaces, should not appear in zero-coverage list"
    assert "outlet" not in zeros, "outlet has surfaces, should not appear in zero-coverage list"


def test_print_report_output(capsys):
    """print_report() produces output containing group names and facet counts."""
    match_results = _make_match_results({
        1: "inlet",
        2: "outlet",
    })
    report = build_coverage_report(match_results, ["inlet", "outlet"])
    report.print_report()

    captured = capsys.readouterr()
    assert "inlet" in captured.out, "Expected 'inlet' in print_report output"
    assert "outlet" in captured.out, "Expected 'outlet' in print_report output"
    # Facet counts should appear
    assert "8" in captured.out, "Expected facet count '8' in print_report output"


def test_all_requested_groups_present():
    """All requested group names appear in group_stats even if zero surfaces matched."""
    match_results = _make_match_results({
        1: "inlet",
    })
    requested = ["inlet", "outlet", "wall"]
    report = build_coverage_report(match_results, requested)

    for grp in requested:
        assert grp in report.group_stats, (
            f"Expected '{grp}' in group_stats even with zero surfaces"
        )

    # Zero-surface groups should have empty surfaces list and zero facets
    assert report.group_stats["outlet"]["surfaces"] == [], (
        "outlet has no surfaces -- should be empty list"
    )
    assert report.group_stats["outlet"]["matched_facets"] == 0, (
        "outlet has no matches -- matched_facets should be 0"
    )
    assert report.group_stats["wall"]["surfaces"] == []
