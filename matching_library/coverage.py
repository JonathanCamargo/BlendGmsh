"""
Coverage report: per-group match statistics and zero-coverage detection.

Provides:
  CoverageReport -- dataclass with match statistics
  build_coverage_report(match_results, requested_groups) -> CoverageReport

Usage:
  report = build_coverage_report(match_results, list(groups.keys()))
  zeros = report.check_zero_coverage_groups(list(groups.keys()))
  if zeros:
      raise RuntimeError(f"Groups with zero matched surfaces: {zeros}")
  report.print_report()
"""

from dataclasses import dataclass, field


@dataclass
class CoverageReport:
    """
    Per-group surface match statistics.

    Attributes
    ----------
    group_stats:
        {group_name: {
            'surfaces': list[int],       -- surface tags assigned to this group
            'matched_facets': int,        -- sum of matched_facets across surfaces
            'total_facets': int,          -- sum of total_facets across surfaces
        }}
    unmatched_surfaces:
        Surface tags where no group matched.
    total_boundary_facets:
        Total facets across all boundary surfaces.
    """

    group_stats: dict = field(default_factory=dict)
    unmatched_surfaces: list = field(default_factory=list)
    total_boundary_facets: int = 0

    def check_zero_coverage_groups(self, requested_groups: list) -> list:
        """
        Return list of group names with zero matched surfaces.

        A group has zero coverage if it does not appear in group_stats or its
        surfaces list is empty. The caller should raise an error if this is non-empty.

        Parameters
        ----------
        requested_groups:
            List of group names that were requested in the JSON input.

        Returns
        -------
        list[str]
            Group names with zero matched surfaces.
        """
        return [
            g for g in requested_groups
            if g not in self.group_stats or len(self.group_stats[g]["surfaces"]) == 0
        ]

    def print_report(self) -> None:
        """Print human-readable per-group match statistics."""
        print(f"Coverage Report: {self.total_boundary_facets} total boundary facets")
        for gname, stats in self.group_stats.items():
            if self.total_boundary_facets > 0:
                pct = 100.0 * stats["matched_facets"] / self.total_boundary_facets
            else:
                pct = 0.0
            print(
                f"  {gname}: {len(stats['surfaces'])} surfaces, "
                f"{stats['matched_facets']} matched facets ({pct:.1f}%)"
            )
        if self.unmatched_surfaces:
            print(f"  _untagged: {len(self.unmatched_surfaces)} unmatched surfaces")


def build_coverage_report(match_results: dict, requested_groups: list) -> "CoverageReport":
    """
    Build a CoverageReport from matcher output.

    Supports multi-group results: a surface can appear in multiple groups'
    stats when groups overlap.

    Parameters
    ----------
    match_results:
        {surface_tag: {'groups': {name: facet_count, ...},
                       'total_facets': int, 'unmatched_facets': int}}
    requested_groups:
        List of group names from the bc_groups JSON (all requested, even if unmatched).

    Returns
    -------
    CoverageReport
    """
    group_stats: dict = {}
    unmatched_surfaces: list = []
    total_boundary_facets: int = 0

    for surf_tag, result in match_results.items():
        total_boundary_facets += result["total_facets"]
        groups = result["groups"]

        if groups:
            for grp, facet_count in groups.items():
                if grp not in group_stats:
                    group_stats[grp] = {
                        "surfaces": [],
                        "matched_facets": 0,
                        "total_facets": 0,
                    }
                group_stats[grp]["surfaces"].append(surf_tag)
                group_stats[grp]["matched_facets"] += facet_count
                group_stats[grp]["total_facets"] += result["total_facets"]
        else:
            unmatched_surfaces.append(surf_tag)

    # Ensure all requested groups appear in group_stats (even with zero coverage)
    for grp in requested_groups:
        if grp not in group_stats:
            group_stats[grp] = {
                "surfaces": [],
                "matched_facets": 0,
                "total_facets": 0,
            }

    return CoverageReport(
        group_stats=group_stats,
        unmatched_surfaces=unmatched_surfaces,
        total_boundary_facets=total_boundary_facets,
    )
