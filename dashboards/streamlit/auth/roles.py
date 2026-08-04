"""Role-based access controls for RiskPulse Streamlit pages."""

from __future__ import annotations

import os
from enum import StrEnum


class DashboardRole(StrEnum):
    """Supported dashboard roles."""

    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


PAGE_PERMISSIONS: dict[str, set[DashboardRole]] = {
    "real_time_monitor": {DashboardRole.ADMIN, DashboardRole.ANALYST, DashboardRole.VIEWER},
    "investigation_console": {DashboardRole.ADMIN, DashboardRole.ANALYST},
    "trend_analysis": {DashboardRole.ADMIN, DashboardRole.ANALYST},
    "model_performance": {DashboardRole.ADMIN},
    "alert_management": {DashboardRole.ADMIN},
}

DEFAULT_USER_ROLES: dict[str, DashboardRole] = {
    "admin": DashboardRole.ADMIN,
    "analyst": DashboardRole.ANALYST,
    "viewer": DashboardRole.VIEWER,
}


def parse_role(value: str | None) -> DashboardRole:
    """Parse a role name, defaulting safely to viewer."""
    if not value:
        return DashboardRole.VIEWER
    try:
        return DashboardRole(value.strip().lower())
    except ValueError:
        return DashboardRole.VIEWER


def configured_user_roles() -> dict[str, DashboardRole]:
    """Build username to role mapping from defaults and environment overrides."""
    roles = DEFAULT_USER_ROLES.copy()

    admin_user = os.environ.get("DASHBOARD_ADMIN_USER", "admin")
    analyst_user = os.environ.get("DASHBOARD_ANALYST_USER", "analyst")
    viewer_user = os.environ.get("DASHBOARD_VIEWER_USER")

    roles[admin_user] = parse_role(os.environ.get("DASHBOARD_ADMIN_ROLE", "admin"))
    roles[analyst_user] = parse_role(os.environ.get("DASHBOARD_ANALYST_ROLE", "analyst"))
    if viewer_user:
        roles[viewer_user] = parse_role(os.environ.get("DASHBOARD_VIEWER_ROLE", "viewer"))

    extra_roles = os.environ.get("DASHBOARD_USER_ROLES", "")
    for item in extra_roles.split(","):
        if ":" not in item:
            continue
        username, role_name = item.split(":", 1)
        username = username.strip()
        if username:
            roles[username] = parse_role(role_name)

    return roles


def role_for_user(username: str | None) -> DashboardRole:
    """Return the configured dashboard role for a username."""
    return configured_user_roles().get(username or "", DashboardRole.VIEWER)


def can_access_page(role: DashboardRole | str, page_key: str) -> bool:
    """Return whether a role can access a page key."""
    parsed_role = role if isinstance(role, DashboardRole) else parse_role(str(role))
    return parsed_role in PAGE_PERMISSIONS.get(page_key, set())


def visible_pages_for_role(
    role: DashboardRole | str,
    pages: dict[str, str],
) -> dict[str, str]:
    """Filter a display-label to page-key mapping by role permissions."""
    parsed_role = role if isinstance(role, DashboardRole) else parse_role(str(role))
    return {
        label: page_key
        for label, page_key in pages.items()
        if can_access_page(parsed_role, page_key)
    }
