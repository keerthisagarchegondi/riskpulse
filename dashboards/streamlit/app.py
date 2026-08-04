"""RiskPulse Streamlit Dashboard — application entry point.

Provides multi-page navigation, session state management,
auto-refresh (30-second intervals), and basic authentication.

Run with:
    streamlit run dashboards/streamlit/app.py
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import sys
from pathlib import Path

import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so relative imports resolve correctly
# when launched via ``streamlit run dashboards/streamlit/app.py``.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dashboards.streamlit.auth.roles import can_access_page, role_for_user, visible_pages_for_role  # noqa: E402
from dashboards.streamlit.pages import (  # noqa: E402
    alert_management,
    investigation_console,
    model_performance,
    real_time_monitor,
    trend_analysis,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("riskpulse.dashboard")

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RiskPulse — Fraud Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
_CSS_PATH = Path(__file__).parent / "static" / "styles.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def _hash_password(password: str) -> str:
    """SHA-256 hash for password comparison (constant-time)."""
    return hashlib.sha256(password.encode()).hexdigest()


_USERS: dict[str, str] = {
    os.environ.get("DASHBOARD_ADMIN_USER", "admin"): _hash_password(
        os.environ.get("DASHBOARD_ADMIN_PASSWORD", "riskpulse2024!")
    ),
    os.environ.get("DASHBOARD_ANALYST_USER", "analyst"): _hash_password(
        os.environ.get("DASHBOARD_ANALYST_PASSWORD", "analyst2024!")
    ),
}
if os.environ.get("DASHBOARD_VIEWER_USER"):
    _USERS[os.environ["DASHBOARD_VIEWER_USER"]] = _hash_password(
        os.environ.get("DASHBOARD_VIEWER_PASSWORD", "viewer2024!")
    )


def _check_credentials(username: str, password: str) -> bool:
    expected_hash = _USERS.get(username)
    if expected_hash is None:
        return False
    return hmac.compare_digest(expected_hash, _hash_password(password))


def _login_form() -> bool:
    """Render login form; return True when authenticated."""
    if st.session_state.get("authenticated"):
        return True

    st.markdown(
        "<div style='text-align:center;margin-top:60px'>"
        "<h1>🛡️ RiskPulse</h1>"
        "<p style='color:#95a5a6'>Fraud Detection & Risk Monitoring Platform</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if _check_credentials(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["role"] = role_for_user(username).value
                st.rerun()
            else:
                st.error("Invalid username or password.")
    return False


# ---------------------------------------------------------------------------
# Database engine (cached singleton)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_engine() -> Engine:
    """Create a synchronous SQLAlchemy engine for dashboard queries."""
    host = os.environ.get("RISKPULSE_DB_HOST", "localhost")
    port = os.environ.get("RISKPULSE_DB_PORT", "5432")
    name = os.environ.get("RISKPULSE_DB_NAME", "riskpulse")
    user = os.environ.get("RISKPULSE_DB_USER", "riskpulse")
    password = os.environ.get("RISKPULSE_DB_PASSWORD", "riskpulse")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    engine = create_engine(
        url,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        pool_recycle=1800,
    )
    logger.info("Database engine created: %s:%s/%s", host, port, name)
    return engine


def _db_healthy(engine: Engine) -> bool:
    """Quick connectivity check."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
_PAGES: dict[str, str] = {
    "📡 Real-Time Monitor": "real_time_monitor",
    "🕵️ Investigation Console": "investigation_console",
    "📈 Trend Analysis": "trend_analysis",
    "Model Performance": "model_performance",
    "Alert Management": "alert_management",
}


def _render_sidebar(engine: Engine) -> str:
    """Render sidebar navigation and metadata; return selected page key."""
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/shield.png",
        width=48,
    )
    st.sidebar.title("RiskPulse")
    st.sidebar.caption(f"Logged in as **{st.session_state.get('username', '')}**")
    role = role_for_user(str(st.session_state.get("username") or ""))
    st.session_state["role"] = role.value
    st.sidebar.caption(f"Role: **{role.value}**")

    visible_pages = visible_pages_for_role(role, _PAGES)
    if not visible_pages:
        st.sidebar.warning("No dashboard pages are available for this role.")
        return ""

    selected = st.sidebar.radio(
        "Navigation",
        options=list(visible_pages.keys()),
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # Database health indicator
    healthy = _db_healthy(engine)
    status_icon = "🟢" if healthy else "🔴"
    st.sidebar.markdown(f"**DB Status:** {status_icon}")

    # Logout
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    return visible_pages[selected]


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
def _setup_auto_refresh(interval_seconds: int = 30) -> None:
    """Inject a JS-based auto-refresh timer."""
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.toggle("Auto-Refresh (30s)", value=True, key="auto_refresh")
    if auto_refresh:
        st.markdown(
            f"""
            <script>
                (function() {{
                    var timer = setTimeout(function() {{
                        window.parent.document.querySelectorAll(
                            'button[kind="header"]'
                        ).forEach(function(el) {{ el.click(); }});
                        window.location.reload();
                    }}, {interval_seconds * 1000});
                }})();
            </script>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not _login_form():
        return

    engine = _get_engine()
    page_key = _render_sidebar(engine)
    _setup_auto_refresh(interval_seconds=30)
    role = role_for_user(str(st.session_state.get("username") or ""))

    if not page_key:
        st.error("Your role does not have access to any dashboard pages.")
        return

    if not can_access_page(role, page_key):
        st.error("You do not have access to this dashboard page.")
        return

    if page_key == "real_time_monitor":
        real_time_monitor.render(engine)
    elif page_key == "investigation_console":
        investigation_console.render(engine)
    elif page_key == "trend_analysis":
        trend_analysis.render(engine)
    elif page_key == "model_performance":
        model_performance.render(engine)
    elif page_key == "alert_management":
        alert_management.render(engine)
    else:
        st.error(f"Unknown page: {page_key}")


if __name__ == "__main__":
    main()
