"""Public Streamlit entry point for gold-free live repository localization."""

from __future__ import annotations

import streamlit as st

from live_repository_tab import (
    render_live_repository_sidebar,
    render_live_repository_tab,
)

st.set_page_config(page_title="GraphFix — Live Repository Localization", layout="wide")
render_live_repository_tab()
render_live_repository_sidebar()
