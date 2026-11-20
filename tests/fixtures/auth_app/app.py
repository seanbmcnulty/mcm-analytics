import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lib.auth import require_password  # noqa: E402
require_password()

import streamlit as st  # noqa: E402
st.set_page_config(page_title="Fixture", layout="wide")
st.title("SECRET HOME CONTENT")
