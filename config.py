"""Shared secret/config helper for local .env and Streamlit Cloud secrets."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_secret(key, default=None):
    """
    Read a configuration value.

    Priority:
      1. Streamlit Cloud secrets (st.secrets) if the app is running in Streamlit.
      2. Local environment variables / .env file.
      3. The provided default.
    """
    try:
        import streamlit as st
        if key in st.secrets:
            value = st.secrets[key]
            return str(value) if value is not None else default
    except Exception:
        pass
    return os.getenv(key, default)
