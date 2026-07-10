import streamlit as st

def configure_page():
    """
    Configure Streamlit page settings.
    """
    st.set_page_config(
        page_title="Object Detection using YOLOv8",  # Setting page title
        page_icon="🏡",     # Setting page icon
        layout="wide",      # Setting layout to wide
        initial_sidebar_state="expanded"    # Expanding sidebar by default
    )

def select_labels(available_labels):
    """
    Select labels from available options.
    """
    selected_labels = st.multiselect(
        "Select Labels",
        available_labels
    )
    return selected_labels
