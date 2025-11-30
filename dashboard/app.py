import streamlit as st

from layout import setup_page
import pages


def main():
    setup_page()

    st.sidebar.title("ML Dashboard")
    page = st.sidebar.radio(
        "Navigation",
        ("Status", "Datasets", "Training", "Inference"),
    )

    if page == "Status":
        pages.page_status()
    elif page == "Datasets":
        pages.page_datasets()
    elif page == "Training":
        pages.page_training()
    elif page == "Inference":
        pages.page_inference()


if __name__ == "__main__":
    main()
