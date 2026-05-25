# app.py
import streamlit as st
import apps.maxent as maxent

# Centralized page config handling
st.set_page_config(
    page_title="PreliZ",
    page_icon="favicon.ico",
    layout="wide"
)

# 1. Persistent Sidebar
pages = {
    "About PreliZ Apps": None,
    "Maximum Entropy": maxent.run,
}

selection = st.sidebar.radio("Select an option", list(pages.keys()))


# 2. Sidebar for the "About" page 
if selection == "About PreliZ Apps":
    st.sidebar.divider()
    st.sidebar.markdown("""
    ### PreliZ the Python Library
    PreliZ is also available as a [Python library](https://preliz.readthedocs.io)!
    """)

    st.sidebar.divider()
    st.sidebar.markdown("""
    ### Citation
    If you find this tool useful, please consider citing our work:

    Icazatti, A., Abril-Pla, O., Klami, A., & Martin, O. A. (2023). PreliZ: A tool-box for prior elicitation. *Journal of Open Source Software*, 8(89), 5499. https://doi.org/10.21105/joss.05499
    """)

    # Main layout for the About Page
    st.title("PreliZ Interactive Apps")
    st.markdown("""
    Welcome! This dashboard houses web interfaces to help you explore and elicit probability distributions.
    
    ### Available Tools:
    * **Maximum Entropy (`maxent`)**: Find the maximum entropy distribution that places a given probability mass between two bounds.
    
    Select a tool from the sidebar to begin your elicitation workflow.
                
    More tools are in development, so check back soon for updates! In the meantime, you can explore the capabilities of PreliZ, the [PreliZ Python library](https://preliz.readthedocs.io).
    """)

# 3. Route to selected app logic
elif pages[selection] is not None:
    pages[selection]()
else:
    st.title(selection)
    st.info("This application module is under construction. Check back soon!")