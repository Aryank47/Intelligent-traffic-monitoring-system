import streamlit as st

# import harr
import hog

# import rcnn

# -------------------- Main Streamlit Application --------------------


def main():
    st.title(
        "Real-Time Traffic Density Detection using Traditional CV and Deep Learning"
    )
    st.subheader(
        "Traffic Density Detection using Traditional Computer Vision and Deep Learning"
    )

    # Dropdown to select the method
    method = st.selectbox(
        "Select the detection method:",
        ["Select a method", "HOG", "RCNN", "Haar Cascade"],
    )

    # Based on the selection, call the appropriate method
    if method == "HOG":
        st.info("Using HOG + SVM method for vehicle detection.")
        hog.hog_main()  # Call the hog_main method from the hog module

    elif method == "RCNN":
        st.info("Using RCNN method for vehicle detection.")
        st.info("RUNNING RCNN...")  # Call the rcnnmain method from the rcnn module

    elif method == "Haar Cascade":
        st.info("Using Haar Cascade method for vehicle detection.")
        st.info(
            "RUNNING Haar Cascade..."
        )  # Call the harrmain method from the harr modul
    else:
        st.write(
            "Please select a detection method from the dropdown above to run the code."
        )


if __name__ == "__main__":
    main()
