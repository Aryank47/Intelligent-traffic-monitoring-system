import streamlit as st
import os
import hog
import rcnn
# -------------------- Main Streamlit Application --------------------


def main():
    st.title("Real-Time Traffic Density Detection using HOG+SVM Vehicle Detector")
    st.markdown(
        """
    **For single images**: This code uses a multi-scale sliding window approach (no background subtraction).  
    **For videos**: It uses background subtraction + HOG+SVM for moving objects.
    """
    )
    hog.hogmain()
    # rcnn.rcnnmain()



if __name__ == "__main__":
    main()