import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from PIL import Image
import os
from fp_insights import main as fp_ins


# Set web page's parameters
st.set_page_config(
    page_title="Evalutation Metrics",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
    )

def read_files():
    """ Create two buttons on the Webpage to open a file dialogue
    that can be used to read the Ground Truths and Prediction files from 
    the user's Computer. This function then returns those files and
    are used in calculate_data function """

    st.title("Select the Required Files")
    # gt_df, pr_df = None, None
    col1, col2 = st.beta_columns([3, 3])  # Create a container with two columns
    gt_container = col1.empty()
    pr_container = col2.empty()
    groundtruths = gt_container.file_uploader("Choose Ground Truth CSV", type="csv")
    predictions = pr_container.file_uploader("Choose Prediction CSV", type="csv")

    return groundtruths, predictions, gt_container, pr_container


@st.cache(persist=True)
def calculate_data(groundtruths, predictions):
    """ Read the ground truth and prediction CSVs with Pandas
    and use them in the main() function (declared as fp_ins)
    to calculate True Positives, False Positives, Missing Detections
    and Extra Detections """

    gt_df = pd.read_csv(groundtruths)
    pr_df = pd.read_csv(predictions)
    st.write(gt_df)
    st.write(pr_df)
    tp_df, fp_df, missing_df, extra_df = fp_ins(gt_df, pr_df)

    return tp_df, fp_df, missing_df, extra_df



def get_dfname():
    """ Creates a sidebar selectbox where the user can choose
    which dataframe to display on the webpage"""

    st.sidebar.title("Select Data to Display")
    files = ["True Positive", "False Positive", "Missing Detections", "Extra Detections"]
    dfname = st.sidebar.selectbox("Choose a dataframe", files)
    return dfname


def get_tp_df(tp):
  
    st.sidebar.title("Select a GT Label")
    gt_classes = ["all_ground_truths"] + list(tp["gt_class"].unique())
    gt_class = st.sidebar.selectbox("Choose a GT class", gt_classes)
    
    st.sidebar.title("Select an Image")
    images = ["all_images"] + list(tp.filename.unique())
    col1, col2 = st.sidebar.beta_columns([4, 1])
    image = col1.selectbox("Check box to write image name instead", images)
    img_name_cbox = col2.checkbox("")
    if img_name_cbox:
        image = st.sidebar.text_input("Enter image name here!", "all_images")

    if gt_class == "all_ground_truths":
        result = tp
    else:
        result = tp.loc[tp['gt_class'] == gt_class]
    
    if image == "all_images":
        pass
    else:
        result = result.loc[result['filename'] == image]

    return result, image

def get_fp_df(fp):
    st.sidebar.title("Select a GT Label")
    gt_classes = ["all_ground_truths"] + list(fp["gt_class"].unique())
    gt_class = st.sidebar.selectbox("Choose a GT class", gt_classes)

    st.sidebar.title("Select a PR Label")
    pr_classes = ["all_predicted"] + list(fp.pr_class.unique())
    pr_class = st.sidebar.selectbox("Choose a PR class", pr_classes)

    st.sidebar.title("Select an Image")
    images = ["all_images"] + list(fp.filename.unique())
    image = st.sidebar.selectbox("Choose an image", images)


    if gt_class == "all_ground_truths":
        result = fp
    else:
        result = fp.loc[fp['gt_class'] == gt_class]

    if pr_class == "all_predicted":
        pass
    else:
        result = result.loc[result['pr_class'] == pr_class]

    if image == "all_images":
        pass
    else:
        result = result.loc[result['filename'] == image]

    return result, image

def get_missing_df(missing_df):
    st.sidebar.title("Select a GT Label")
    gt_classes = ["all_ground_truths"] + list(missing_df.gt_class.unique())
    gt_class = st.sidebar.selectbox("Choose a GT class", gt_classes)

    st.sidebar.title("Select an Image")
    images = ["all_images"] + list(missing_df.filename.unique())
    image = st.sidebar.selectbox("Choose an image", images)


    if gt_class == "all_ground_truths":
        result = missing_df
    else:
        result = missing_df.loc[missing_df['gt_class'] == gt_class]

    if image == "all_images":
        pass
    else:
        result = result.loc[result['filename'] == image]

    return result, image


def get_extra_df(extra_df):
    st.sidebar.title("Select a Class Label")
    classes = ["all_ground_truths"] + list(extra_df['class'].unique())
    class_ = st.sidebar.selectbox("Choose a class", classes)

    st.sidebar.title("Select an Image")
    images = ["all_images"] + list(extra_df.filename.unique())
    image = st.sidebar.selectbox("Choose an image", images)

    st.title("Extra Detections")

    if class_ == "all_ground_truths":
        result = extra_df
    else:
        result = extra_df.loc[extra_df['class'] == class_]

    if image == "all_images":
        pass
    else:
        result = result.loc[result['filename'] == image]

    return result, image    


def run_app(tp, fp, missing_df, extra_df):
    dfname = get_dfname()
    col1, _ = st.beta_columns([4, 1])
    # container = st.beta_container()
    # container = st.empty()
    if dfname == "True Positive":
        col1.subheader("True Positive Data")
        df, image_name = get_tp_df(tp)
    elif dfname == "False Positive":
        col1.subheader("False Positive Data")
        df, image_name = get_fp_df(fp)
    elif dfname == "Missing Detections":
        col1.subheader("Missing Predictions")
        df, image_name = get_missing_df(missing_df) 
    else:
        col1.subheader("Extra Predictions")
        df, image_name = get_extra_df(extra_df)

    col1.write(df)
    show_image = st.sidebar.checkbox("Check to display image", False)
    
    if show_image:      
        img_resize = st.sidebar.slider("Resize Image", min_value=1, max_value=10, value=9, step=1)
        if img_resize > 5:
            img_resize = (10 % img_resize) + 1
        elif img_resize == 5:
            img_resize = 6
        elif img_resize == 4:
            img_resize = 7
        elif img_resize == 3:
            img_resize = 8
        elif img_resize == 2:
            img_resize = 9
        elif img_resize == 1:
            img_resize = 10
        if image_name == "all_images":
            st.exception(RuntimeError("cannot show all_images"))
        else:
            try:
                image = Image.open(os.path.join("images", image_name))
                width, height = image.size
                image = image.resize((width//img_resize, height//img_resize))
                st.image(image, caption = image_name, use_column_width=False)
                # container.empty()
            except:
                 st.error("Image %s not found   " % image_name)


def main():
    program_exec = False
    if program_exec == False:
        groundtruths, predictions, gt_container, pr_container = read_files()
        program_exec = True
    if groundtruths != None and predictions != None: 
        gt_container.empty()
        pr_container.empty()
        tp_df, fp_df, missing_df, extra_df = calculate_data(groundtruths, predictions)
        run_app(tp_df, fp_df, missing_df, extra_df)

if __name__ == '__main__':
    main()