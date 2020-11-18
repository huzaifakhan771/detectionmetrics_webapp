import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk


def iou(boxA, boxB):
    # if boxes dont intersect
    if _boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = _getIntersectionArea(boxA, boxB)
    union = _getUnionAreas(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    assert iou >= 0
    return iou

# boxA = (Ax1,Ay1,Ax2,Ay2)
# boxB = (Bx1,By1,Bx2,By2)
def _boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def _getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

def _getUnionAreas(boxA, boxB, interArea=None):
    area_A = _getArea(boxA)
    area_B = _getArea(boxB)
    if interArea is None:
        interArea = _getIntersectionArea(boxA, boxB)
    return float(area_A + area_B - interArea)

def _getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

# @st.cache(persist=True)
def main(gt_dataframe, pr_dataframe):
    score_th = 0.20
    # pr_dataframe = pd.read_csv('inputs/detections.csv')

    gt_dataframe['gt_bbox']= gt_dataframe[['xmin','ymin','xmax','ymax']].values.tolist()
    pr_dataframe['pr_bbox']= pr_dataframe[['xmin','ymin','xmax','ymax']].values.tolist()
    
    gt_dataframe = gt_dataframe.drop(['width','height','xmin','ymin','xmax','ymax'],axis=1)
    try:
        pr_dataframe = pr_dataframe.drop(['score', 'xmin','ymin','xmax','ymax'],axis=1)
    except:
        pr_dataframe = pr_dataframe.drop(['width', 'height', 'xmin','ymin','xmax','ymax'],axis=1)
    
    classes = gt_dataframe['class'].unique()
    
    df = pd.DataFrame(columns=['filename','gt_class','pr_class','gt_bbox','pr_bbox'])
    missing_df = pd.DataFrame(columns=['filename','gt_class','gt_bbox'])
    extra_df = pr_dataframe.copy()

    for _class in classes:
        gt_df = gt_dataframe.loc[(gt_dataframe["class"] == _class)]
        #pr = pr_dataframe.loc[(pr_dataframe["class"] == _class)]
        filenames = gt_df['filename'].unique()
        for filename in filenames:
            gt = gt_df.loc[(gt_df['filename'] == filename)]
            pr = pr_dataframe.loc[(pr_dataframe["filename"] == filename)]
            indexes = pr.index

            for gt_index,gt_row in gt.iterrows():
                iou_list = []
                for pr_index,pr_row in pr.iterrows():
                    score = iou(gt_row['gt_bbox'],pr_row['pr_bbox'])
                    iou_list.append(score)

                max_value = max(iou_list)
                if max_value > score_th:
                    index = indexes[iou_list.index(max_value)]
                    df = df.append({'filename':filename,'gt_class':gt['class'][index],'pr_class':pr['class'][index],
                        'gt_bbox': gt['gt_bbox'][index], 'pr_bbox':pr['pr_bbox'][index]},ignore_index=True)
                    st.write("here")
                    # print (index)
                    # print (ext)
                    # print (index)
                    extra_df = extra_df.drop([index])
                else:
                    print ('--------- missing --------------')
                    missing_df = missing_df.append({'filename':filename,'gt_class':gt['class'][gt_index],'gt_bbox': gt['gt_bbox'][gt_index]},ignore_index=True)

        
            
    tp = df.loc[df['gt_class'] == df['pr_class']]
    fp = df.loc[df['gt_class'] != df['pr_class']]

    return tp, fp, missing_df, extra_df

def get_dfname():
    st.sidebar.title("Select Data to Display")
    files = ["True Positive", "False Positive", "Missing Detections", "Extra Detections"]
    dfname = st.sidebar.selectbox("Choose a dataframe", files)
    return dfname

def get_tp_df(tp):
    st.title("True Positive Data")
    
    st.sidebar.title("Select a GT Label")
    gt_classes = ["all_ground_truths"] + list(tp.gt_class.unique())
    gt_class = st.sidebar.selectbox("Choose a GT class", gt_classes)
    
    st.sidebar.title("Select an Image")
    images = ["all_images"] + list(tp.filename.unique())
    image = st.sidebar.selectbox("Choose an image", images)

    if gt_class == "all_ground_truths":
        result = tp
    else:
        result = tp.loc[tp['gt_class'] == gt_class]
    
    if image == "all_images":
        pass
    else:
        result = result.loc[result['filename'] == image]

    return result

def get_fp_df(fp):
    st.sidebar.title("Select a GT Label")
    gt_classes = ["all_ground_truths"] + list(fp.gt_class.unique())
    gt_class = st.sidebar.selectbox("Choose a GT class", gt_classes)

    st.sidebar.title("Select a PR Label")
    pr_classes = ["all_predicted"] + list(fp.pr_class.unique())
    pr_class = st.sidebar.selectbox("Choose a PR class", pr_classes)

    st.sidebar.title("Select an Image")
    images = ["all_images"] + list(fp.filename.unique())
    image = st.sidebar.selectbox("Choose an image", images)


    st.title("False Positive Data")

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

    return result

def get_missing_df(missing_df):
    st.sidebar.title("Select a GT Label")
    gt_classes = ["all_ground_truths"] + list(missing_df.gt_class.unique())
    gt_class = st.sidebar.selectbox("Choose a GT class", gt_classes)

    st.sidebar.title("Select an Image")
    images = ["all_images"] + list(missing_df.filename.unique())
    image = st.sidebar.selectbox("Choose an image", images)

    st.title("Missing Detections")

    if gt_class == "all_ground_truths":
        result = missing_df
    else:
        result = missing_df.loc[missing_df['gt_class'] == gt_class]

    if image == "all_images":
        pass
    else:
        result = result.loc[result['filename'] == image]

    return result


def get_extra_df(extra_df):
    st.sidebar.title("Select a Class Label")
    print(extra_df)
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

    return result    

def run_app(tp, fp, missing_df, extra_df):
    dfname = get_dfname()
    if dfname == "True Positive":
        st.write(get_tp_df(tp))
    elif dfname == "False Positive":
        st.write(get_fp_df(fp))
    elif dfname == "Missing Detections":
        st.write(get_missing_df(missing_df))
    else:
        st.write(get_extra_df(extra_df))

# tp.to_csv('CSVs/tp_right_classified.csv')
# fp.to_csv('CSVs/fp_wrongly_classified.csv')
# missing_df.to_csv('CSVs/fn_missing_detections.csv')
# extra_df.to_csv('CSVs/fp_extra_detections.csv')



if __name__ == '__main__':
    st.set_page_config(
    page_title="Evalutation Metrics",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
    )
    gt_csv = st.file_uploader("Choose a Grount Truth CSV")
    pr_csv = st.file_uploader("Choose a Detection CSV")
    from io import StringIO

    if gt_csv != None and pr_csv != None: 
        gt_dataframe = pd.read_csv(gt_csv)
        pr_dataframe = pd.read_csv(pr_csv)
        tp, fp, missing_df, extra_df = main(gt_dataframe, pr_dataframe)
        run_app(tp, fp, missing_df, extra_df)
    
    # uploaded_file = st.file_uploader("Choose a CSV file")
    # if uploaded_file is not None:
    #     df = pd.read_csv(uploaded_file)
    #     del df["Unnamed: 0"] 
    #     st.write(df)
    # if uploaded_file is not None:
        # bytes_data = uploaded_file.read()
        # st.write(bytes_data)

        # To convert to a string based IO:
        # stringio = StringIO(uploaded_file.decode("utf-8"))
        # st.write(stringio)

        # To read file as string:
        # string_data = stringio.read()
        # st.write(string_data)
    # else: 
        # uploaded_file.seek(0)
        

#     st.text_area('Text to analyze', '''
# ...     It was the best of times, it was the worst of times, it was
# ...     the age of wisdom, it was the age of foolishness, it was
# ...     the epoch of belief, it was the epoch of incredulity, it
# ...     was the season of Light, it was the season of Darkness, it
# ...     was the spring of hope, it was the winter of despair, (...)
# ...     ''')

    # if st.button("Upload Image"):
        # st.write("please select an image")

    # options = st.multiselect('What are your favorite colors',['Green', 'Yellow', 'Red', 'Blue'],['Yellow', 'Red'])
    
    # img_name = st.sidebar.text_input("Enter the image name", "img_2e99.jpg")
    # st.write(img_name)