import pandas as pd
import pickle
import sys, os
import warnings
warnings.filterwarnings('ignore')

def getFeatures(course_name):
    arg = str(int(course_name)) + "_features.pkl"
    path = "models"
    fname = os.path.join(path, arg)
    with open(fname, 'rb') as pfile:
        features = pickle.load(pfile)
    return features['features']

def predict(course, df):
    arg = str(int(course)) + "_features.pkl"
    path = "models"
    fname = os.path.join(path, arg)
    with open(fname, 'rb') as pfile:
        features = pickle.load(pfile)
    explanatory_encoder_ = features['explanatory_encoder']
    courses_explanatory = [f for f in features['features'] if f.startswith("grade_")]
    for col_name in courses_explanatory:
        df[col_name] = list(map(explanatory_encoder_.get, df[col_name]))

    major_one_hot = ['major_ECOSTA','major_STA','major_STAMACH']
    df[major_one_hot] = 0.
    my_major = "major_" + df['major']
    df[my_major] = 1.
    df = df.drop(columns=['major'])

    arg = str(int(course)) + ".pkl"
    path = "models"
    fname = os.path.join(path, arg)
    with open(fname, 'rb') as pfile:
        clf = pickle.load(pfile)
    prediction = clf.predict(df)[0]
    proba = clf.predict_proba(df)
    response_encoder_ = features['response_encoder']
    response_encoder_ = {int(v): k for k, v in response_encoder_.items()}
    predicted_grade = response_encoder_[prediction]
    return predicted_grade, proba, df

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         quit()
#     arg = sys.argv[1] + ".pkl"
#     path = "models"
#     fname = os.path.join(path, arg)
#     with open(fname, 'rb') as pfile:
#         clf = pickle.load(pfile)
#     print(clf.feature_importances_)

#     data = [3, 54.0, 0.5, 0.0, 1.0, 4.00, 3.39, 3.12, 3.35, 3.22, 0, 1, 0]
#     predict_df = pd.DataFrame([data], columns=columns)
#     print(predict_df)
#     prediction = clf.predict(predict_df)
#     print(prediction)

#     print(getFeatures(36401))
