import numpy as np
import pickle

model = pickle.load(open("student_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

def predict_student_result(
    attendance,
    study_hours,
    internal_marks,
    assignment_completion,
    previous_score
):

    input_data = np.array([[
        attendance,
        study_hours,
        internal_marks,
        assignment_completion,
        previous_score
    ]])

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    predicted_result = label_encoder.inverse_transform(prediction)[0]

    confidence = np.max(probability) * 100

    return predicted_result, confidence