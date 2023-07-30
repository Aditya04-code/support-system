from operator import index
import numpy as np
import joblib
import pandas as pd
import pickle
import streamlit as st
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-repeat: no-repeat;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
add_bg_from_local('C:/Users/adity/OneDrive/Desktop/mp/minor_P/support-system/coding files/Images/bg10.jpg')   
st.markdown("<h1 id='soft_name' style='text-align: center; color: red; font-size: 40px ; text-decoration: underline;'>Smart Rx</h1><br>", unsafe_allow_html=True)
loaded_model = open("C:/Users/adity/OneDrive/Desktop/mp/minor_P/support-system/coding files/best.pkl","rb")
classifer = joblib.load(loaded_model)
st.sidebar.subheader("Upload Symptoms")
Uploaded_symptoms=st.sidebar.file_uploader("Upload Symptoms",type=['csv'],label_visibility="hidden")
st.sidebar.subheader("User input Symptoms")


def prediction(input_data):
    input_data_as_np = np.asarray(input_data)
    input_data_reshape = input_data_as_np.reshape(1,-1)
    prediction = classifer.predict(input_data_reshape)
    return int(prediction)


def user_interface():
    skin_rashes  = st.sidebar.checkbox("skin_rashes",value=False)
    itching = st.sidebar.checkbox("itching")
    skin_rash = st.sidebar.checkbox("skin_rash")
    nodal_skin_eruptions = st.sidebar.checkbox("nodal_skin_eruptions")
    continuous_sneezing = st.sidebar.checkbox("continuous_sneezing")
    shivering = st.sidebar.checkbox("shivering")
    chills = st.sidebar.checkbox("chills")
    joint_pain = st.sidebar.checkbox("joint_pain")
    stomach_pain = st.sidebar.checkbox("stomach_pain")
    acidity = st.sidebar.checkbox("acidity")
    ulcers_on_tongue = st.sidebar.checkbox("ulcers_on_tongue")
    muscle_wasting = st.sidebar.checkbox("muscle_wasting")
    vomiting = st.sidebar.checkbox("vomiting")
    burning_micturition = st.sidebar.checkbox("burning_micturition")
    spotting_urination = st.sidebar.checkbox("spotting_urination")
    fatigue = st.sidebar.checkbox("fatigue")
    weight_gain = st.sidebar.checkbox("weight_gain")
    anxiety = st.sidebar.checkbox("anxiety")
    cold_hands_and_feets = st.sidebar.checkbox("cold_hands_and_feets")
    mood_swings = st.sidebar.checkbox("mood_swings")
    weight_loss = st.sidebar.checkbox("weight_loss")
    restlessness = st.sidebar.checkbox("restlessness")
    lethargy = st.sidebar.checkbox("lethargy")
    patches_in_throat = st.sidebar.checkbox("patches_in_throat")
    irregular_sugar_level = st.sidebar.checkbox("irregular_sugar_level")
    cough = st.sidebar.checkbox("cough")
    high_fever = st.sidebar.checkbox("high_fever")
    sunken_eyes = st.sidebar.checkbox("sunken_eyes")
    breathlessness = st.sidebar.checkbox("breathlessness")
    sweating = st.sidebar.checkbox("sweating")
    dehydration = st.sidebar.checkbox("dehydration")
    indigestion = st.sidebar.checkbox("indigestion")
    headache = st.sidebar.checkbox("headache")
    yellowish_skin = st.sidebar.checkbox("yellowish_skin")
    dark_urine = st.sidebar.checkbox("dark_urine")
    nausea = st.sidebar.checkbox("nausea")
    loss_of_appetite = st.sidebar.checkbox("loss_of_appetite")
    pain_behind_the_eyes = st.sidebar.checkbox("pain_behind_the_eyes")
    back_pain = st.sidebar.checkbox("back_pain")
    constipation = st.sidebar.checkbox("constipation")
    abdominal_pain = st.sidebar.checkbox("abdominal_pain")
    diarrhoea = st.sidebar.checkbox("diarrhoea")
    mild_fever = st.sidebar.checkbox("mild_fever")
    yellow_urine = st.sidebar.checkbox("yellow_urine")
    yellowing_of_eyes = st.sidebar.checkbox("yellowing_of_eyes")
    acute_liver_failure = st.sidebar.checkbox("acute_liver_failure")
    fluid_overload = st.sidebar.checkbox("fluid_overload")
    swelling_of_stomach = st.sidebar.checkbox("swelling_of_stomach")
    swelled_lymph_nodes = st.sidebar.checkbox("swelled_lymph_nodes")
    malaise = st.sidebar.checkbox("malaise")
    blurred_and_distorted_vision = st.sidebar.checkbox("blurred_and_distorted_vision")
    phlegm = st.sidebar.checkbox("phlegm")
    throat_irritation = st.sidebar.checkbox("throat_irritation")
    redness_of_eyes = st.sidebar.checkbox("redness_of_eyes")
    sinus_pressure = st.sidebar.checkbox("sinus_pressure")
    runny_nose = st.sidebar.checkbox("runny_nose")
    congestion = st.sidebar.checkbox("congestion")
    chest_pain = st.sidebar.checkbox("chest_pain")
    weakness_in_limbs = st.sidebar.checkbox("weakness_in_limbs")
    fast_heart_rate = st.sidebar.checkbox("fast_heart_rate")
    pain_during_bowel_movements = st.sidebar.checkbox("pain_during_bowel_movements")
    pain_in_anal_region = st.sidebar.checkbox("pain_in_anal_region")
    bloody_stool = st.sidebar.checkbox("bloody_stool")
    irritation_in_anus = st.sidebar.checkbox("irritation_in_anus")
    neck_pain = st.sidebar.checkbox("neck_pain")
    dizziness = st.sidebar.checkbox("dizziness")
    cramps = st.sidebar.checkbox("cramps")
    bruising = st.sidebar.checkbox("bruising")
    obesity = st.sidebar.checkbox("obesity")
    swollen_legs = st.sidebar.checkbox("swollen_legs")
    swollen_blood_vessels = st.sidebar.checkbox("swollen_blood_vessels")
    puffy_face_and_eyes = st.sidebar.checkbox("puffy_face_and_eyes")
    enlarged_thyroid = st.sidebar.checkbox("enlarged_thyroid")
    brittle_nails = st.sidebar.checkbox("brittle_nails")
    swollen_extremeties = st.sidebar.checkbox("swollen_extremeties")
    excessive_hunger = st.sidebar.checkbox("excessive_hunger")
    extra_marital_contacts = st.sidebar.checkbox("extra_marital_contacts")
    drying_and_tingling_lips = st.sidebar.checkbox("drying_and_tingling_lips")
    slurred_speech = st.sidebar.checkbox("slurred_speech")
    knee_pain = st.sidebar.checkbox("knee_pain")
    hip_joint_pain = st.sidebar.checkbox("hip_joint_pain")
    muscle_weakness = st.sidebar.checkbox("muscle_weakness")
    stiff_neck = st.sidebar.checkbox("stiff_neck")
    swelling_joints = st.sidebar.checkbox("swelling_joints")
    movement_stiffness = st.sidebar.checkbox("movement_stiffness")
    spinning_movements = st.sidebar.checkbox("spinning_movements")
    loss_of_balance = st.sidebar.checkbox("loss_of_balance")
    unsteadiness = st.sidebar.checkbox("unsteadiness")
    weakness_of_one_body_side = st.sidebar.checkbox("weakness_of_one_body_side")
    loss_of_smell = st.sidebar.checkbox("loss_of_smell")
    bladder_discomfort = st.sidebar.checkbox("bladder_discomfort")
    foul_smell_of_urine = st.sidebar.checkbox("foul_smell_of_urine")
    continuous_feel_of_urine = st.sidebar.checkbox("continuous_feel_of_urine")
    passage_of_gases = st.sidebar.checkbox("passage_of_gases")
    internal_itching = st.sidebar.checkbox("internal_itching")
    toxic_look_typhos = st.sidebar.checkbox("toxic_look_typhos")
    depression = st.sidebar.checkbox("depression")
    irritability = st.sidebar.checkbox("irritability")
    muscle_pain = st.sidebar.checkbox("muscle_pain")
    altered_sensorium = st.sidebar.checkbox("altered_sensorium")
    red_spots_over_body = st.sidebar.checkbox("red_spots_over_body")
    belly_pain = st.sidebar.checkbox("belly_pain")
    abnormal_menstruation = st.sidebar.checkbox("abnormal_menstruation")
    dischromic_patches = st.sidebar.checkbox("dischromic_patches")
    watering_from_eyes = st.sidebar.checkbox("Watering from eyes")
    increased_appetite = st.sidebar.checkbox("Increased appetite")
    polyuria = st.sidebar.checkbox("Polyuria")
    family_history = st.sidebar.checkbox("Family history")
    mucoid_sputum = st.sidebar.checkbox("Mucoid sputum")
    rusty_sputum = st.sidebar.checkbox("Rusty sputum")
    lack_of_concentration = st.sidebar.checkbox("Lack of concentration")
    visual_disturbances = st.sidebar.checkbox("Visual disturbances")
    receiving_blood_transfusion = st.sidebar.checkbox("Receiving blood transfusion")
    receiving_unsterile_injections = st.sidebar.checkbox("Receiving unsterile injections")
    coma = st.sidebar.checkbox("Coma")
    stomach_bleeding = st.sidebar.checkbox("Stomach bleeding")
    distention_of_abdomen = st.sidebar.checkbox("Distention of abdomen")
    history_of_alcohol_consumption = st.sidebar.checkbox("History of alcohol consumption")
    fluid_overload_1 = st.sidebar.checkbox("Fluid overload.1")
    blood_in_sputum = st.sidebar.checkbox("Blood in sputum")
    prominent_veins_on_calf = st.sidebar.checkbox("Prominent veins on calf")
    palpitations = st.sidebar.checkbox("Palpitations")
    painful_walking = st.sidebar.checkbox("Painful walking")
    pus_filled_pimples = st.sidebar.checkbox("Pus-filled pimples")
    blackheads = st.sidebar.checkbox("Blackheads")
    scurring = st.sidebar.checkbox("Scurring")
    skin_peeling = st.sidebar.checkbox("Skin peeling")
    silver_like_dusting = st.sidebar.checkbox("Silver-like dusting")
    small_dents_in_nails = st.sidebar.checkbox("Small dents in nails")
    inflammatory_nails = st.sidebar.checkbox("Inflammatory nails")
    blister = st.sidebar.checkbox("Blister")
    red_sore_around_nose = st.sidebar.checkbox("Red sore around nose")
    yellow_crust_ooze = st.sidebar.checkbox("Yellow crust ooz   e")

    data ={
        'skin_rash':skin_rashes,
        "watering_from_eyes": watering_from_eyes,
        "increased_appetite": increased_appetite,
        "polyuria": polyuria,
        "family_history": family_history,
        "mucoid_sputum": mucoid_sputum,
        "rusty_sputum": rusty_sputum,
        "lack_of_concentration": lack_of_concentration,
        "visual_disturbances": visual_disturbances,
        "receiving_blood_transfusion": receiving_blood_transfusion,
        "receiving_unsterile_injections": receiving_unsterile_injections,
        "coma": coma,
        "stomach_bleeding": stomach_bleeding,
        "distention_of_abdomen": distention_of_abdomen,
        "history_of_alcohol_consumption": history_of_alcohol_consumption,
        "fluid_overload.1": fluid_overload_1,
        "blood_in_sputum": blood_in_sputum,
        "prominent_veins_on_calf": prominent_veins_on_calf,
        "palpitations": palpitations,
        "painful_walking": painful_walking,
        "pus_filled_pimples": pus_filled_pimples,
        "blackheads": blackheads,
        "scurring": scurring,
        "skin_peeling": skin_peeling,
        "silver_like_dusting": silver_like_dusting,
        "small_dents_in_nails": small_dents_in_nails,
        'itching': itching,
    'nodal_skin_eruptions': nodal_skin_eruptions,
    'continuous_sneezing': continuous_sneezing,
    'shivering': shivering,
    'chills': chills,
    'joint_pain': joint_pain,
    'stomach_pain': stomach_pain,
    'acidity': acidity,
    'ulcers_on_tongue': ulcers_on_tongue,
    'muscle_wasting': muscle_wasting,
    'vomiting': vomiting,
    'burning_micturition': burning_micturition,
    'spotting_ urination': spotting_urination,
    'fatigue': fatigue,
    'weight_gain': weight_gain,
    'anxiety': anxiety,
    'cold_hands_and_feets': cold_hands_and_feets,
    'mood_swings': mood_swings,
    'weight_loss': weight_loss,
    'restlessness': restlessness,
    'lethargy': lethargy,
    'patches_in_throat': patches_in_throat,
    'irregular_sugar_level': irregular_sugar_level,
    'cough': cough,
    'high_fever': high_fever,
    'sunken_eyes': sunken_eyes,
    'breathlessness': breathlessness,
    'sweating': sweating,
    'dehydration': dehydration,
    'indigestion': indigestion,
    'headache': headache,
    'yellowish_skin': yellowish_skin,
    'dark_urine': dark_urine,
    'nausea': nausea,
    'loss_of_appetite': loss_of_appetite,
    'pain_behind_the_eyes': pain_behind_the_eyes,
    'back_pain': back_pain,
    'constipation': constipation,
    'abdominal_pain': abdominal_pain,
    'diarrhoea': diarrhoea,
    'mild_fever': mild_fever,
    'yellow_urine': yellow_urine,
    'yellowing_of_eyes': yellowing_of_eyes,
    'acute_liver_failure': acute_liver_failure,
    'fluid_overload': fluid_overload,
    'swelling_of_stomach': swelling_of_stomach,
    'swelled_lymph_nodes': swelled_lymph_nodes,
    'malaise': malaise,
    'blurred_and_distorted_vision': blurred_and_distorted_vision,
    'phlegm': phlegm,
    'throat_irritation': throat_irritation,
    'redness_of_eyes': redness_of_eyes,
    'sinus_pressure': sinus_pressure,
    'runny_nose': runny_nose,
    'congestion': congestion,
    'chest_pain': chest_pain,
    'weakness_in_limbs': weakness_in_limbs,
    'fast_heart_rate': fast_heart_rate,
    'pain_during_bowel_movements': pain_during_bowel_movements,
    'pain_in_anal_region': pain_in_anal_region,
    'bloody_stool': bloody_stool,
    'irritation_in_anus': irritation_in_anus,
    'neck_pain': neck_pain,
    'dizziness': dizziness,
    'cramps': cramps,
    'bruising': bruising,
    'obesity': obesity,
    'swollen_legs': swollen_legs,
    'swollen_blood_vessels': swollen_blood_vessels,
'puffy_face_and_eyes': puffy_face_and_eyes,
'enlarged_thyroid': enlarged_thyroid,
'brittle_nails': brittle_nails,
'swollen_extremeties': swollen_extremeties,
'excessive_hunger': excessive_hunger,
'extra_marital_contacts': extra_marital_contacts,
'drying_and_tingling_lips': drying_and_tingling_lips,
'slurred_speech': slurred_speech,
'knee_pain': knee_pain,
'hip_joint_pain': hip_joint_pain,
'muscle_weakness': muscle_weakness,
'stiff_neck': stiff_neck,
'swelling_joints': swelling_joints,
'movement_stiffness': movement_stiffness,
'spinning_movements': spinning_movements,
'loss_of_balance': loss_of_balance,
'unsteadiness': unsteadiness,
'weakness_of_one_body_side': weakness_of_one_body_side,
'loss_of_smell': loss_of_smell,
'bladder_discomfort': bladder_discomfort,
'foul_smell_of urine': foul_smell_of_urine,
'continuous_feel_of_urine': continuous_feel_of_urine,
'passage_of_gases': passage_of_gases,
'internal_itching': internal_itching,
'toxic_look_(typhos)': toxic_look_typhos,
'depression': depression,
'irritability': irritability,
'muscle_pain': muscle_pain,
'altered_sensorium': altered_sensorium,
'red_spots_over_body': red_spots_over_body,
'belly_pain': belly_pain,
'abnormal_menstruation': abnormal_menstruation,
'dischromic_patches': dischromic_patches,
'inflammatory_nails': inflammatory_nails,
'blister': blister,
'red_sore_around_nose': red_sore_around_nose,
'yellow_crust_ooze': yellow_crust_ooze
    }
    if(Uploaded_symptoms):
        features = pd.read_csv(Uploaded_symptoms)
    else:
        features = pd.DataFrame(data,index=[0])
    return features
df = user_interface()
# st.write(df)
diseases = {'Fungal infection': 1, 
            'Allergy': 2, 
            'GERD': 3, 
            'Chronic cholestasis': 4,
            'Drug Reaction': 5, 
            'Peptic ulcer diseae': 6, 
            'AIDS': 7, 
            'Diabetes ': 8,
            'Gastroenteritis': 9, 
            'Bronchial Asthma': 10, 
            'Hypertension ': 11, 
            'Migraine': 12,
            'Cervical spondylosis': 13, 
            'Paralysis (brain hemorrhage)': 14, 
            'Jaundice': 15,
            'Malaria': 16, 
            'Chicken pox': 17, 
            'Dengue': 18, 
            'Typhoid': 19, 
            'hepatitis A': 20,
            'Hepatitis B': 21, 
            'Hepatitis C': 22, 
            'Hepatitis D': 23, 
            'Hepatitis E': 24,
            'Alcoholic hepatitis': 25, 
            'Tuberculosis': 26, 
            'Common Cold': 27, 
            'Pneumonia': 28,
            'Dimorphic hemmorhoids(piles)': 29, 
            'Heart attack': 30, 
            'Varicose veins': 31,
            'Hypothyroidism': 32, 
            'Hyperthyroidism': 33, 
            'Hypoglycemia': 34,
            'Osteoarthristis': 35, 
            'Arthritis': 36,
            '(vertigo) Paroymsal  Positional Vertigo': 37, 
            'Acne': 38,
            'Urinary tract infection': 39, 
            'Psoriasis': 40, 
            'Impetigo': 41}
dis_link = {
    1: "https://www.mayoclinic.org/diseases-conditions/fungal-infections/symptoms-causes/syc-20354247",
    2: "https://www.mayoclinic.org/diseases-conditions/allergies/symptoms-causes/syc-20351497",
    3: "https://www.mayoclinic.org/diseases-conditions/gerd/symptoms-causes/syc-20361940",
    4: "https://www.mayoclinic.org/diseases-conditions/cholestasis-of-pregnancy/symptoms-causes/syc-20363257",
    5: "https://www.mayoclinic.org/diseases-conditions/drug-allergy/symptoms-causes/syc-20371835",
    6: "https://www.mayoclinic.org/diseases-conditions/peptic-ulcer/symptoms-causes/syc-20354223",
    7: "https://www.mayoclinic.org/diseases-conditions/hiv-aids/symptoms-causes/syc-20373524",
    8: "https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444",
    9: "https://www.mayoclinic.org/diseases-conditions/viral-gastroenteritis/symptoms-causes/syc-20378847",
    10: "https://www.mayoclinic.org/diseases-conditions/asthma/symptoms-causes/syc-20369653",
    11: "https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/symptoms-causes/syc-20373410",
    12: "https://www.mayoclinic.org/diseases-conditions/migraine-headache/symptoms-causes/syc-20360201",
    13: "https://www.mayoclinic.org/diseases-conditions/cervical-spondylosis/symptoms-causes/syc-20370787",
    14: "https://www.mayoclinic.org/diseases-conditions/stroke/symptoms-causes/syc-20350113",
    15: "https://www.mayoclinic.org/diseases-conditions/jaundice/symptoms-causes/syc-20373711",
    16: "https://www.mayoclinic.org/diseases-conditions/malaria/symptoms-causes/syc-20351184",
    17: "https://www.mayoclinic.org/diseases-conditions/chickenpox/symptoms-causes/syc-20351282",
    18: "https://www.mayoclinic.org/diseases-conditions/dengue-fever/symptoms-causes/syc-203 Dengue fever",
    19: "https://www.mayoclinic.org/diseases-conditions/typhoid-fever/symptoms-causes/syc-20378661",
    20: "https://www.mayoclinic.org/diseases-conditions/hepatitis-a/symptoms-causes/syc-20367007",
    21: "https://www.mayoclinic.org/diseases-conditions/hepatitis-b/symptoms-causes/syc-20366802",
	22: "https://www.mayoclinic.org/diseases-conditions/hepatitis-c/symptoms-causes/syc-20354278",
    23: "https://www.mayoclinic.org/diseases-conditions/hepatitis-d/symptoms-causes/syc-20370809",
    24: "https://www.mayoclinic.org/diseases-conditions/hepatitis-e/symptoms-causes/syc-20373219",
    25: "https://www.mayoclinic.org/diseases-conditions/alcoholic-hepatitis/symptoms-causes/syc-20351388",
    26: "https://www.mayoclinic.org/diseases-conditions/tuberculosis/symptoms-causes/syc-20351250",
    27: "https://www.mayoclinic.org/diseases-conditions/common-cold/symptoms-causes/syc-20351605",
    28: "https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204",
    29: "https://www.mayoclinic.org/diseases-conditions/hemorrhoids/symptoms-causes/syc-20360268",
    30: "https://www.mayoclinic.org/diseases-conditions/heart-attack/symptoms-causes/syc-20373106",
    31: "https://www.mayoclinic.org/diseases-conditions/varicose-veins/symptoms-causes/syc-20350643",
    32: "https://www.mayoclinic.org/diseases-conditions/hypothyroidism/symptoms-causes/syc-20350284",
    33: "https://www.mayoclinic.org/diseases-conditions/hyperthyroidism/symptoms-causes/syc-20373659",
    34: "https://www.mayoclinic.org/diseases-conditions/hypoglycemia/symptoms-causes/syc-20373685",
    35: "https://www.mayoclinic.org/diseases-conditions/osteoarthritis/symptoms-causes/syc-20351925",
    36: "https://www.mayoclinic.org/diseases-conditions/arthritis/symptoms-causes/syc-20350772",
    37: "https://www.mayoclinic.org/diseases-conditions/vertigo/symptoms-causes/syc-20370055",
    38: "https://www.mayoclinic.org/diseases-conditions/acne/symptoms-causes/syc-20368047",
    39: "https://www.mayoclinic.org/diseases-conditions/urinary-tract-infection/symptoms-causes/syc-20353447",
    40: "https://www.mayoclinic.org/diseases-conditions/psoriasis/symptoms-causes/syc-20355840",
    41: "https://www.mayoclinic.org/diseases-conditions/impetigo/symptoms-causes/syc-20352352"
}
diseases_swapped = {v: k for k, v in diseases.items()}
for i in diseases_swapped:
    if prediction(df) == i:
        st.markdown("<h3 style='color: white;'>There are chances the patient may be suffering from the given condition 👨‍⚕️</h3>", unsafe_allow_html=True)
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.write(f"<span id='dis'>{diseases_swapped[i]}<span>", unsafe_allow_html=True)
        st.write(dis_link[i]) 
#st.write(diseases_swapped)
st.sidebar.subheader("Download Symptoms")
st.sidebar.download_button("Download Symptoms",data=df.to_csv(index=False),file_name="symptoms.csv",mime="text/csv")
