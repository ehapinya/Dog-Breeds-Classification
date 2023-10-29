import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import cv2
from tensorflow.keras.models import Sequential, Dropout, GlobalAveragePooling2D, Dense
from glob import glob
from keras.optimizers import Adam

dog_names = ['Affenpinscher', 'Afghan_hound',  'Airedale_terrier',  'Akita',  'Alaskan_malamute',  'American_eskimo_dog',  'American_foxhound',  'American_staffordshire_terrier',  'American_water_spaniel',  'Anatolian_shepherd_dog',  'Australian_cattle_dog',  'Australian_shepherd',  'Australian_terrier',  'Basenji',  'Basset_hound',  'Beagle',  'Bearded_collie',  'Beauceron',  'Bedlington_terrier',  'Belgian_malinois',  'Belgian_sheepdog',  'Belgian_tervuren',  'Bernese_mountain_dog',  'Bichon_frise',  'Black_and_tan_coonhound',  'Black_russian_terrier',  'Bloodhound',  'Bluetick_coonhound',  'Border_collie',  'Border_terrier',  'Borzoi',  'Boston_terrier',  'Bouvier_des_flandres',  'Boxer',  'Boykin_spaniel',  'Briard',  'Brittany',  'Brussels_griffon',  'Bull_terrier',  'Bulldog',  'Bullmastiff',  'Cairn_terrier',  'Canaan_dog',  'Cane_corso',  'Cardigan_welsh_corgi',  'Cavalier_king_charles_spaniel',  'Chesapeake_bay_retriever',  'Chihuahua',  'Chinese_crested',  'Chinese_shar-pei',  'Chow_chow',  'Clumber_spaniel',  'Cocker_spaniel',  'Collie',  'Curly-coated_retriever',  'Dachshund',  'Dalmatian',  'Dandie_dinmont_terrier',  'Doberman_pinscher',  'Dogue_de_bordeaux',  'English_cocker_spaniel',  'English_setter',  'English_springer_spaniel',  'English_toy_spaniel',  'Entlebucher_mountain_dog',  'Field_spaniel',  'Finnish_spitz',  'Flat-coated_retriever',  'French_bulldog',  'German_pinscher',  'German_shepherd_dog',  'German_shorthaired_pointer',  'German_wirehaired_pointer',  'Giant_schnauzer',  'Glen_of_imaal_terrier',  'Golden_retriever',  'Gordon_setter',  'Great_dane',  'Great_pyrenees',  'Greater_swiss_mountain_dog',  'Greyhound',  'Havanese',  'Ibizan_hound',  'Icelandic_sheepdog',  'Irish_red_and_white_setter',  'Irish_setter',  'Irish_terrier',  'Irish_water_spaniel',  'Irish_wolfhound',  'Italian_greyhound',  'Japanese_chin',  'Keeshond',  'Kerry_blue_terrier',  'Komondor',  'Kuvasz',  'Labrador_retriever',  'Lakeland_terrier',  'Leonberger',  'Lhasa_apso',  'Lowchen',  'Maltese',  'Manchester_terrier',  'Mastiff',  'Miniature_schnauzer',  'Neapolitan_mastiff',  'Newfoundland',  'Norfolk_terrier',  'Norwegian_buhund',  'Norwegian_elkhound',  'Norwegian_lundehund',  'Norwich_terrier',  'Nova_scotia_duck_tolling_retriever',  'Old_english_sheepdog',  'Otterhound',  'Papillon',  'Parson_russell_terrier',  'Pekingese',  'Pembroke_welsh_corgi',  'Petit_basset_griffon_vendeen',  'Pharaoh_hound',  'Plott',  'Pointer',  'Pomeranian',  'Poodle',  'Portuguese_water_dog',  'Saint_bernard',  'Silky_terrier',  'Smooth_fox_terrier',  'Tibetan_mastiff',  'Welsh_springer_spaniel',  'Wirehaired_pointing_griffon',  'Xoloitzcuintli',  'Yorkshire_terrier']

st.title('Dog Breeds Classification App')
Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
Xception_model.add(Dropout(0.5))
Xception_model.add(Dense(133, activation='softmax'))
Xception_model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
Xception_model.load_weights('weights.best.Xception.hdf5')
ResNet50_model = ResNet50(weights='imagenet')
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

def path_to_tensor(img):
    # Note the change here. The input is now an image not img_path.
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def face_detector(img):
    # Change to accept an image rather than path
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img_arr = np.array(img)
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img, verbose=0))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def Xception_predict_breed(processed_image):
  predicted_vector = Xception_model.predict(processed_image)
  return dog_names[np.argmax(predicted_vector)]

def final_pred_Xception(processed_image):
    if face_detector(Image.fromarray(np.uint8(processed_image[0]))):
        pred_dog_breed = Xception_predict_breed(processed_image)
        return f"This image is a human, this human looks like {pred_dog_breed}."
    elif dog_detector(Image.fromarray(np.uint8(processed_image[0]))):
        pred_dog_breed = Xception_predict_breed(processed_image)
        return f"This image is a dog, its breed is {pred_dog_breed}."
    else:
        return "Error occurred."

if uploaded_file is not None and st.button('Predict'):
    processed_image = path_to_tensor(image)
    prediction = final_pred_Xception(processed_image)
    st.write(f"Prediction: {prediction}") 
