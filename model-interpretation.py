# Persiapan-persiapan
import pandas as pd
import numpy as np
import pickle
from distance import DistanceModel
import matplotlib.pyplot as plt
import seaborn as sns

transformer = pickle.load(open("transformer.pkl", "rb")) # Transformasi data: Ketika data user masuk, akan diotak-atik dulu disini sebelum dikasih makan ke machine learning
rf_model = pickle.load(open("mlModel_Revised(2).pkl", "rb")) # Model Random Forest: Buat prediksi Fat Category
distance_model = DistanceModel(k=100) # Model Distance (KNN modified): Buat ngasih rekomendasi makanan dan nutrisi
modified_data = pd.read_csv('final_data_revised.csv') # Data yang udah diolah sebelumnya, udah ada Fat Category nya
real_data = pd.read_csv("FINAL FOOD DATASET/ADDITIONAL/all_group_data.csv")[modified_data.columns[:-1]] # Data asli yang belum diolah, buat ngasih rekomendasi makanan yang lebih sesuai dengan user input
real_data["Fat Category"] = modified_data["Fat Category"] # Biar bisa di filter
food_list = pd.read_csv("FINAL FOOD DATASET/ADDITIONAL/all_group_data.csv").food # List makanan yang ada di dataset asli
real_data_mean = real_data.mean(numeric_only=True) # Mean dari data asli yang belum diolah, buat ngasih rekomendasi makanan yang lebih sesuai dengan user input

# Simulasi user input
user_input = {
    "Fat": 2.5,
    "Caloric Value": 100,
    "Protein": 10.0,
    "Vitamin B2": 0.1,
    "Zinc": 1.0,
    "Vitamin B3": 0.15,
    "Iron": 0.5,
    "Phosphorus": 120.0,
    "Vitamin B1": 0.05,
    "Vitamin B5": 0.08,
    "Potassium": 150.0,
    "Magnesium": 15.0,
    "Calcium": 100.0,
    "Vitamin B6": 0.05,
}

# Data preprocssing user input
user_input = np.array(list(user_input.values())).reshape(1, -1)
user_input_transformed = transformer.transform(user_input)

# Prediksi Fat Category
predicted_fat = rf_model.predict(user_input_transformed)[0]

# Rekomendasi makanan-makanan dan nutrisi
if predicted_fat == "Low fats": # Kasih rekomendasi makanan yang lemaknya agak lebih tinggi

    # FIlter data yang fat category nya Low fats agar bisa ngasih rekomendasi yang gak terlalu ekstrem
    low_fats_modified_data = modified_data[modified_data["Fat Category"] == "Low fats"]
    
    # Cari alternative food (Kalkulasinya pake modified data)
    distance_model.fit(low_fats_modified_data.drop(columns=["Fat Category"]), low_fats_modified_data["Fat Category"])
    indices = distance_model.k_nearest_indices(user_input_transformed[0])
    alternative_indices = indices[:10]
    alternative_food = food_list.loc[alternative_indices]

    # Cari low fat foods yang punya kalori dan fat lebih tinggi dari user input
    filtered_low_fats_real_data = real_data[
        (real_data["Fat"] > user_input[0][0]) &
        (real_data["Caloric Value"] > user_input[0][1]) &
        (real_data["Fat Category"] == "Low fats")
    ]

    available_indices = []
    for idx in indices[::-1]:
        if idx in filtered_low_fats_real_data.index and len(available_indices) < 10:
            available_indices.append(idx)

        if len(available_indices) >= 10:
            break

    recommended_foods = food_list.loc[available_indices]
    filtered_low_fats_real_data = filtered_low_fats_real_data.loc[available_indices] # Data nutrisi dari recommended foods

    # Jaga-jaga kalo misalkan yang di filter itu kurang dari 10, kita ambil dari high fats foods
    if len(filtered_low_fats_real_data) < 10:
        adjusted_distance_model = DistanceModel(k=10-len(filtered_low_fats_real_data))
        high_fats_modified_data = modified_data[modified_data["Fat Category"] == "High fats"]
        adjusted_distance_model.fit(high_fats_modified_data.drop(columns=["Fat Category"]), high_fats_modified_data["Fat Category"])
        adjusted_indices = adjusted_distance_model.k_nearest_indices(user_input_transformed[0])
        filtered_low_fats_real_data = pd.concat([filtered_low_fats_real_data, real_data.loc[adjusted_indices]], axis=0)
        recommended_foods = pd.concat([recommended_foods, food_list.loc[adjusted_indices]], axis=0)

    # Menentukan tingkat signifikansi (seberapa berpengaruh) nutrisi-nutrisi yang ada di recommended foods
    recommended_data_mean = filtered_low_fats_real_data.mean(numeric_only=True)
    mean_difference = (real_data_mean - recommended_data_mean) ** 2

    df_mean_difference = pd.DataFrame(mean_difference, columns=["Mean Difference"])
    df_mean_difference = df_mean_difference.sort_values(by="Mean Difference", ascending=False) # Ambil 5 nutrisi yang paling berpengaruh

    # Buat dikirim ke server
    json_data = {
        "alternative_food": alternative_food.values.tolist(),
        "recommended_foods": recommended_foods.values.tolist(),
        "mean_difference": df_mean_difference.to_dict()
    }

    print(json_data)

elif predicted_fat == "High fats": # Kasih rekomendasi makanan yang lemaknya agak lebih rendah

    # FIlter data yang fat category nya High fats agar bisa ngasih rekomendasi yang gak terlalu ekstrem
    high_fats_modified_data = modified_data[modified_data["Fat Category"] == "High fats"]
    
    # Cari alternative food (Kalkulasinya pake modified data)
    distance_model.fit(high_fats_modified_data.drop(columns=["Fat Category"]), high_fats_modified_data["Fat Category"])
    indices = distance_model.k_nearest_indices(user_input_transformed[0])
    alternative_indices = indices[:10]
    alternative_food = food_list.loc[alternative_indices]

    # Cari high fat foods yang punya kalori dan fat lebih rendah dari user input
    filtered_high_fats_real_data = real_data[
        (real_data["Fat"] < user_input[0][0]) &
        (real_data["Caloric Value"] < user_input[0][1]) &
        (real_data["Fat Category"] == "High fats")
    ]

    available_indices = []
    for idx in indices[::-1]:
        if idx in filtered_high_fats_real_data.index and len(available_indices) < 10:
            available_indices.append(idx)

        if len(available_indices) >= 10:
            break

    recommended_foods = food_list.loc[available_indices]
    filtered_high_fats_real_data = filtered_high_fats_real_data.loc[available_indices] # Data nutrisi dari recommended foods

    # Jaga-jaga kalo misalkan yang di filter itu kurang dari 10, kita ambil dari high fats foods
    if len(filtered_high_fats_real_data) < 10:
        adjusted_distance_model = DistanceModel(k=10-len(filtered_high_fats_real_data))
        low_fats_modified_data = modified_data[modified_data["Fat Category"] == "Low fats"]
        adjusted_distance_model.fit(low_fats_modified_data.drop(columns=["Fat Category"]), low_fats_modified_data["Fat Category"])
        adjusted_indices = adjusted_distance_model.k_nearest_indices(user_input_transformed[0])
        filtered_high_fats_real_data = pd.concat([filtered_high_fats_real_data, real_data.loc[adjusted_indices]], axis=0)
        recommended_foods = pd.concat([recommended_foods, food_list.loc[adjusted_indices]], axis=0)

    # Menentukan tingkat signifikansi (seberapa berpengaruh) nutrisi-nutrisi yang ada di recommended foods
    recommended_data_mean = filtered_high_fats_real_data.mean(numeric_only=True)
    mean_difference = (real_data_mean - recommended_data_mean) ** 2

    df_mean_difference = pd.DataFrame(mean_difference, columns=["Mean Difference"])
    df_mean_difference = df_mean_difference.sort_values(by="Mean Difference", ascending=False) # Ambil 5 nutrisi yang paling berpengaruh

    # Buat dikirim ke server
    json_data = {
        "alternative_food": alternative_food.values.tolist(),
        "recommended_foods": recommended_foods.values.tolist(),
        "mean_difference": df_mean_difference.to_dict()
    }

    print(json_data)