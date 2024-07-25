import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np

# Contoh data (X dan y harus diganti dengan dataset Anda)
import pandas as pd
uber = pd.read_csv("D:\\Matkul Sem 4\\MPML Phyton\\uber.csv")

# Menghapus baris dengan nilai NA pada lebih dari satu kolom
check = ['dropoff_longitude', 'dropoff_latitude']
uber1 = uber.dropna(subset=check)

# Daftar kolom yang akan diperiksa
check1 = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']

# Menghitung jumlah baris yang memiliki nilai 0 pada kolom tertentu
rows_with_zero = uber1[check1].apply(lambda row: (row == 0).any(), axis=1).sum()

# Mengganti nilai 0 dengan NaN pada kolom tertentu
uber1[check1] = uber1[check1].replace(0, np.nan)


# Menghapus baris dengan nilai NA pada lebih dari satu kolom
uber1 = uber1.dropna(subset=check1)

import math

def haversine(lon1, lat1, lon2, lat2):
    # Radius bumi dalam kilometer
    R = 6371

    # Konversi derajat ke radian
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Selisih koordinat
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Rumus Haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Jarak dalam kilometer
    distance_km = R * c

    # Konversi jarak ke meter
    distance_m = distance_km * 1000
    return distance_m


# Menambahkan kolom jarak
uber1['distance_m'] = uber1.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'],
row['dropoff_longitude'], row['dropoff_latitude']), axis=1)


from sklearn.preprocessing import StandardScaler

X = uber1[['distance_m', 'passenger_count']]
y = uber1['fare_amount']

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)







# Inisialisasi model Decision Tree
DT_best = DecisionTreeRegressor(random_state=42)

# Latih model dengan data latih
DT_best.fit(X_train, y_train)

# Streamlit app
st.title("Prediksi Harga Transportasi")

# Input dari pengguna
jarak = st.number_input("Masukkan Jarak yang Ingin Ditempuh (m):", min_value=0.0, step=0.1)
jumlah_penumpang = st.number_input("Masukkan Jumlah Penumpang:", min_value=1, step=1)

# Prediksi harga
if st.button("Prediksi Harga"):
    estimasi_harga = DT_best.predict([[jarak, jumlah_penumpang]])
    st.write(f"Estimasi Harga untuk jarak {jarak} m dan {jumlah_penumpang} penumpang adalah: {estimasi_harga[0]:.2f}")


