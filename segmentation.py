from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def segment_guests(data, features, n_clusters=4):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    model = KMeans(n_clusters=n_clusters, random_state=42)
    segments = model.fit_predict(X)

    data['guest_segment'] = segments

    return data
