import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras import regularizers

# Step 1: Load the dataset
data_path = 'Mall_Customers.csv'  # Path to the dataset
df = pd.read_csv(data_path)

# Step 2: Select relevant features
# We'll use 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)'
selected_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
data = df[selected_columns].values

# Step 3: Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Save the scaler immediately after scaling the data
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Step 4: Split data into training and validation sets
X_train, X_val = train_test_split(data_scaled, test_size=0.2, random_state=42)

# Step 5: Define the Autoencoder with L2 regularization
autoencoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # Latent space
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(X_train.shape[1], activation='sigmoid')
])

# Step 6: Compile the model with an optimizer
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')

# Step 7: Define Early Stopping and Learning Rate Scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
def scheduler(epoch, lr):
    if epoch > 10:
        return lr * 0.5  # Reduce learning rate after 10 epochs
    return lr
lr_scheduler = LearningRateScheduler(scheduler)

# Step 8: Train the Autoencoder with callbacks
history = autoencoder.fit(
    X_train, X_train, 
    epochs=100, 
    batch_size=256, 
    validation_data=(X_val, X_val),
    callbacks=[early_stopping, lr_scheduler]
)

# Step 9: Plot Training and Validation Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the trained Autoencoder model
autoencoder.save('autoencoder_model.h5')

# Step 10: Extract Latent Features from the Autoencoder
latent_features_train = autoencoder.predict(X_train)
latent_features_val = autoencoder.predict(X_val)

# Step 11: Visualize Latent Features using PCA (2D or 3D)
pca = PCA(n_components=2)  # You can change this to 3 for 3D visualization
latent_features_pca = pca.fit_transform(latent_features_train)

plt.scatter(latent_features_pca[:, 0], latent_features_pca[:, 1], c='blue', label='Latent Features')
plt.title('Latent Features Visualization (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Step 12: Perform K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(latent_features_train)

# Step 13: Evaluate Clustering Quality with Silhouette Score
silhouette_avg = silhouette_score(latent_features_train, clusters)
print("Silhouette Score:", silhouette_avg)

# Step 14: Visualize Clusters
plt.scatter(latent_features_pca[:, 0], latent_features_pca[:, 1], c=clusters, cmap='viridis')
plt.title('Cluster Visualization (K-Means)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Optional: Save the cluster assignments to the dataset
df['Cluster'] = kmeans.predict(autoencoder.predict(data_scaled))
df.to_csv('Mall_Customers_with_Clusters.csv', index=False)
print("Cluster assignments saved to Mall_Customers_with_Clusters.csv")

# Save the trained KMeans model to a file
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)






    
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import pickle
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA

# # Step 1: Load the dataset
# data_path = 'Mall_Customers.csv'  # Path to the dataset
# df = pd.read_csv(data_path)

# # Step 2: Select relevant features
# # We'll use 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)'
# selected_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
# data = df[selected_columns].values

# # Step 3: Scale the data
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data)

# # Save the scaler immediately after scaling the data
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)

# # Step 4: Split data into training and validation sets
# X_train, X_val = train_test_split(data_scaled, test_size=0.2, random_state=42)

# # Step 5: Define the Autoencoder
# autoencoder = tf.keras.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),  # Latent space
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(X_train.shape[1], activation='sigmoid')
# ])

# # Compile the model
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# # Step 6: Train the Autoencoder
# history = autoencoder.fit(
#     X_train, X_train, 
#     epochs=100, 
#     batch_size=256, 
#     validation_data=(X_val, X_val)
# )

# # Step 7: Plot Training and Validation Loss
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Autoencoder Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # Save the trained Autoencoder model
# autoencoder.save('autoencoder_model.h5')

# # Step 8: Extract Latent Features from the Autoencoder
# latent_features_train = autoencoder.predict(X_train)
# latent_features_val = autoencoder.predict(X_val)

# # Step 9: Visualize Latent Features using PCA (2D or 3D)
# pca = PCA(n_components=2)  # You can change this to 3 for 3D visualization
# latent_features_pca = pca.fit_transform(latent_features_train)

# plt.scatter(latent_features_pca[:, 0], latent_features_pca[:, 1], c='blue', label='Latent Features')
# plt.title('Latent Features Visualization (PCA)')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()

# # Step 10: Perform K-Means Clustering
# kmeans = KMeans(n_clusters=5, random_state=42)
# clusters = kmeans.fit_predict(latent_features_train)

# # Step 11: Evaluate Clustering Quality with Silhouette Score
# silhouette_avg = silhouette_score(latent_features_train, clusters)
# print("Silhouette Score:", silhouette_avg)

# # Step 12: Visualize Clusters
# plt.scatter(latent_features_pca[:, 0], latent_features_pca[:, 1], c=clusters, cmap='viridis')
# plt.title('Cluster Visualization (K-Means)')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.colorbar(label='Cluster')
# plt.show()

# # Optional: Save the cluster assignments to the dataset
# df['Cluster'] = kmeans.predict(autoencoder.predict(data_scaled))
# df.to_csv('Mall_Customers_with_Clusters.csv', index=False)
# print("Cluster assignments saved to Mall_Customers_with_Clusters.csv")

# # Save the trained KMeans model to a file
# with open('kmeans_model.pkl', 'wb') as f:
#     pickle.dump(kmeans, f)
