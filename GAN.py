import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

tf.get_logger().setLevel(logging.ERROR)

class Gan():
    def __init__(self):
        self.data = pd.read_csv("graduation_dataset_preprocessed_feature_selected.csv")
        self.n_epochs = 200
        self.latent_dim = 27  

    def _noise(self):
        noise = np.random.normal(0, 1, (self.data.shape[0], self.latent_dim))
        return noise

    def _generator(self):
        model = tf.keras.Sequential(name="Generator_model")
        model.add(tf.keras.layers.Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=self.latent_dim))  # Adjusted input_dim
        model.add(tf.keras.layers.Dense(30, activation='relu'))
        model.add(tf.keras.layers.Dense(self.data.shape[1], activation='linear'))
        return model

    def _discriminator(self):
        model = tf.keras.Sequential(name="Discriminator_model")
        model.add(tf.keras.layers.Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=self.data.shape[1]))
        model.add(tf.keras.layers.Dense(50, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def _GAN(self, generator, discriminator):
        discriminator.trainable = False
        generator.trainable = True
        model = tf.keras.Sequential(name="GAN")
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train(self, generator, discriminator, gan):
        for epoch in range(self.n_epochs):
            generated_data = generator.predict(self._noise())
            labels = np.concatenate([np.ones(self.data.shape[0]), np.zeros(self.data.shape[0])])
            X = np.concatenate([self.data, generated_data])
            discriminator.trainable = True
            d_loss, _ = discriminator.train_on_batch(X, labels)

            noise = self._noise()
            g_loss = gan.train_on_batch(noise, np.ones(self.data.shape[0]))

            print(f'>{epoch+1}, d_loss={d_loss:.3f}, g_loss={g_loss:.3f}')

        return generator

    def predict(self, generator):
        # Generate synthetic data
        gan_data = generator.predict(gan_instance._noise())
        gan_data_df = pd.DataFrame(gan_data)

        # Save the generated data to a CSV file
        gan_data_df.to_csv("generated_data.csv", index=False)
        # Load real data
        real_data = self.data
        
        # Merge real and generated data
        combined_data = np.vstack([real_data, gan_data])
        labels = np.concatenate([np.ones(len(real_data)), np.zeros(len(gan_data))])
        
        # Train a classifier
        X_train, X_test, y_train, y_test = train_test_split(combined_data, labels, test_size=0.25, random_state=1)
        
        # Example classifier: Random Forest
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Classifier Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    gan_instance = Gan()
    generator = gan_instance._generator()  # Create the generator model
    discriminator = gan_instance._discriminator()  # Create the discriminator model
    gan = gan_instance._GAN(generator, discriminator)  # Create the GAN
    trained_generator = gan_instance.train(generator, discriminator, gan)
    gan_instance.predict(trained_generator) 
    
    