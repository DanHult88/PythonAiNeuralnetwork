# PythonAiNeuralnetwork
Neuralt nätverk


Kod för att köra ett neuralt nätverk i exempelvis google colab

# Steg 1 importera dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Steg 2 Läs in dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Steg 3 skapa klassnamn
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot',]

# Steg 4 normalisera data
train_images, test_images = train_images / 255.0, test_images / 255.0

# Steg 5 skapa model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Steg 6 kompilera modellen
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']

)

# Steg 7 träna modell
model.fit(train_images, train_labels, epochs=5)

# Steg 8 testa modell
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("\nTestAccuracy:", test_acc)

# Steg 9 Förutsäga bilder
predictions = model.predict(test_images)

plt.figure(figsize=(15,10))
for i in range(20):
  plt.subplot(4, 5, i + 1 )
  plt.imshow(test_images[i], cmap=plt.cm.binary)
  predicted_label = class_names[np.argmax(predictions[i])]
  true_label = class_names[test_labels[i]]
  plt.title(f"Predicted: {predicted_label} \n True: {true_label}", fontsize=9 )
  plt.axis("off")
plt.tight_layout()
plt.show()
