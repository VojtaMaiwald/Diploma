import numpy as np
import matplotlib.pyplot as plt

# Example confusion matrix data
cm = np.array(
    [[278,  16,  45,  59,   9,  16,  41,  36],
     [ 29, 341,   8,  39,   1,  14,   4,  64],
     [ 85,   6, 286,  25,  29,  22,  36,  11],
     [ 59,  25,  23, 316,  55,  11,   6,   5],
     [ 24,  11,  48, 111, 265,  21,  19,   1],
     [ 51,  24,  45,  37,  21, 239,  70,  13],
     [ 79,   5,  46,  32,  19,  49, 255,  15],
     [120,  76,  13,  25,   2,  17,  33, 213],]
 )

#classes = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
classes = ["Neutrální", "Šťastný", "Smutný", "Překvapený", "Vystrašený", "Znechucený", "Naštvaný", "Opovržlivý"]

# Convert absolute numbers to percentages
#cm = cm / np.sum(cm)
#cm = cm / 500
#cm = cm / np.sum(cm) * 100

# Set the figure size
fig, ax = plt.subplots(figsize=(10, 9))

# Plot confusion matrix
#plt.imshow(cm, cmap=plt.cm.YlGnBu)
plt.imshow(cm, cmap=plt.cm.Blues)
#plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(cm.shape[0])
plt.xticks(tick_marks, classes, rotation = 45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predikce')
plt.ylabel('Skutečnost')
plt.grid(False)

# Add text labels to the plot, with white color for the highest numbers
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > thresh else "black"
        #color = "white" if cm[i, j] < thresh else "black"
        #plt.text(j, i, "{:.2f}".format(cm[i, j]), ha='center', va='center', color=color)
        plt.text(j, i, cm[i, j], ha='center', va='center', color=color)

#plt.savefig("confusion_matrix.svg", format="svg")
plt.show()