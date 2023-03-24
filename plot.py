import numpy as np
import matplotlib.pyplot as plt

# Example confusion matrix data
cm = np.array([[226,  12,  54,  55,  20,  16,  29,  88],
               [ 21, 370,   4,  25,   3,  11,   1,  65],
               [ 73,  15, 257,  24,  58,  19,  32,  22],
               [ 39,  33,  31, 232, 125,  13,   7,  20],
               [ 20,   8,  27,  70, 350,  10,   7,   8],
               [ 45,  28,  44,  26,  46, 239,  53,  19],
               [ 79,   6,  44,  36,  51,  55, 202,  27],
               [ 73,  90,  19,  15,  10,  25,  24, 243]])

# Convert absolute numbers to percentages
#cm = cm / np.sum(cm) * 100

# Plot confusion matrix
#plt.imshow(cm, cmap=plt.cm.YlGnBu)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(cm.shape[0])
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.grid(False)

# Add text labels to the plot, with white color for the highest numbers
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > thresh else "black"
        #color = "white" if cm[i, j] < thresh else "black"
        #plt.text(j, i, "{:.1f} %".format(cm[i, j]), ha='center', va='center', color=color)
        plt.text(j, i, cm[i, j], ha='center', va='center', color=color)

#plt.savefig("confusion_matrix.svg", format="svg")
plt.show()