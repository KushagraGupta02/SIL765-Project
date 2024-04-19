import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib

# read images from data directory
def read_images():
    images = []
    for filename in os.listdir('data_new'):
        img = cv2.imread(os.path.join('data_new', filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    print("Images Loaded")
    return images

def image_entropy(img):
    # Compute the histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return entropy

count = 0
def compression_1_ratio(img, threshold):
    global count
    original_img = img
    print("here")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(count)+'_0.ppm', img)
    print("ppm file done!")
    # run the command ./spiht c0.1  count.ppm count.ims on terminal
    s = f"./src/spiht c0.1 {count}_0.ppm {count}_1.ppm"
    os.system(s)
    original_size = os.path.getsize(f"{count}_0.ppm")
    compressed_size = os.path.getsize(f"{count}_1.ppm")
    print("sys call done!")
    # read the file count.ims
    # compressed_img = cv2.imread(str(count)+'_1.ppm', cv2.IMREAD_GRAYSCALE)  
    count += 1  
    # original_size = original_img.size
    # print(original_size)
    # compressed_size = compressed_img.size
    compression_ratio = original_size / compressed_size
    print("1 ratio", compression_ratio)
    return compression_ratio


# SVD based image compression
def compression_2_ratio(img, threshold):
    # Perform Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(img)
    
    # Determine the number of singular values to keep based on the threshold
    total_energy = np.sum(S)
    energy_threshold = threshold * total_energy
    cumulative_energy = np.cumsum(S)
    num_singular_values = np.argmax(cumulative_energy >= energy_threshold) + 1
    
    # Truncate the singular values and reconstruct the compressed image
    S_truncated = np.diag(S[:num_singular_values])
    U_truncated = U[:, :num_singular_values]
    V_truncated = V[:num_singular_values, :]
    compressed_img = U_truncated @ S_truncated @ V_truncated
    
    # Calculate the compression ratio
    original_size = img.size
    compressed_size = num_singular_values * (U_truncated.size + S_truncated.size + V_truncated.size)
    compression_ratio = original_size / compressed_size
    print("2 ratio", compression_ratio)
    return compression_ratio

def svd_image_compression(original_image, k):
    # Perform SVD
    U, s, Vt = np.linalg.svd(original_image, full_matrices=False)
    
    # Truncate singular values
    S = np.diag(s[:k])
    U_truncated = U[:, :k]
    Vt_truncated = Vt[:k, :]
    
    # Reconstruct the compressed image
    compressed_image = np.dot(U_truncated, np.dot(S, Vt_truncated))
    
    # Convert pixel values to unsigned integer 8-bit
    compressed_image = np.uint8(compressed_image)
    
    # Calculate compression ratio
    original_size = original_image.size
    compressed_size = U_truncated.size + S.size + Vt_truncated.size
    compression_ratio = original_size / compressed_size
    
    return original_image, compressed_image, compression_ratio

# SVD based image compression
def compression_2_ratio(img, threshold):
    _, _, compression_ratio = svd_image_compression(img, 50)    
    return compression_ratio

def main():
    images = read_images()
    entropies = [image_entropy(img) for img in images]
    compression_1_ratios = []
    compression_2_ratios = []
    for img in images:
        c1 = compression_1_ratio(img, 0.1)
        c2 = compression_2_ratio(img, 0.1)
        print(c2)
        compression_1_ratios.append(c1/40)
        compression_2_ratios.append(c2)
    # train a classifier whether to use compression 1 or 2 according to the entropy
    # Prepare the data
    
    # save compression ratios arrays in a files 1, 2
    print(compression_1_ratios, compression_2_ratios)
    with open("parameters.txt", "w") as file:
    # Write the content to the file
        file.write("C1 ratios")
        for i in compression_1_ratios:
            file.write(str(i))
        file.write("C2 ratios")
        for i in compression_2_ratios:
            file.write(str(i))
    
    X = np.array(entropies).reshape(-1, 1)  # Reshape the entropies array to match the input format
    y = np.array([0 if compression_1_ratios[i]>compression_2_ratios[i] else 1 for i in range(len(compression_1_ratios))])  # Use compression 1 if compression ratio is higher

    # print number of 0s and 1s in y
    print("0 count: ", np.count_nonzero(y == 0))
    print("1 count: ", np.count_nonzero(y == 1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    # Evaluate the classifier
    accuracy = classifier.score(X_test, y_test)
    print("Accuracy:", accuracy)
    y_pred = classifier.predict(X_test)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Predict on test set
    y_pred = classifier.predict(X_test)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = tn / (tn + fp)
    auroc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Print the metrics
    print("False Positive:", fp)
    print("False Negative:", fn)
    print("True Positive:", tp)
    print("True Negative:", tn)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Specificity:", specificity)
    print("AUROC:", auroc)
    print("False Positive Rate:", fpr)
    print("True Positive Rate:", tpr)
        
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Calculate AUC (Area Under Curve)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    # Save the model to a file
    joblib.dump(classifier, 'decision_tree_model.pkl')


if __name__ == "__main__":
    # img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    # original_img, compressed_img, compression_ratio = svd_image_compression(img, 150)
    # print("Compression Ratio:", compression_ratio)
    # cv2.imshow('Original Image', original_img)
    # cv2.imshow('Compressed Image', compressed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(0)
    main()