# relevent libraries :
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

data_dir = 'gtsrb-german-traffic-sign'
train_path = 'gtsrb-german-traffic-sign/Train'
test_path = 'gtsrb-german-traffic-sign/Test'

# Label Overview
classes = { 0:'',
            1:'',
            2:'',
            3:'',
            4:'',
            5:'',
            6:"",
            7:'',
            8:'',
            9:'',
            10:'',
            11:'',
            12:'',
            13:'',
            14:'Stop',
            15:'',
            16:'',
            17:'',
            18:'',
            19:'Turn left',
            20:'Turn right',
            21:'curve',
            22:'',
            23:'',
            24:'',
            25:'',
            26:'Traffic signals',
            27:'',
            28:'',
            29:'',
            30:'',
            31:'',
            32:'',
            33:'Turn right',
            34:'Turn left',
            35:'',
            36:'',
            37:'',
            38:'Turn right',
            39:'Turn left',
            40:'',
            41:'',
            42:'' }

# Resizing the images to 30x30x3
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3


# loading the trained model :
model = load_model("gtsrb-german-traffic-sign/traficsigns1.h5")

# testing data
test = pd.read_csv(data_dir + '/Test.csv')

labels = test["ClassId"].values
imgs = test["Path"].values

data =[]

for img in imgs:
    try:
        image = cv2.imread(data_dir + '/' +img)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
        data.append(np.array(resize_image))
    except:
        print("Error in " + img)
X_test = np.array(data)
X_test = X_test/255

pred = model.predict_step(X_test)

#Accuracy with the test data
pred_classes = np.argmax(pred, axis=-1)
# print('Test Data accuracy: ', accuracy_score(labels, pred_classes) * 100)


# classification report :

from sklearn.metrics import classification_report

# print(classification_report(labels, pred_classes))

# testing several images

# plt.figure(figsize = (25, 25))
# start_index = 0
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     prediction = pred_classes[start_index + i]
#     actual = labels[start_index + i]
#     col = 'g'
#     if prediction != actual:
#         col = 'r'
#     plt.xlabel('Actual={} || Pred={}'.format(actual, prediction), color = col)
#     plt.imshow(X_test[start_index + i])
# plt.show()


# testing a single image :

# import cv2
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model
#
# # Load the image using OpenCV or PIL
# image_path = 'stop.png'
# image = cv2.imread(image_path)
#
# # Preprocess the image
# image_fromarray = Image.fromarray(image, 'RGB')
# resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
# image_data = np.array(resize_image)
# image_data = image_data / 255.0
#
# # Add a batch dimension
# image_data = np.expand_dims(image_data, axis=0)
#
# # Load the saved model
# model_path = "gtsrb-german-traffic-sign/traficsigns.h5"
# model = load_model(model_path)
#
# # Get the class probabilities and find the class with the highest probability
# probabilities = model.predict(image_data)
# predicted_class = np.argmax(probabilities)
#
# # Print the predicted class
# print("Predicted class:", predicted_class)
# print("Predicted class name:", classes[predicted_class])


# tesing livestream :


def preprocess_image(image):
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
    image_data = np.array(resize_image)
    image_data = image_data / 255.0
    image_data = np.expand_dims(image_data, axis=0)
    return image_data


# Open the video stream
cap = cv2.VideoCapture(0)

while True:
    # Capture each frame
    ret, frame = cap.read()

    # Preprocess the frame and make predictions
    preprocessed_frame = preprocess_image(frame)
    probabilities = model.predict(preprocessed_frame)
    predicted_class = np.argmax(probabilities)

    # Display the frame with the predicted class name
    cv2.putText(frame, classes[predicted_class], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-time Traffic Sign Recognition', frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
