from flask import Flask, jsonify, request
import cv2 as cv
from matplotlib import pyplot as plt
import os

app = Flask(__name__)

@app.route('/example', methods=['GET'])
def example():
    # Load and display images
    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
    image1 = cv.imread("../input/sptire/train/images/14_19_l_jpg.rf.8323d9f848377e32ca451017a3a80731.jpg")
    ax[0].imshow(image1)
    # Load other images...

    # Run YOLOv7
    # Note: This part requires further separation and adjustment for use in an API

    # Display and return the processed images
    fig.show()
    return jsonify({'message': 'Images displayed'})

if __name__ == '__main__':
    app.run(debug=True)
