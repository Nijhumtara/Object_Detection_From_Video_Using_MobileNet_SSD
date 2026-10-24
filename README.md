# Video Object Detection Using MobileNet SSD

Detects objects in a video using a pretrained MobileNet SSD model (Caffe) and OpenCV's DNN module. Each frame is processed, detected objects are marked with bounding boxes, class labels and confidence scores, and the result is saved as a new video. Built and run in Google Colab.

## Features
- Loads a pretrained MobileNet SSD model with `cv2.dnn.readNetFromCaffe`
- Processes an uploaded video frame by frame
- Draws bounding boxes with class name and confidence score
- Filters weak detections with a confidence threshold (default 0.5)
- Saves the annotated video, keeping the original resolution and FPS

## Detectable Classes
The model is trained on PASCAL VOC and detects 20 classes: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor.

## Tech Stack
Python, OpenCV (DNN), NumPy, Google Colab

## How It Works
1. Load the model (`deploy.prototxt` and `mobilenet_iter_73000.caffemodel`).
2. Read the video and get its width, height and FPS.
3. For each frame, convert it into a 300x300 blob and run a forward pass.
4. Scale the detected boxes back to the frame size and draw those above the confidence threshold.
5. Write each annotated frame to `detected_output.mp4`.

## How to Run
1. Open the notebook in Google Colab.
2. Upload `deploy.prototxt` and `mobilenet_iter_73000.caffemodel` to `/content`.
3. Run the cells in order and upload your video when prompted.
4. The annotated video downloads automatically when processing finishes.

## Limitations
- Uses a pretrained model, so it detects only the 20 VOC classes
- Processes uploaded videos, not a live camera feed
- Accuracy drops on small, distant or heavily occluded objects

## Future Improvements
- Live webcam detection
- Compare against other detectors such as YOLO
- Measure and report FPS and accuracy
