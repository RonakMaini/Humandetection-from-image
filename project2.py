import cv2
import argparse
import numpy as np

def preprocess_image(image, max_dim=800):
   
    height, width = image.shape[:2]
    if max(height, width) > max_dim:
        scale_factor = max_dim / float(max(height, width))
        image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))
    return image

def detect(frame, hog, args):
   
    # Perform people detection with valid hyperparameters
    coordinates, weights = hog.detectMultiScale(
        frame,
        winStride=tuple(args ["winStride"]),
        padding=tuple(args["padding"]),
        scale=args["scale"],
        hitThreshold=args["hitThreshold"]
    )
    
    human_count = 0
    for x, y, w, h in coordinates:
        # Draw rectangles around detected people
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Human {human_count + 1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 25), 2)
        human_count += 1

    # Display status and the count of detected humans
    cv2.putText(frame, 'Status: Detecting', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Humans: {human_count}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('Output', frame)
    return frame

def by_image(path, hog, args):
    """Detect people in an image."""
    image = cv2.imread(path)
    if image is None:
        print(f"Error: Image not found at path '{path}'")
        return
    image = preprocess_image(image)
    detect(image, hog, args)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_arguments():
    """Parse command-line arguments with hyperparameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image file", default=None)
    parser.add_argument("--scale", type=float, default=1.02, help="Image pyramid scale factor")
    parser.add_argument("--winStride", type=int, nargs=2, default=(4, 4), help="Window stride (width, height)")
    parser.add_argument("--padding", type=int, nargs=2, default=(8, 8), help="Padding (width, height)")
    parser.add_argument("--hitThreshold", type=float, default=0.4, help="SVM hit threshold")
    return vars(parser.parse_args())

if __name__ == "__main__":
    # Initialize the HOG descriptor for people detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Parse command-line arguments
    args = parse_arguments()

    # Check if an image path is provided
    if args["image"]:
        by_image(args["image"], hog, args)
    else:
        print("Error: No image path provided.")
