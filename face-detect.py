import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
image_path = 'Parliment2.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# Draw rectangles and add centered labels
for idx, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    label = f"{idx+1}"
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    text_x = x + (w - text_width) // 2
    cv2.putText(image, label, (text_x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 180), 2)  # Pink text

# Save the output image
output_path = 'output_Parliment2.jpg'
cv2.imwrite(output_path, image)

print(f"Output image saved as {output_path}")
