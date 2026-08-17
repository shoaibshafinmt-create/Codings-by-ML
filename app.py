import cv2

# Load the Haar cascade for cars
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'cars.xml')

# Open video file (or use 0 for webcam)
cap = cv2.VideoCapture('cars_video.mp4')  # Replace with your video path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars
    cars = car_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw squares around each car
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show output
    cv2.imshow('Car Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()