import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Draw hand landmarks and skeleton on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Check if landmark is visible
                if landmark.visibility < 0 or landmark.presence < 0:
                    continue
                
                # Get the pixel coordinates of the landmark
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # Draw a dot on the landmark position
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Draw skeleton
            connections = mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start_point = tuple((int(hand_landmarks.landmark[start_idx].x * w), int(hand_landmarks.landmark[start_idx].y * h)))
                end_point = tuple((int(hand_landmarks.landmark[end_idx].x * w), int(hand_landmarks.landmark[end_idx].y * h)))
                
                # Check if either start or end landmark is not visible
                if hand_landmarks.landmark[start_idx].visibility < 0 or hand_landmarks.landmark[start_idx].presence < 0 or \
                   hand_landmarks.landmark[end_idx].visibility < 0 or hand_landmarks.landmark[end_idx].presence < 0:
                    continue
                
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Gesture Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
