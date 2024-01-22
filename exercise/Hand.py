import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize utilities for drawing hand landmarks
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Set up camera
cap = cv2.VideoCapture(0)

# # Global variable to store the rotation angle
# rotation_angle = 0
# # update for global use
# def update_rotation_angle(new_angle):
#     global rotation_angle
#     rotation_angle = new_angle

def calculate_angle(center_x, center_y, point):
    # Calculate angle between a center point and another point
    angle = math.atan2(point.y - center_y, point.x - center_x)
    return math.degrees(angle)


def calculate_rotation(wrist, middle_tip, image_width, image_height):
    # Convert from normalized position to pixel coordinates
    wrist_x, wrist_y = wrist.x * image_width, wrist.y * image_height
    middle_tip_x, middle_tip_y = middle_tip.x * image_width, middle_tip.y * image_height

    # Calculate angle between the wrist-middle_tip line and the horizontal axis
    angle = math.atan2(middle_tip_y - wrist_y, middle_tip_x - wrist_x)
    rotation_angle = math.degrees(angle)

    # Normalize angle to range [-180, 180]
    rotation_angle = -((rotation_angle + 180) % 360 - 180)

    return rotation_angle, wrist_x, wrist_y



# Draw Rotated Rectangle
def draw_rotated_rectangle(image, wrist_x, wrist_y, width, height, angle):
    # Create a transparent overlay
    overlay = np.zeros_like(image, dtype=np.uint8)

    # Define the rectangle based on the wrist location
    rect_start_point = (int(wrist_x - width / 2), int(wrist_y - height / 2))
    rect_end_point = (int(wrist_x + width / 2), int(wrist_y + height / 2))

    # Draw the rectangle on the transparent overlay
    cv2.rectangle(overlay, rect_start_point, rect_end_point, (0, 255, 0), -1)  # -1 fills the rectangle

    # Get rotation matrix for the overlay (rotate around the wrist)
    rotation_matrix = cv2.getRotationMatrix2D((wrist_x, wrist_y), angle, 1)

    # Rotate the overlay
    rotated_overlay = cv2.warpAffine(overlay, rotation_matrix, (overlay.shape[1], overlay.shape[0]))

    # Add the rotated overlay to the image
    return cv2.addWeighted(image, 1, rotated_overlay, 1, 0)


# Process frames in loops
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Pre-process frame
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Post-process the Frame
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Define image_width and image_height after reading the frame
        image_height, image_width, _ = image.shape
        # Convert normalized coordinates to pixel coordinates
        image_width, image_height = image.shape[1], image.shape[0]


        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmarks for wrist and middle fingertip
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                wrist_x = int(wrist.x * image_width)
                wrist_y = int(wrist.y * image_height)

                # Draw hand landmarks.
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Get landmarks for fingertips
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                index_tip.x *= image_width
                index_tip.y *= image_height
                thumb_tip.x *= image_width
                thumb_tip.y *= image_height
                pinky_tip.x *= image_width
                pinky_tip.y *= image_height


                # Calculate rotation, center, width, and height
                rotation_angle, wrist_x, wrist_y = calculate_rotation(wrist, middle_tip, image_width, image_height)
                width = max(index_tip.x, thumb_tip.x, pinky_tip.x) - min(index_tip.x, thumb_tip.x, pinky_tip.x)
                height = max(index_tip.y, thumb_tip.y, pinky_tip.y) - min(index_tip.y, thumb_tip.y, pinky_tip.y)

                # # Update the global rotation angle
                # update_rotation_angle(rotation_angle)

                # Dynamic rectangle dimensions based on hand size
                rect_width = int(width)  # Modify as needed
                rect_height = int(height)  # Modify as needed

                # Draw the rotated rectangle
                image = draw_rotated_rectangle(image, wrist_x, wrist_y, rect_width, rect_height, rotation_angle)

                # Print the rotation angle
                print(f"Rotation Angle: {rotation_angle}")

        # Display processed frame
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()