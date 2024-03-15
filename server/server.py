import dlib
import cv2
import numpy as np
import open3d as o3d

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "/home/ash93/poc/server/shape_predictor_68_face_landmarks (3).dat.bz2"  # Replace this with the actual path
predictor = dlib.shape_predictor(predictor_path)

# Load the 3D face model
face_model_path = "/home/ash93/poc/server/uploads_files_2958480_female+head.obj"
face_mesh = o3d.io.read_triangle_mesh(face_model_path)

# Load an image
image_path = "/home/ash93/poc/server/sam.jpeg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Loop over the faces and draw facial landmarks
for face in faces:
    shape = predictor(gray, face)

    # Convert 2D landmarks to numpy array
    landmarks_2d = np.array([[p.x, p.y] for p in shape.parts()])

    # Convert the 2D landmarks to 3D using the 3D face model
    # Assuming that the 3D face model has corresponding landmarks in the same order as the 2D landmarks
    # This mapping should be defined according to your specific 3D face model
    # Here, we assume that the first 68 vertices of the 3D face model correspond to the 68 2D landmarks
    landmarks_3d = np.array(face_mesh.vertices)[:68]

    # Project 3D landmarks onto the 2D image plane
    projection_matrix = np.array([[1, 0, 0], [0, 1, 0]])  # Simple projection matrix (identity)
    projected_landmarks = np.dot(landmarks_3d, projection_matrix.T)

    # Draw the projected 3D landmarks
    for x, y in projected_landmarks:
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)

# Display the result
cv2.imshow("3D Facial Landmark Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
