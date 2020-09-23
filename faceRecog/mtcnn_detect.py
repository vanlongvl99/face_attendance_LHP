from mtcnn.mtcnn import MTCNN

detector = MTCNN()

def mtcnn_detect(image):
    faces = detector.detect_faces(image)
    for person in faces:
        