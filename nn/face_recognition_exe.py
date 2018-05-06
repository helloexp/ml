
import face_recognition
from PIL import Image,ImageDraw

facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
    ]

def face_recongnition_e1():

    img = face_recognition.load_image_file("/Users/tong/PycharmProjects/machinelearning/resource/destop.jpg")
    face_locations=face_recognition.face_locations(img,model="cnn")

    print(face_locations)
    # for face_location in face_locations:
    #
    #     # Print the location of each face in this image
    #     top, right, bottom, left = face_location
    #     print(
    #         "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
    #
    #     # You can access the actual face itself like this:
    #     face_image = img[top:bottom, left:right]
    #     pil_image = Image.fromarray(face_image)
    #     pil_image.show()

    face_landmarks_list = face_recognition.face_landmarks(img,face_locations)

    print(face_landmarks_list)

    pil_image = Image.fromarray(img)
    d = ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:
        for facial_feature in facial_features:
            d.line(face_landmarks[facial_feature], width=5)

    pil_image.show()


if __name__ == '__main__':


    face_recongnition_e1()


