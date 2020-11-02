from src.licensePlateRecognition import *

if __name__ == "__main__":
    while True:
        folder = input("Enter Directory Path : ")
        if folder:
            print('processing ...')
            for filename in os.listdir(folder):
                image = cv2.imread(os.path.join(folder, filename))
                lpr = LicensePlateRecognition(image)
                lpr.run(filename.replace(".jpg", ""))
            print('done.')
        else:
            print("Invalid Folder!")
