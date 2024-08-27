import os
import base64
import requests
from tkinter import Tk, Label, Button, filedialog, Frame
from PIL import Image, ImageTk

# 设置OpenAI的API密钥
api_key = 'sk-proj-8DZ8o0Fes75agebcxrCKT3BlbkFJBwM8ITd2R1PkJiCB7Hnk'


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Function to send image to OpenAI and get the response
def get_image_classification(image_path):
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = """
    You are an well-trained advanced model discriminator proficient in analyzing driver behavior from an in-car perspective.

    Please classify the driver's behavior into one of the following categories by only returning the category code (c0 to c7) without any additional text or explanation:
    c0: safe driving
    c1: texting
    c2: talking on the phone
    c3: operating the radio
    c4: drinking
    c5: reaching behind
    c6: hair and makeup
    c7: talking to passenger

    You should focus more on the actions of the person sitting in the driver's seat and carefully distinguish between the states provided (c0 to c7). Here are some annotations to help you:
    - safe driving (c0): The driver is looking ahead with both hands on the steering wheel.
    - texting (c1): The driver is looking at their phone, not close to the ear.
    - talking on the phone (c2): The driver is holding the phone close to their ear or face.
    - operating the radio (c3):The driver’s right hand reaches towards the central console to operate the controls, with only one hand remaining on the steering wheel.
    - drinking (c4): The driver is holding a water bottle or cup with one hand.
    - reaching behind (c5): The driver’s right hand is stretched towards the rear seats of the vehicle and is not visible in the image.
    - hair and makeup (c6): The driver is not holding a phone and is using a hand to groom their hair or apply makeup, possibly in front of a mirror.
    - talking to passenger (c7): The driver is engaging in conversation with the passenger.

    Please carefully observe the actions of the driver in the driver's seat. Try to improve the accuracy of your response through careful observation of the driver!

    Additional guidelines to help with classification:
    - If the driver has both hands on the steering wheel, it can only be c0 or c7 (but c7 will clearly show the driver talking to someone, and c7 might not always have both hands on the steering wheel).
    - If the driver is holding a phone with one hand, it can only be c1 or c2.
    - If the driver is holding a cup or bottle, it can only be c4.
    - If the driver’s right hand is extended forward and not fully visible in the image, it can only be c3.
    - If the driver’s right hand is extended backward and not fully visible in the image, it can only be c5.
    - If the driver’s hand is near their head or face without holding anything, it can only be c6.
    - The driver must have both hands on the steering wheel and be looking ahead for safe driving.

    Respond with only the category code (e.g., c0, c1, c2, etc.).
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


# Risk scores for each classification
risk_scores = {
    'c0': 0,
    'c1': 80,
    'c2': 80,
    'c3': 30,
    'c4': 70,
    'c5': 90,
    'c6': 90,
    'c7': 50
}

# Classification descriptions
classification_descriptions = {
    'c0': 'safe driving',
    'c1': 'texting',
    'c2': 'talking on the phone',
    'c3': 'operating the radio',
    'c4': 'drinking',
    'c5': 'reaching behind',
    'c6': 'hair and makeup',
    'c7': 'talking to passenger'
}


# GUI Functionality
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        load_image(file_path)
        classification = get_image_classification(file_path)
        print(classification)
        category_code = classification['choices'][0]['message']['content'].strip()
        classification_description = classification_descriptions.get(category_code, 'Unknown classification')
        classification_label.config(text=f"Classification: {classification_description}")

        risk_score = risk_scores.get(category_code, 'Unknown')
        if risk_score == 0:
            risk_label.config(text="Risk Score: 0 (Safe Driving)")
        else:
            risk_label.config(text=f"Risk Score: {risk_score}")


def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((400, 400), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo


# Create GUI
root = Tk()
root.title("Driver Behavior Classifier")
root.geometry("800x500")  # 设置窗口大小

# Main Frame
main_frame = Frame(root)
main_frame.pack(expand=True, fill="both")

# Display Image
image_label = Label(main_frame)
image_label.grid(row=0, column=0, padx=10, pady=10)

# Create a frame for classification and risk score
info_frame = Frame(main_frame)
info_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Classification Result
classification_label = Label(info_frame, text="Classification: ", font=("Helvetica", 16))
classification_label.pack()

# Risk Score Result
risk_label = Label(info_frame, text="Risk Score: ", font=("Helvetica", 16))
risk_label.pack()

# Open File Button
open_button = Button(main_frame, text="select image", command=open_file)
open_button.grid(row=2, column=0, columnspan=2, pady=20)

# Ensure grid expands properly
main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(0, weight=1)
main_frame.rowconfigure(1, weight=1)

root.mainloop()
