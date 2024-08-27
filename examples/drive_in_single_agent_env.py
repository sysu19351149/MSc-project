#!/usr/bin/env python
"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
import argparse
import logging
import random
from pynput.keyboard import Key, Controller
import base64
import requests

import cv2
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE

keyboard = Controller()
last_choice = 'w'


def encode_image(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# 设置OpenAI的API密钥
api_key = '****'


# Function to send image to OpenAI and get the response
def get_image_classification(image, str1):
    base64_image = encode_image(image)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = """             
    I'm running the metadrive simulation script, assuming you are an autodrive agent.
    Next I'll give you the observation information,including the camera information, as well as velocity information, 
    steering information and acceleration information, 
    I need you, as an autopilot perceptual model, to autonomously understand this autopilot information and process it 
    to output a message in the same format, please note that the dimensions of the output data should be exactly the same as the input.
    """
    prompt += str1

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
        "max_tokens": 1
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


if __name__ == "__main__":
    config = dict(
        # controller="steering_wheel",
        use_render=True,
        manual_control=True,
        traffic_density=0.1,
        num_scenarios=10000,
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(show_lidar=True, show_navi_mark=False, show_line_to_navi_mark=False),
        # debug=True,
        # debug_static_world=True,
        map=4,  # seven block
        start_seed=10,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="rgb_camera", choices=["lidar", "rgb_camera"])
    args = parser.parse_args()
    if args.observation == "rgb_camera":
        config.update(
            dict(
                image_observation=True,
                sensors=dict(rgb_camera=(RGBCamera, 400, 300)),
                interface_panel=["rgb_camera", "dashboard"]
            )
        )
    env = MetaDriveEnv(config)
    try:
        o, _ = env.reset(seed=13)
        print(HELP_MESSAGE)
        env.agent.expert_takeover = False
        if args.observation == "rgb_camera":
            assert isinstance(o, dict)
            print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        else:
            assert isinstance(o, np.ndarray)
            print("The observation is an numpy array with shape: ", o.shape)
            print(o)
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            if i % 5 == 0:
                try:
                    keyboard.release(last_choice)
                except:
                    pass
                str1 = "Navigation information: " + str(info["navigation_command"])
                str2 = "  velocity information: " + str(info["velocity"])
                str3 = "  steering information: " + str(info["steering"])
                str4 = "  acceleration information: " + str(info["acceleration"])
                str5 = "reward: " + str(r)
                str_ = str1 + str2 + str3 + str4 + str5
                print(str_)
                result = get_image_classification(o["image"][..., -1], str_)
                last_choice = str(result['choices'][0]['message']['content'])
                print(last_choice)
                try:
                    keyboard.press(last_choice)
                except:
                    pass

            if args.observation == "rgb_camera":
                cv2.imshow('RGB Image in Observation', o["image"][..., -1])
                cv2.waitKey(1)
            if (tm or tc) and info["arrive_dest"]:
                env.reset(env.current_seed + 1)
                env.current_track_agent.expert_takeover = True
    finally:
        env.close()
