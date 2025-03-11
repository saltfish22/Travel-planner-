import os
from openai import OpenAI
import gradio as gr  # 图片理解和网站UI
import requests
from gradio.components import HTML
import uuid #通用唯一标识符
from sparkai.core.messages import ChatMessage, AIMessageChunk
from dwspark.config import Config #星火的设置
from dwspark.models import ChatModel, ImageUnderstanding, Text2Audio, Audio2Text, EmbeddingModel,Text2Img #星火的模型
from PIL import Image
import io
import base64
import random
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics.pairwise import cosine_similarity
import pickle #序列化和反序列化
import re
import time
import json
import numpy as np
from text2audio.infer import audio2lip
# 日志
from loguru import logger
from langchain_community.tools.tavily_search import TavilySearchResults
import datetime
from http import HTTPStatus
from dashscope import Generation
import dashscope
from pydub import AudioSegment
# get location
def get_location_data(location, api_key):
    """
    向 QWeather API 发送 GET 请求以获取天气数据。

    :param location: 地点名称或经纬度（例如："beijing" 或 "116.405285,39.904989"）
    :param api_key: 你的 QWeather API 密钥
    :return: 响应的 JSON 数据
    """
    # 构建请求 URL
    url = f"https://geoapi.qweather.com/v2/city/lookup?location={location}&key={api_key}"

    # 发送 GET 请求
    response = requests.get(url)

    # 检查响应状态码
    if response.status_code == 200:
        # 返回 JSON 数据
        return response.json()
    else:
        # 处理错误情况
        print(f"请求失败，状态码：{response.status_code}")
        print(response.text)
        return None
# get weather
def get_weather_forecast(location_id, api_key):
    """
    向QWeather API发送请求以获取未来几天的天气预报。

    参数:
    - location: 地点ID或经纬度
    - api_key: 你的QWeather API密钥
    - duration: 预报的时长，'3d' 或 '7d'

    返回:
    - 响应的JSON内容
    """

    # 构建请求的URL
    url = f"https://devapi.qweather.com/v7/weather/3d?location={location_id}&key={api_key}"

    # 发送GET请求
    response = requests.get(url)

    # 检查请求是否成功
    if response.status_code == 200:
        # 返回响应的JSON内容
        return response.json()
    else:
        # 如果请求不成功，打印错误信息
        print(f"请求失败，状态码：{response.status_code}，错误信息：{response.text}")
        return None
# get image from text
# def generate_image(prompt):
#     logger.info(f'get_image: {prompt}')
#     output_path = './demo.jpg'
#     t2i.gen_image(prompt, output_path)
#     return output_path
# initial gradio window

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def chat_with_gpt(chat_destination, chat_history, chat_departure, chat_days, chat_style, chat_budget, chat_people,
                  chat_other):
    # Formatting Prompt Message
    prompt = (
        "chat_departure：{}，chat_destination：{} ，chat_days：{} ，chat_style：{} ，chat_budget：{}，chat_people：{}，other requirements：{}"
        .format(chat_departure, chat_destination, chat_days, chat_style, chat_budget, chat_people, chat_other)
    )

    # append into history
    chat_history.append((chat_destination, ''))

    # use gpt 3.5-turbo
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": '''
            你现在是一位专业的旅行规划师，你的责任是根据旅行出发地、目的地、天数、行程风格（紧凑、适中、休闲）、预算、随行人数，帮助我规划旅游行程并生成详细的旅行计划表。请你以表格的方式呈现结果。旅行计划表的表头请包含日期、地点、行程计划、交通方式、餐饮安排、住宿安排、费用估算、备注。所有表头都为必填项，请加深思考过程，严格遵守以下规则：
            1. 日期请以DayN为格式如Day1，明确标识每天的行程。
            2. 地点需要呈现当天所在城市，请根据日期、考虑地点的地理位置远近，严格且合理制定地点，确保行程顺畅。
            3. 行程计划需包含位置、时间、活动，其中位置需要根据地理位置的远近进行排序。位置的数量可以根据行程风格灵活调整，如休闲则位置数量较少、紧凑则位置数量较多。时间需要按照上午、中午、晚上制定，并给出每一个位置所停留的时间（如上午10点-中午12点）。活动需要准确描述在位置发生的对应活动（如参观博物馆、游览公园、吃饭等），并需根据位置停留时间合理安排活动类型。
            4. 交通方式需根据地点、行程计划中的每个位置的地理距离合理选择，如步行、地铁、出租车、火车、飞机等不同的交通方式，并尽可能详细说明。
            5. 餐饮安排需包含每餐的推荐餐厅、类型（如本地特色、快餐等）、预算范围，就近选择。
            6. 住宿安排需包含每晚的推荐酒店或住宿类型（如酒店、民宿等）、地址、预估费用，就近选择。
            7. 费用估算需包含每天的预估总费用，并注明各项费用的细分（如交通费、餐饮费、门票费等）。
            8. 备注中需要包括对应行程计划需要考虑到的注意事项，保持多样性，涉及饮食、文化、天气、语言等方面的提醒。
            9. 请特别考虑随行人数的信息，确保行程和住宿安排能满足所有随行人员的需求。
            10.旅游总体费用不能超过预算。
            Now please follow the above rules strictly and generate a reasonable and detailed travel planner based on my travel origin, destination, number of days, itinerary style (compact, moderate, casual), budget, and number of people accompanying you.
            Remember you have to generate the travel planner in tabular form based on the information I provided such as travel destination, number of days, etc. The final answer must be in tabular form, answered in English.
            '''},
            {"role": "user", "content": prompt}
        ]
    )

    # get answer
    answer = response.choices[0].message.content

    # update history
    information = (
        "departure：{}，destination：{} ，days：{} ，style：{} ，budget：{}，people：{}"
        .format(chat_departure, chat_destination, chat_days, chat_style, chat_budget, chat_people)
    )
    chat_history[-1] = (information, answer)

    return  chat_history # answer,

css=""" 
#col-left {
    margin: 0 auto;
    max-width: 430px;
}
#col-mid {
    margin: 0 auto;
    max-width: 430px;
}
#col-right {
    margin: 0 auto;
    max-width: 430px;
}
#col-showcase {
    margin: 0 auto;
    max-width: 1100px;
}
#button {
    color: blue;
}

"""
gr.set_static_paths(paths=["static"])
image_path = "static/logo.png"
# manage gradio window
with gr.Blocks(css=css) as demo:
    html_code = """
     <!DOCTYPE html>
        <html lang="zh-CN">        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    background-color: #f8f9fa;
                    margin: 0;
                    padding: 10px;
                }
                .container {
                    max-width: 1500px;
                    margin: auto;
                    background-color: #ffffff;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    padding: 10px;
                }
                .logo img {
                    display: block;
                    margin: 0 auto;
                    border-radius: 7px;
                }
                .content h2 {
                    text-align: center;
                    color: #999999;
                    font-size: 24px;
                    margin-top: 20px;
                }
                .content p {
                    text-align: center;
                    color: #cccccc;
                    font-size: 16px;
                    line-height: 1.5;
                    margin-top: 30px;
                }
            </style>
        </head>
    <body>
            <div class="container">
                <div class="logo">
                    <img src="https://img.picui.cn/free/2024/11/13/67347d3094fae.png" alt="Logo" width="30%">
                </div>
                <div class="content">
                    <h2>😀 Welcom to “6221Traval AI”，Your dedicated travel companion! We are committed to providing you with personalized travel planning, companionship and sharing services to make your journey fun and memorable.\n</h2>     
                </div>
            </div>
    </body>
"""
    gr.HTML(html_code) #manage the page
    with gr.Tab("Travel Q&A agency"):
        Weather_APP_KEY = '797ab5e76cdf458b82b1283e100b9a5b'

        def weather_process(location):
            api_key = Weather_APP_KEY  # Hefeng_API_KEY
            location_data = get_location_data(location, api_key)
            # print(location_data)
            if not location_data:
                return "Unable to get city information, please check your input."
            location_id = location_data.get('location', [{}])[0].get('id')
            # print(location_id)
            if not location_id:
                return "Unable to get ID from city info."
            weather_data = get_weather_forecast(location_id, api_key)
            if not weather_data or weather_data.get('code') != '200':
                return "Cannot get weather forecast, please check your input and API key."
            # Building HTML tables to display weather data
            html_content = "<table>"
            html_content += "<tr>"
            html_content += "<th>DATE</th>"
            html_content += "<th>TEMPDay</th>"
            html_content += "<th>TEMPNight</th>"
            html_content += "<th>tempMax</th>"
            html_content += "<th>tempMin</th>"
            html_content += "<th>windDirDay</th>"
            html_content += "<th>windScaleDay</th>"
            html_content += "<th>windSpeedDay</th>"
            html_content += "<th>windDirNight</th>"
            html_content += "<th>windScaleNight</th>"
            html_content += "<th>windSpeedNight</th>"
            html_content += "<th>precip</th>"
            html_content += "<th>uvIndex</th>"
            html_content += "<th>humidity</th>"
            html_content += "</tr>"

            for day in weather_data.get('daily', []):
                html_content += f"<tr>"
                html_content += f"<td>{day['fxDate']}</td>"
                html_content += f"<td>{day['textDay']} ({day['iconDay']})</td>"
                html_content += f"<td>{day['textNight']} ({day['iconNight']})</td>"
                html_content += f"<td>{day['tempMax']}°C</td>"
                html_content += f"<td>{day['tempMin']}°C</td>"
                html_content += f"<td>{day.get('windDirDay', '未知')}</td>"
                html_content += f"<td>{day.get('windScaleDay', '未知')}</td>"
                html_content += f"<td>{day.get('windSpeedDay', '未知')} km/h</td>"
                html_content += f"<td>{day.get('windDirNight', '未知')}</td>"
                html_content += f"<td>{day.get('windScaleNight', '未知')}</td>"
                html_content += f"<td>{day.get('windSpeedNight', '未知')} km/h</td>"
                html_content += f"<td>{day.get('precip', '未知')} mm</td>"
                html_content += f"<td>{day.get('uvIndex', '未知')}</td>"
                html_content += f"<td>{day.get('humidity', '未知')}%</td>"
                html_content += "</tr>"
            html_content += "</table>"

            return HTML(html_content)

        with gr.Tab("Neighborhood Search & Networking Search & Weather Search"):

            weather_input = gr.Textbox(label="Please enter the city name to check the weather", placeholder="For example: Washington DC")
            weather_output = gr.HTML(value="", label="Weather Check Results")
            query_button = gr.Button("Check Weather", elem_id="button")
            query_button.click(weather_process, [weather_input], [weather_output])

    with gr.Tab("Travel Planner"):
        # with gr.Group():
        with gr.Row():
            chat_departure = gr.Textbox(label="Enter the tour departure point", placeholder="Please enter your departure point")
            gr.Examples(["Beijing", "Washington", "New York", "Tokyo", "Paris", "London"],chat_departure, label='departure example', examples_per_page=6)
            chat_destination = gr.Textbox(label="Enter your travel destination", placeholder="Please enter the place you want to go")
            gr.Examples(["Beijing", "Washington", "New York", "Tokyo", "Paris", "London"],chat_destination, label='destination example', examples_per_page=6)

        with gr.Accordion("Personalized options (number of days, itinerary style, budget, number of participants)", open=False):
            with gr.Group():
                with gr.Row():
                    chat_days = gr.Slider(minimum=1, maximum=20, step=1, value=3, label='Number of travel days')
                    chat_style = gr.Radio(choices=['Compact', 'Moderate', 'Leisure'], value='Moderate', label='Trip Style',elem_id="button")
                    chat_budget = gr.Textbox(label="Enter budget (with units)", placeholder="Please enter your budget")
                with gr.Row():
                    chat_people = gr.Textbox(label="Enter the number of participants", placeholder="Please enter your number of participants")
                    chat_other = gr.Textbox(label="Special preferences, requirements (may write none)", placeholder="Please write your special preferences, requirements")
                # chatting window
        llm_submit_tab = gr.Button("Submit", visible=True, elem_id="button")
        chatbot = gr.Chatbot([], elem_id="chat-box", label="Chatting-window", height=600)
        # click logic
        llm_submit_tab.click(fn=chat_with_gpt,
                             inputs=[chat_destination, chatbot, chat_departure, chat_days, chat_style, chat_budget,
                                     chat_people, chat_other], outputs=[chatbot]) #outputs=[chat_destination,chatbot]

if __name__ == "__main__":
    demo.queue().launch( share=True)

#allowed_paths=["static"],