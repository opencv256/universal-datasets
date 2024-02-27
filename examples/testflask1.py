from flask import Flask, request, render_template
import boto3, json
from werkzeug.utils import secure_filename
from socket import *

app = Flask(__name__)

@app.route('/receiveString', methods=['POST'])
def receive_string_from_spring_boot():
    if request.method == 'POST':
        received_string = request.data.decode('utf-8')
        # 받은 문자열 처리
        print("스프링부트에서 받은 문자열:", received_string)
        return "플라스크에서 문자열을 성공적으로 받았습니다!"
    else:
        return "이 엔드포인트는 POST 요청만 허용됩니다.", 405

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)  # Change host and port as needed