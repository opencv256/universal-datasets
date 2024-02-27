from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/receive-json', methods=['POST'])
def receive_json():
    # POST 요청으로 받은 데이터를 처리하여 응답으로 반환
    data = request.json
    print(data)
    print(data['message'])
    # 받은 메시지를 이용하여 응답 메시지를 생성
    response_message = f"Received message: {data['message']}"

    # 생성된 응답 메시지를 JSON 형태로 반환
    response_data = {
        'response_message': response_message
    }
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=False,host="127.0.0.1",port=5002)
