import pandas as pd  # pandas 모듈을 pd라는 이름으로 import
import matplotlib.pyplot as plt  # matplotlib.pyplot 모듈을 plt라는 이름으로 import
from tabulate import tabulate  # tabulate 모듈에서 tabulate 함수를 import
plt.rc('font', family='Malgun Gothic')  # 맑은 고딕 폰트를 사용하도록 설정

def generate_table(result_df):
    # 텍스트 테이블로 출력
    table = tabulate(result_df, headers='keys', tablefmt='psql', showindex=True)  # DataFrame을 텍스트 테이블로 변환
    return table  # 변환된 테이블 반환

def generate_graph(result_df):
    # 확률을 기준으로 내림차순 정렬
    result_df = result_df.sort_values(by='확률', ascending=False)  # '확률' 열을 기준으로 내림차순 정렬

    plt.figure(figsize=(10, 6))  # 그래프의 크기를 지정
    plt.bar(result_df['질병'], result_df['확률'], color='aqua', width=0.5)  # 막대 그래프를 생성하여 표시
    plt.title('질병별 확률')  # 그래프 제목 설정
    plt.xlabel('질병')  # x축 라벨 설정
    plt.ylabel('확률')  # y축 라벨 설정
    plt.xticks(rotation=90)  # x축 라벨을 90도 회전하여 표시

    # 그래프에 숫자 표시
    for i, v in enumerate(result_df['확률']):
        plt.text(i, v, str(v), ha='center', va='bottom')  # 막대 위에 숫자를 표시

    # 그래프 출력
    plt.show()  # 그래프를 화면에 출력
    plt.close()  # 그래프 창을 닫음

def visualize_results(outputs, model):
    output_probabilities_dict = dict(zip(model.pathologies, outputs[0].detach().numpy()))  # 출력 확률과 질병 이름을 딕셔너리로 생성

    # 결과를 담을 데이터프레임 생성
    result_df = pd.DataFrame(columns=['질병', '확률'])  # '질병'과 '확률' 열을 가진 빈 데이터프레임 생성

    # 데이터프레임에 결과 추가
    for disease in output_probabilities_dict:
        if disease not in ["Enlarged Cardiomediastinum", "Lung Lesion", "Lung Opacity", "Fracture", "Cardiomegaly"]:
            probability = output_probabilities_dict[disease]  # 질병의 확률 추출
            probability_percent = round(probability * 100, 2)  # 확률을 퍼센트로 변환하고 소수점 둘째 자리까지 반올림
            result_df = pd.concat([result_df, pd.DataFrame({'질병': [disease], '확률': [probability_percent]})], ignore_index=True)  # 결과를 데이터프레임에 추가

    result_df.index = result_df.index+1  # 인덱스를 1부터 시작하도록 설정

    # 표 생성
    table = generate_table(result_df)  # 테이블 생성 함수 호출

    # 그래프 출력
    generate_graph(result_df)  # 그래프 생성 함수 호출

    return table  # 생성된 테이블 반환
