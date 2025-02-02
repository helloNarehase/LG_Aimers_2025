import pandas as pd

def load_train(path = "Data/train.csv"):
    return pd.read_csv(path).drop(columns=['ID', "시술 시기 코드"])

def load_test(path = "Data/test.csv"):
    return pd.read_csv(path).drop(columns=['ID', "시술 시기 코드"])


class label_encode_with_onehot:
    def __init__(self):
        self.label_id = {}
        self.inital_format = {"id" : 0, "count" : 1}
        self.cur_id = 0
    
    def encode(self, datas:list):
        return_labels = []
        for data in datas:
            ch = self.label_id.get(data)
            if ch is not None:
                self.label_id[data]["count"] += 1
                return_labels.append(self.label_id[data]["id"])
            else:
                self.label_id[data] = self.inital_format.copy()
                self.label_id[data]["id"] = self.cur_id
                return_labels.append(self.label_id[data]["id"])
                self.cur_id += 1

        return return_labels
    
    def with_onehot(self, data:list):
        maxi = max(data)+1
        def one_hot(maxi, id):
            oh_format = [0] * maxi
            oh_format[id] = 1

            return oh_format
        return list(map(lambda id: one_hot(maxi, id), data))
        
        
    def __call__(self, datas):
        return self.encode(datas)
    
def automate_label_encoding(df, encodes = {}):
    string_columns = df.select_dtypes(include=['object']).columns

    for column in string_columns:
        if column in encodes.keys():
            label_encoder = encodes[column]
        else:
            label_encoder = label_encode_with_onehot()
        # 원핫 인코딩된 데이터 얻기
        labeled_data = label_encoder(df[column].astype(str).tolist())

        # 원핫 인코딩된 데이터는 2차원 배열, 이를 DataFrame으로 변환
        # encoded_df = pd.DataFrame(labeled_data, columns=[f"{column}" for i in range(len(labeled_data[0]))])
        
        # 원본 열 제거
        df = df.drop(column, axis=1)
        df[column] = labeled_data

        # 원본 DataFrame에 원핫 인코딩된 열을 추가
        # df = pd.concat([df, encoded_df], axis=1)
        
        encodes[column] = label_encoder
    return df, encodes


numeric_columns = [
    "임신 시도 또는 마지막 임신 경과 연수",
    "총 생성 배아 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",
    "저장된 배아 수",
    "미세주입 후 저장된 배아 수",
    "해동된 배아 수",
    "해동 난자 수",
    "수집된 신선 난자 수",
    "저장된 신선 난자 수",
    "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수",
    "기증자 정자와 혼합된 난자 수",
    "난자 채취 경과일",
    "난자 해동 경과일",
    "난자 혼합 경과일",
    "배아 이식 경과일",
    "배아 해동 경과일"
]

categorical_columns = [
    # "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 자극 여부",
    "배란 유도 유형",
    "단일 배아 이식 여부",
    "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
    "배아 생성 주요 이유",
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "IVF 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "IVF 출산 횟수",
    "DI 출산 횟수",
    "난자 출처",
    "정자 출처",
    "난자 기증자 나이",
    "정자 기증자 나이",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부",
    "PGD 시술 여부",
    "PGS 시술 여부"
]

selected_features = [
 '시술 당시 나이',
 '임신 시도 또는 마지막 임신 경과 연수',
 '배란 자극 여부',
 '단일 배아 이식 여부',
 '남성 주 불임 원인',
 '남성 부 불임 원인',
 '여성 주 불임 원인',
 '여성 부 불임 원인',
 '부부 주 불임 원인',
 '불임 원인 - 난관 질환',
 '불임 원인 - 남성 요인',
 '불임 원인 - 여성 요인',
 '불임 원인 - 자궁경부 문제',
 '불임 원인 - 정자 면역학적 요인',
 '불임 원인 - 정자 형태',
 '배아 생성 주요 이유',
 '클리닉 내 총 시술 횟수',
 'IVF 시술 횟수',
 'DI 시술 횟수',
 '총 임신 횟수',
 '총 출산 횟수',
 'IVF 출산 횟수',
 '미세주입된 난자 수',
 '이식된 배아 수',
 '저장된 배아 수',
 '해동된 배아 수',
 '수집된 신선 난자 수',
 '저장된 신선 난자 수',
 '혼합된 난자 수',
 '신선 배아 사용 여부',
 'PGD 시술 여부',
 '난자 해동 경과일',
 '난자 혼합 경과일',
 '배아 이식 경과일'
]

sel = [ 0,  1,  4,  6,  9, 10, 11, 12, 13, 16, 17, 19, 20, 23, 25, 26, 28,
       29, 30, 31, 34, 35, 38, 40, 42, 44, 46, 47, 48, 56, 59, 62, 63, 64]
categorical = ['시술 당시 나이', '시술 유형', '특정 시술 유형', '배란 유도 유형', '배아 생성 주요 이유', '난자 출처', '정자 출처',
       '난자 기증자 나이', '정자 기증자 나이']