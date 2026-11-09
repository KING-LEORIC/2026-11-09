from pathlib import Path
import sys
import json
import cv2
import numpy as np
import pandas as pd
# 현재 실행 중인 스크립트 위치 계산
# - __file__ : 현재 파이썬 파일 경로
# - resolve() : 절대경로 변환
# - parent : 상위 디렉터리
base = Path(__file__).resolve().parent
# 프로젝트 루트 디렉터리 계산
# - base.parent : 현재 디렉터리의 상위 경로
root = base.parent
# 데이터 디렉터리 경로
data = root / 'data'
# 출력 디렉터리 경로
out = base / 'outputs'
# 출력 디렉터리가 없으면 생성
# - exist_ok=True : 이미 존재해도 오류 발생하지 않음
out.mkdir(exist_ok=True)
# 사용자 모듈 import 경로 추가
# - day16_documents 폴더를 import 검색 경로에 등록
sys.path.insert(0, str(root / 'day16_documents'))
from fields import extract, extract_items, validate, reconcile
from evaluate import cer
# 실습 데이터(JSON) 로드
# - lesson.json : OCR 평가용 샘플 데이터
# - utf-8 인코딩으로 읽기
lesson = json.loads((data / 'lesson.json').read_text(encoding='utf-8'))

# 결과 테이블 생성을 위한 행 데이터 저장
# - 각 OCR 후보의 평가 결과를 누적
rows=[]
for key, sample in lesson['variants'].items():
    # OCR 후보별 평가 결과 기록
    # - case : 데이터 구간 식별자
    # - candidate : OCR 후보 이름
    # - cer(...) : 문자 오류율(Character Error Rate)
    # - backend : 사용 OCR 엔진
    # - inference_ms : 추론 시간
    rows.append({'case':sample['slice'],'candidate':sample['candidate'],'cer':cer(sample['reference_text'],sample['ocr']['text']), 'backend':'recorded_tesseract','inference_ms':None})
# 수집된 평가 결과를 데이터프레임으로 변환
table=pd.DataFrame(rows)
# 같은 개발 문서에서 선택한 결과입니다. 독립 test 성능이 아닙니다.
# TODO C01

# CER 기준 상위 3개 최적 후보 선택
# - cer : 문자 오류율(낮을수록 좋음)
# - nsmallest(3,'cer') : CER가 가장 낮은 3개 행 선택
# - best : 최종 선정 후보 데이터프레임
best: pd.DataFrame = table.nsmallest(3, 'cer')
# TODO 끝 C01
# 전체 후보 결과 저장
# - 후보별 CER 및 메타데이터를 CSV 파일로 기록
table.to_csv(out/'candidates.csv',index=False,encoding='utf-8-sig')
# 선정된 최상위 후보 저장
# - CER 기준 상위 3개 결과만 저장
best.to_csv(out/'selected.csv',index=False,encoding='utf-8-sig')
# 선정 결과 출력
# - case : 샘플 구간
# - candidate : 후보 이름
# - cer : 문자 오류율
# - round(4) : 소수점 넷째 자리까지 표시
print(best[['case','candidate','cer']].round(4).to_string(index=False))
