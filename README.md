# PeptoMatch Cell

동물 세포주(CHO, Hybridoma, VERO, BHK-21, MDCK, PK-15, DF-1) 전용 펩톤 추천 + 배양 데이터 수집 시스템.

미생물용 PeptoMatch와 별개의 독립 앱이며, 같은 FastAPI + Jinja2 + SQLite + Railway 스택을 사용한다.

## 구성

```
peptomatchCell/
├── gateway.py                 # FastAPI 진입점
├── src/peptomatch_cell/
│   ├── cell_culture_db.py     # SQLite 영속화 (실험/조건/시계열/메트릭)
│   ├── cell_line_priors.py    # 세포주별 demand profile
│   ├── composition_features.py
│   ├── scoring.py             # 펩톤 추천 엔진
│   └── ...
├── templates/                 # base/home/recommend/culture
├── data/
│   ├── composition_template.xlsx
│   └── cell_culture_data.db   # 자동 생성
├── start.sh, Procfile, railway.toml, nixpacks.toml
```

## 로컬 실행

```bash
# 1) PeptoMatch Cell
cd peptomatchCell
pip install -r requirements.txt
uvicorn gateway:app --reload --port 8100

# 2) cell-culture-app (Growth Curve App 역할)
cd ../cell-culture-app
pip install -r requirements.txt
PEPTOMATCH_CELL_URL=http://localhost:8100 python server.py
```

`cell-culture-app`(8000)에서 엑셀을 업로드하면 처리 결과가 자동으로
`peptomatchCell`(8100)의 `/api/ingest`로 POST되어 SQLite에 적재된다.

## 주요 엔드포인트

| Method | Path                              | 용도 |
|--------|-----------------------------------|------|
| GET    | `/`                               | 대시보드 |
| GET    | `/recommend`                      | 세포주별 펩톤 추천 UI |
| GET    | `/culture`                        | 배양 실험 브라우저 |
| POST   | `/api/ingest`                     | cell-culture-app 결과 적재 |
| GET    | `/api/cell/summary`               | 통계 |
| GET    | `/api/cell/experiments`           | 실험 목록 |
| GET    | `/api/cell/conditions`            | 조건 + 메트릭 |
| GET    | `/api/cell/timeseries?condition_id=` | 시계열 |
| GET    | `/api/cell/ml-data`               | ML 학습용 평탄화 데이터 |
| POST   | `/api/recommend`                  | 세포주별 펩톤 추천 |
| GET    | `/healthz`                        | 헬스체크 |

## DB 스키마

* `culture_experiments` — 엑셀 파일 1개 = 실험 1행
* `culture_conditions` — treatment(SOY-1, IMDM, SOY-1+RICE-1...) 1개 = 조건 1행
* `culture_timeseries` — (조건, metric{vcd,viability,ivcd,titer,mu,qp}) 시계열
* `culture_metrics` — 조건별 집계 (max_vcd, mu_max, max_titer, combined_score, ...)

## Railway 배포

`railway.toml` + `nixpacks.toml` + `start.sh`가 peptomatch와 동일한 구조이므로
`railway up` 한 번에 배포 가능. Volume을 `/app/data`에 마운트하면 DB가 영속화된다.

cell-culture-app 서비스의 환경변수에 `PEPTOMATCH_CELL_URL=https://<peptomatch-cell>.railway.app`
을 설정해주면 두 서비스가 자동 연동된다.
