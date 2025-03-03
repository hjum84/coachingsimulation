# 1. Python 3.9 slim 이미지를 기반으로 사용합니다.
FROM python:3.9-slim

# 2. 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# 3. 필요한 OS 패키지를 설치합니다.
#    여기서 ffmpeg와 libsndfile1을 설치합니다.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
# 4. requirements.txt 파일을 복사하고, 파이썬 라이브러리를 설치합니다.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 5. 프로젝트의 모든 파일을 컨테이너에 복사합니다.
COPY . .

# 6. Flask 앱을 실행합니다.
CMD ["python", "improved-coaching-app.py"]
