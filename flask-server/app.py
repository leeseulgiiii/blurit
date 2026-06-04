from flask import Flask, request, jsonify, send_file, send_from_directory
from ultralytics import YOLO
from flask_cors import CORS
import os
import uuid
import cv2

app = Flask(__name__)
CORS(app)

# 모델 로드
model = YOLO("C:/Users/LG/OneDrive/Desktop/BlurIT/yolo_m.pt")

# 클래스 ID → 이름 매핑
CLASS_NAMES = {
    0: "alcohol",
    1: "insulting_gesture",
    2: "blood",
    3: "cigarette",
    4: "gun",
    5: "knife"
}

# 업로드 폴더 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/uploads/<filename>")
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/")
def home():
    return "YOLO Flask 서버 작동 중입니다."

@app.route("/video-predict", methods=["POST"])
def video_predict():
    if 'file' not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files['file']
    filename = f"{uuid.uuid4().hex}.mp4"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # ✅ 사용자 입력: 블러링 강도 (0~10단계)
    try:
        blur_level = int(request.form.get("blur_strength", 5))  # 기본값 5
        blur_level = max(0, min(blur_level, 10))  # 제한
    except ValueError:
        blur_level = 5

    # 단계에 따라 resize 크기 매핑
    # 예: 0 → 4, 10 → 60
    mosaic_size = int(60 - (blur_level * 5.6))  # 0단계=60, 10단계=4
    mosaic_size = max(2, mosaic_size)  # 혹시 모를 오류 방지용 하한


    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return jsonify({"error": "비디오 열기 실패"}), 500

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if width == 0 or height == 0 or fps == 0:
        return jsonify({"error": "비디오 정보가 올바르지 않습니다."}), 500

    output_name = f"processed_{filename}"
    output_path = os.path.join(UPLOAD_FOLDER, output_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    timeline = []
    frame_idx = 0
    last_logged_second = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        try:
            results = model.predict(frame, verbose=False)
            boxes = results[0].boxes

            if boxes:
                current_time = frame_idx / fps
                current_second = int(current_time)

                if current_second > last_logged_second:
                    class_ids = [int(box.cls[0]) for box in boxes]
                    class_names = [CLASS_NAMES.get(cid, f"unknown({cid})") for cid in class_ids]
                    timeline.append(
                        f"Frame {frame_idx} | Time: {current_time:.2f} sec | Detections: {len(boxes)} | Classes: {class_names}"
                    )
                    last_logged_second = current_second

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size != 0:
                    # ✅ 사용자 설정 강도 적용 (mosaic_size)
                    blur = cv2.resize(roi, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
                    mosaic = cv2.resize(blur, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                    frame[y1:y2, x1:x2] = mosaic

        except Exception as e:
            print(f"❌ YOLO 예측 중 오류: {e}")
            continue

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # 타임라인 저장
    timeline_name = f"timeline_{filename}.txt"
    timeline_path = os.path.join(UPLOAD_FOLDER, timeline_name)
    with open(timeline_path, "w", encoding='utf-8') as f:
        f.write("\n".join(timeline))

    return jsonify({
        "video": output_name,
        "timeline": timeline_name
    })

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
