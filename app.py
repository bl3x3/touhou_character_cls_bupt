import time
import threading
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
import cv2
from ultralytics import YOLO
import numpy as np
from camera import ThreadedCamera
import json
import os
from datetime import datetime

app = Flask(__name__, static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*")

# 数据库配置
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///touhou_stats.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# 统计数据模型
class CharacterStats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    character_name = db.Column(db.String(100), nullable=False)
    count = db.Column(db.Integer, default=0)
    timestamp = db.Column(db.DateTime, default=datetime.now)


class RegionStats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    region_name = db.Column(db.String(100), nullable=False)
    count = db.Column(db.Integer, default=0)
    timestamp = db.Column(db.DateTime, default=datetime.now)


# 全局变量
camera = None
processing = False
inference_thread = None
stop_event = threading.Event()


# 创建保存图片的目录
SAVE_DIR = "saved_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


@app.route("/")
def index():
    """
    返回前端页面
    """
    return render_template("index.html")  # 若你的前端HTML放在templates/目录下


@app.route("/api/startProcessing", methods=["POST"])
def start_processing():
    """
    启动模型推理线程
    """
    global processing, inference_thread, stop_event
    if not processing:
        processing = True
        stop_event.clear()
        inference_thread = threading.Thread(target=run_inference_loop)
        inference_thread.start()
    return jsonify({"status": "processing started"}), 200


@app.route("/api/stopProcessing", methods=["POST"])
def stop_processing():
    """
    停止模型推理线程
    """
    global processing, inference_thread, stop_event
    if processing:
        processing = False
        stop_event.set()  # 通知线程退出
        if inference_thread is not None:
            inference_thread.join()
            inference_thread = None
    return jsonify({"status": "processing stopped"}), 200


@app.route("/api/camera_info")
def get_camera_info():
    """获取摄像头信息"""
    global camera
    if camera:
        return jsonify({"width": camera.frame_width, "height": camera.frame_height})
    return jsonify({"error": "Camera not initialized"}), 404


@app.route("/api/name_mapping")
def get_name_mapping():
    """获取英文名到中文名的映射关系"""
    try:
        with open("assets/cvt_name_full_version.json", "r", encoding="utf-8") as f:
            name_mapping = json.load(f)
        return jsonify(name_mapping), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def gen_frames():
    """生成视频流"""
    global camera
    while True:
        frame = camera.read()
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def load_model():
    """
    加载模型
    """
    return YOLO("model/yolo11m-02-01-best.pt", task="classify", verbose=False)


def center_crop_resize(image, ratio, size):
    """
    将图像按比例中心裁剪并调整大小
    """
    h, w, _ = image.shape
    crop_size = int(min(h, w) * ratio)
    center_h, center_w = h // 2, w // 2
    start_h, start_w = center_h - crop_size // 2, center_w - crop_size // 2
    cropped = image[start_h : start_h + crop_size, start_w : start_w + crop_size]
    resized = cv2.resize(cropped, size)
    return resized


def run_inference_loop():
    """后台线程：进行模型推理"""
    global camera
    try:
        model = load_model()
        classes = model.names

        while not stop_event.is_set():
            frame = camera.read()
            frame = center_crop_resize(frame, 0.65, (224, 224))
            # 进行模型推理
            results = model(frame)

            probs = results[0].probs
            top5 = probs.top5
            top5_conf = probs.top5conf.cpu().tolist()

            # results = [
            #     {"className": classes[int(i)], "probability": round(p, 4)}
            #     for i, p in zip(top5, top5_conf)
            # ]
            results = [
                {
                    "className": classes[int(i)],
                    "classNameCN": cvt_name_to_cn[classes[int(i)]],
                    "probability": round(p, 4),
                }
                for i, p in zip(top5, top5_conf)
            ]

            socketio.emit("inference_result", {"top5": results})
            time.sleep(0.8)  # 控制推理频率

    except Exception as e:
        print(f"Error in inference loop: {e}")


@app.route("/api/save_image", methods=["POST"])
def save_image():
    """保存图片到本地"""
    try:
        # 获取Base64编码的图片数据
        data = request.get_json()
        image_data = data.get("image")
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        # 解码Base64图片数据
        import base64

        image_data = base64.b64decode(image_data.split(",")[1])

        # 生成文件名（使用当前时间）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.png"
        filepath = os.path.join(SAVE_DIR, filename)

        # 保存图片
        with open(filepath, "wb") as f:
            f.write(image_data)

        # 通过WebSocket发送图片保存通知
        socketio.emit("image_saved", {"filename": filename, "timestamp": timestamp})

        return (
            jsonify({"status": "success", "filename": filename, "filepath": filepath}),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/update_stats", methods=["POST"])
def update_stats():
    """更新统计数据"""
    try:
        data = request.get_json()
        character = data.get("character")
        region = data.get("region")

        # 更新角色统计
        if character:
            char_stat = CharacterStats.query.filter_by(character_name=character).first()
            if char_stat:
                char_stat.count += 1
            else:
                char_stat = CharacterStats(character_name=character, count=1)
                db.session.add(char_stat)

        # 更新地域统计
        if region:
            region_stat = RegionStats.query.filter_by(region_name=region).first()
            if region_stat:
                region_stat.count += 1
            else:
                region_stat = RegionStats(region_name=region, count=1)
                db.session.add(region_stat)

        db.session.commit()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/get_stats")
def get_stats():
    """获取统计数据"""
    try:
        # 获取角色统计
        char_stats = (
            CharacterStats.query.order_by(CharacterStats.count.desc()).limit(5).all()
        )
        char_stats_data = [
            {"name": stat.character_name, "count": stat.count} for stat in char_stats
        ]

        # 获取地域统计
        region_stats = (
            RegionStats.query.order_by(RegionStats.count.desc()).limit(5).all()
        )
        region_stats_data = [
            {"name": stat.region_name, "count": stat.count} for stat in region_stats
        ]

        return (
            jsonify({"characters": char_stats_data, "regions": region_stats_data}),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    try:
        # 创建数据库表
        with app.app_context():
            db.create_all()

        print("Starting camera...")
        print("请选择摄像头源：")
        print("0: 使用电脑自带摄像头")
        print("1: 使用外接摄像头")
        while True:
            try:
                source = int(input("请输入摄像头源(0 或 1):"))
                if source == 0 or source == 1:
                    break
                else:
                    print("无效的选择，请输入 0 或 1")
            except ValueError:
                print("请输入有效的数字")
        camera = ThreadedCamera(camera_id=source)
        camera.start()

        print(f"Load name converting...")
        global cvt_name_to_cn
        with open("docs/cvt_name_full_version.json", "r", encoding="utf-8") as f:
            cvt_name_to_cn = json.load(f)

        app.run(debug=False, port=5000)  # 使用 debug=False 避免重复启动
    finally:
        if camera:
            camera.release()
