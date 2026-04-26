# ⚙️ Bolt/Nut Counter AI — Pi 5 + AI HAT (Hailo-8L)

ระบบ AI นับจำนวน Bolt, Nut, Washer, Screw สำหรับ Production Station
รองรับ: Hailo-8L NPU (Pi AI HAT) → fallback YOLOv8 CPU

---

## 📁 โครงสร้างไฟล์

```
bolt_counter/
├── detector.py          # Core detection engine (Hailo + YOLOv8)
├── webcam_live.py       # โหมด Webcam realtime (window)
├── web_app.py           # Web dashboard ดูผ่าน browser
├── folder_test.py       # Batch test + accuracy report
├── train_model.py       # Train/export custom model
├── station_config.json  # Config per station
├── requirements.txt
└── models/
    ├── best.pt          # YOLOv8 weights
    └── best.hef         # Hailo HEF model
```

---

## 🚀 Setup บน Raspberry Pi 5

### 1. ติดตั้ง OS + dependencies
```bash
# Raspberry Pi OS 64-bit (Bookworm)
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv libcamera-apps

# สร้าง virtual environment
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. ติดตั้ง Hailo SDK (Pi AI HAT)
```bash
# ดาวน์โหลดจาก https://hailo.ai/developer-zone/
# ตาม official guide ของ Raspberry Pi AI HAT:
# https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html

sudo apt install hailo-all
# หรือ
pip install hailort  # ตาม version ที่ support
```

### 3. สร้างโฟลเดอร์ models
```bash
mkdir -p models logs results
```

---

## 🎯 วิธีใช้งาน

### โหมด 1: Webcam Live (หน้าต่าง CV2)
```bash
python webcam_live.py

# ตัวเลือก:
python webcam_live.py \
  --camera 0 \
  --conf 0.45 \
  --expected '{"bolt":4,"nut":4,"washer":8}' \
  --fullscreen

# Keys: [q] ออก  [s] screenshot  [r] reset log
```

### โหมด 2: Web Dashboard (เปิด browser)
```bash
python web_app.py --camera 0 --port 5000

# เปิด browser: http://<IP_ของ_Pi>:5000
# ดูได้จาก PC / มือถือในวง LAN เดียวกัน
```

### โหมด 3: Folder Batch Test (ทดสอบความแม่น)
```bash
# ทดสอบทั้งโฟลเดอร์
python folder_test.py \
  --images /path/to/test_images/ \
  --labels /path/to/labels/ \
  --output results/ \
  --conf 0.45

# ผลลัพธ์: CSV + JSON + ภาพ annotated
# แสดง Precision / Recall / F1 per class
```

---

## 🏋️ Train โมเดลของตัวเอง

### เตรียม Dataset (YOLO format)
```
dataset/
├── images/
│   ├── train/   # ภาพ training (80%)
│   └── val/     # ภาพ validation (20%)
├── labels/
│   ├── train/   # .txt files
│   └── val/
└── data.yaml    # สร้างอัตโนมัติ
```

Label format (.txt): `class_id cx cy w h` (normalize 0-1)
```
0 0.512 0.345 0.089 0.102   # bolt
1 0.234 0.678 0.075 0.088   # nut
```

### Train
```bash
python train_model.py train \
  --dataset dataset/ \
  --classes bolt,nut,washer,screw \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --base yolov8n.pt
```

### Export → ONNX → Hailo HEF
```bash
python train_model.py export --model models/best.pt

# แปลง ONNX → HEF ด้วย Hailo SDK:
hailomz compile yolov8n \
  --ckpt models/best.onnx \
  --hw-arch hailo8l \
  --calib-path dataset/images/val \
  --output-dir models/
```

### Validate accuracy
```bash
python train_model.py validate \
  --model models/best.pt \
  --yaml dataset/data.yaml
```

---

## 🔧 Config Station แต่ละจุด

แก้ `station_config.json`:
```json
{
  "station_id": "STATION-01",
  "expected_counts": {
    "bolt": 4,
    "nut": 4,
    "washer": 8
  },
  "conf_threshold": 0.45,
  "use_hailo": true
}
```

---

## 🤖 Auto-start บน Pi (systemd)

```bash
# สร้าง service
sudo nano /etc/systemd/system/bolt-counter.service
```

```ini
[Unit]
Description=Bolt Counter AI Station
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/bolt_counter
ExecStart=/home/pi/bolt_counter/venv/bin/python web_app.py --camera 0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable bolt-counter
sudo systemctl start bolt-counter
sudo systemctl status bolt-counter
```

---

## 📊 Class IDs

| ID | Class  | Color       |
|----|--------|-------------|
| 0  | bolt   | สีฟ้าเหลือง |
| 1  | nut    | สีเขียว     |
| 2  | washer | สีส้ม       |
| 3  | screw  | สีม่วง      |

---

## 💡 Tips

- **กล้อง**: ใช้ Fixed focus, ไฟ LED ring สม่ำเสมอ, พื้นหลังตัดกัน
- **ความแม่น**: ถ่าย 100+ ภาพต่อ class, หลายมุม/แสง
- **Hailo-8L**: เร็วกว่า CPU ประมาณ 10-15x (~25-50ms/frame)
- **Calibration**: ใช้ `--conf 0.45` เป็นค่าเริ่มต้น ปรับตามการทดสอบจริง

---

## 🔗 Resources

- [Raspberry Pi AI HAT+ docs](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html)
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Roboflow](https://roboflow.com/) — annotate + export dataset
