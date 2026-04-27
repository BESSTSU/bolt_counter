py -3.12 folder_test.py ^
  --images  dataset/valid/images/ ^
  --labels  dataset/valid/labels/ ^
  --conf    0.35 ^
  --iou     0.3 ^
  --model   runs/detect/bolt_counter13/weights/best.pt
  --no-hailo