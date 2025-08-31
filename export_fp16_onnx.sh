# python export.py --weights yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640 --fp16
pip install onnx
cd yolov7/
python export.py --weights yolov7.pt --grid --end2end --simplify --topk-all 10000 --iou-thres 0.65 --conf-thres 0.001 --img-size 640 640 --max-wh 640 --fp16 --device 0
cd ../