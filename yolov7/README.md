## YOLOv7 

### Hardware friendly changes
- Change the activation operation from SiLU to HardSiLU in quantization

### Prepare

#### Prepare the environment

##### For vitis-ai docker user
```bash
conda activate vitis-ai-pytorch
pip install -r yolov7/requirements.txt
```

##### Others
```bash
conda create -n yolov7 python=3.7
conda activate yolov7
pip install -r yolov7/requirements.txt
```

#### Prepare the dataset
Put coco2017 dataset under the ./yolov7/coco directory, dataset directory structure like:
```markdown
+ yolov7/coco/
    + labels/
    + annotations/
    + images/
    + test-dev2017.txt 
    + train2017.txt
    + val2017.txt
```

### For yolov7 Eval/QAT


#### Eval float model
```bash
cd yolov7/
python test_nndct.py --data data/coco.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val --quant_mode float
```

#### Run Post-training quantisation
```bash
cd yolov7/
# run calibration & test & dump xmodel
python test_nndct.py --data dataset/license_data/license_plates.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --device 0 --weights dataset/yolov7n_best.pt --name yolov7_640_val --quant_mode calib --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish

python test_nndct.py --data dataset/license_data/license_plates.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --device 0 --weights dataset/yolov7n_best.pt --name yolov7_640_val --quant_mode calib --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish
```

#### Dump PTQ model
```bash
cd yolov7/
python test_nndct.py --data data/coco.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val --quant_mode test --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish --dump_model
```

#### Quantization aware training 
```bash
cd yolov7/
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9004 train_qat.py --workers 8 --device 0,1,2,3 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights yolov7.pt --name yolov7_qat --hyp data/hyp.scratch.p5_qat.yaml --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish --log_threshold_scale 100
```

#### Run Quantization aware training model quantization
```bash
cd yolov7/
python test_nndct.py --data data/coco.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/yolov7_qat/weights/best.pt --name yolov7_640_val --quant_mode test --nndct_qat --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish
```

#### Dump QAT model
```bash
cd yolov7/
python test_nndct.py --data data/coco.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/yolov7_qat/weights/best.pt --name yolov7_640_val --quant_mode test --nndct_qat --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish --dump_model
```
