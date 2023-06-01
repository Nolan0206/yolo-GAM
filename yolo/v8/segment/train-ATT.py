from ultralytics import YOLO
if __name__ == '__main__':
    # 加载模型
    model = YOLO("/mnt/YOLOv8_Segmentation_DeepSORT_Object_Tracking/ultralytics/models/v8/seg/yolov8s-seg-CBAM-C2f.yaml")  # 从头开始构建新模型
    # model = YOLO("yolov8s.pt")  # 加载预训练模型（推荐用于训练）

    # Use the model
    results = model.train(data="/mnt/YOLOv8_Segmentation_DeepSORT_Object_Tracking/ultralytics/yolo/v8/segment/dataset.location/data.yaml", epochs=80, batch=16, workers=8, close_mosaic=0, name='cfg')  # 训练模型
    # results = model.val()  # 在验证集上评估模型性能
    # results = model("https://ultralytics.com/images/bus.jpg")  # 预测图像
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
