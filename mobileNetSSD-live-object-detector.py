
'''
# 使用方式：
python mobileNetSSD-live-object-detector.py \
    --prototxt MobileNetSSD_deploy.prototxt.txt \
    --model MobileNetSSD_deploy.caffemodel

其他可用參數：
	--input video.mp4
    

--prototxt MobileNetSSD_deploy.prototxt.txt：指定網路架構檔案，定義神經網路的層次結構、連接方式、輸入輸出格式等，是 Caffe 框架的網路定義格式
--model MobileNetSSD_deploy.caffemodel：指定預訓練模型權重檔案，.caffemodel 包含已經訓練好的神經網路權重參數
'''
import cv2
import numpy as np
import argparse
import time
from imutils.video import VideoStream, FPS


class ObjectDetector:
    """MobileNet SSD 物件偵測器"""

    def __init__(self, prototxt_path, model_path, confidence_threshold=0.2):
        self.confidence_threshold = confidence_threshold

        # MobileNet SSD 訓練時使用的類別標籤
        self.CLASSES = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]

        # 為每個類別隨機產生邊框顏色
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

        # 載入模型
        print("[資訊] 載入模型中...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    def detect_objects(self, frame):
        """偵測影格中的物件"""
        h, w = frame.shape[:2]

        # 轉換為 blob 格式
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )

        # 執行偵測
        self.net.setInput(blob)
        detections = self.net.forward()

        return self._process_detections(detections, w, h)

    def _process_detections(self, detections, width, height):
        """處理偵測結果"""
        results = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                class_idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                start_x, start_y, end_x, end_y = box.astype("int")

                results.append({
                    'class_idx': class_idx,
                    'class_name': self.CLASSES[class_idx],
                    'confidence': confidence,
                    'box': (start_x, start_y, end_x, end_y),
                    'color': self.COLORS[class_idx]
                })

        return results

    def draw_detections(self, frame, detections):
        """在影格上繪製偵測結果"""
        for detection in detections:
            start_x, start_y, end_x, end_y = detection['box']
            color = detection['color']

            # 繪製邊界框
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

            # 繪製標籤
            label = f"{detection['class_name']}: {detection['confidence']*100:.2f}%"
            y = start_y - 15 if start_y - 15 > 15 else start_y + 15
            cv2.putText(frame, label, (start_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame


def setup_video_source(input_path=None):
    """設定影片來源"""
    if not input_path:
        print("[資訊] 啟動攝影機串流...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        return vs, False
    else:
        print("[資訊] 開啟影片檔案...")
        return cv2.VideoCapture(input_path), True


def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="MobileNet SSD 即時物件偵測")
    parser.add_argument("-p", "--prototxt", required=True,
                        help="Caffe 'deploy' prototxt 檔案的路徑")
    parser.add_argument("-m", "--model", required=True,
                        help="Caffe 預訓練模型的路徑")
    parser.add_argument("-i", "--input", type=str,
                        help="（選擇性）輸入影片檔案的路徑")
    parser.add_argument("-c", "--confidence", type=float, default=0.2,
                        help="過濾低信心預測的最小信心閾值")
    return parser.parse_args()


def main():
    """主程式"""
    args = parse_arguments()

    # 初始化物件偵測器
    detector = ObjectDetector(
        args.prototxt,
        args.model,
        args.confidence
    )

    # 設定影片來源
    vs, is_file = setup_video_source(args.input)

    # 初始化 FPS 計數器
    fps = FPS().start()

    try:
        # 主要處理迴圈
        while True:
            # 讀取影格
            frame = vs.read()
            if is_file:
                frame = frame[1]

            # 如果讀取失敗（影片結束），跳出迴圈
            if frame is None:
                break

            # 偵測物件
            detections = detector.detect_objects(frame)

            # 繪製結果
            frame = detector.draw_detections(frame, detections)

            # 顯示影格
            cv2.imshow("物件偵測", frame)

            # 檢查是否按下 'q' 鍵
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            fps.update()

    finally:
        # 清理資源
        fps.stop()
        print(f"[資訊] 執行時間: {fps.elapsed():.2f} 秒")
        print(f"[資訊] 大約 FPS: {fps.fps():.2f}")

        cv2.destroyAllWindows()
        if hasattr(vs, 'stop'):
            vs.stop()
        else:
            vs.release()


if __name__ == "__main__":
    main()
