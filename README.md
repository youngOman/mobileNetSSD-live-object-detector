# MobileNetSSD Live Object Detector｜即時物件偵測器

用 OpenCV 與 MobileNetV1 SSD 模型開發的即時影像物件偵測程式，可從攝影機或指定的影片檔中中偵測並標註常見物件，如人、車、機車、巴士等

## Features 功能特色
- 可以選擇辨識即時攝影機或影片檔案輸入的內容
- 使用預訓練 MobileNet SSD 深度學習模型
- 自動標註物件邊界框與信心分數

## 模型資訊
- 模型：MobileNetSSD（Caffe 預訓練）
- 目前程式可偵測物件類別如 `self.CLASSES` 陣列中所示

