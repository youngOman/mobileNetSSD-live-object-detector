# MobileNetSSD 即時物件偵測程式

基於 MobileNetSSD & OpenCV 的深度學習模型的即時物件辨識程式，可從攝影機或指定的影片檔中中偵測並標註常見物件，如人、車、機車、巴士等

## Features

- 可以選擇辨識即時攝影機或影片檔案輸入的內容
- 使用預訓練 MobileNet SSD 深度學習模型
- 自動標註物件邊界框與信心分數
- 可自定義閾值、調整偵測信心度參數

## Info

- 模型：MobileNetSSD（Caffe 預訓練）
- 目前程式可偵測物件類別如 `self.CLASSES` 陣列中所示

# Demo

![demo](https://github.com/youngOman/mobileNetSSD-live-object-detector/blob/main/images/video_detect_result.png)   
