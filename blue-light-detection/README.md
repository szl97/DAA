# 高速路高清摄像头蓝光检测

**算法**：图像增强 + 高亮光斑候选提取 + 形态学去噪 + 车辆目标检测 + 蓝光-车辆空间关联 + 干扰过滤。

## 功能

- 输入高速公路监控单帧图像；
- 检测画面中的车辆目标；
- 提取高亮蓝光区域；
- 判断蓝光区域是否属于车辆，而非路牌、路灯、天空或路面反光；
- 输出总车辆数、含蓝光车辆数量、蓝光来源类型；
- 生成可视化结果图。

## 使用方法

```bash
pip install -r requirements.txt
python main.py --input input/highway_720p.jpg
python main.py --input input/night_vehicle_road_0000.jpg --output output/night_result_0000.jpg
```

如果未安装 ultralytics 或未下载 YOLO 权重，程序会自动退化为传统轮廓候选检测，便于在普通环境中演示完整流程。

## 输出

- `output/result.jpg`：检测可视化结果；
- 控制台统计：总车辆数、含蓝光车辆数、蓝光区域数、被过滤干扰数。
