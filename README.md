# ultralytics_wangfan040803
目前使用的ultralytics版本为: 8.3.241  
时间：2025-12-17
## 教程
- [自定义模块/模型接入教程](docs/CUSTOM_ULTRALYTICS_MODULE_GUIDE.md)
## 主要修改内容

- 新增 ViT 预训练特征模块 `DinoV2Patches`：添加 `ultralytics/nn/modules/pretrained_vit.py`，并在 `ultralytics/nn/modules/__init__.py` 导出（含 `__all__` 注册），同时在 `ultralytics/nn/tasks.py` 导入并加入 `parse_model()` 的 `base_modules` 以支持在 YAML 中直接引用。
YAMl位置在：`yaml\yolo_dinov2_small.yaml`

