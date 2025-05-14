# 步骤1：将ONNX模型导入STM32Cube.AI
# 在Cube.AI中选择模型 → 分析计算量 → 生成优化后的C代码

# 步骤2：集成到STM32工程
- 调用`ai_gmm_predict()`函数进行推理。
- 使用CMSIS-DSP库加速概率计算（如指数运算、矩阵乘法）。

# 示例代码片段（STM32端）：
#include "ai_gmm.h"
float input_features[13];  // MFCC特征（13维）
ai_gmm_process(&gmm_model, input_features, &output_probability);