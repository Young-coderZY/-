from sklearn.mixture import GaussianMixture
import numpy as np

# 示例：加载预处理后的声纹特征（MFCC/GFCC）
# 假设数据集已标准化为二维数组（样本数×特征维度）
X = np.load("局放声纹特征.npy")  

# 初始化GMM模型（建议通过BIC/AIC自动选择高斯分量数）
n_components = 3  # 根据实验调整
gmm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=500)

# 训练模型
gmm.fit(X)

# 保存模型参数（均值、协方差、权重）
np.save("gmm_means.npy", gmm.means_)
np.save("gmm_covariances.npy", gmm.covariances_)
np.save("gmm_weights.npy", gmm.weights_)

# 导出为ONNX格式（需使用sklearn-onnx转换工具）
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(gmm, initial_types=initial_type)
with open("gmm_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())