# In[1]
'''首先，我们生成一个包含噪声和局部放电信号的模拟超声波信号'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(0)

# 生成时间序列
sampling_rate = 1e6  # 采样率为1 MHz
duration = 0.01  # 信号持续时间为10 ms
t = np.linspace(0, duration, int(sampling_rate * duration))

# 生成噪声
noise = np.random.normal(0, 1, len(t))

# 生成局部放电信号 (40kHz 正弦波)
frequency = 40e3  # 局部放电信号频率为40 kHz
pd_signal = 0.5 * np.sin(2 * np.pi * frequency * t)

# 将局部放电信号添加到噪声中
signal = noise + pd_signal

# 绘制生成的信号
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label='超声波信号')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.title('模拟超声波信号')
plt.legend()
plt.show()

# In[2]
'''接下来，我们对信号进行处理，并提取特征（如峰值检测）'''

# 滤波器设计与应用
from scipy.signal import butter, filtfilt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 设定带通滤波器参数
lowcut = 30e3
highcut = 50e3

# 对信号进行带通滤波
filtered_signal = bandpass_filter(signal, lowcut, highcut, sampling_rate)

# 绘制滤波后的信号
plt.figure(figsize=(12, 6))
plt.plot(t, filtered_signal, label='滤波后的超声波信号')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.title('滤波后的超声波信号')
plt.legend()
plt.show()

# 特征提取：峰值检测
peaks, _ = find_peaks(filtered_signal, height=0.4)

# 绘制峰值检测结果
plt.figure(figsize=(12, 6))
plt.plot(t, filtered_signal, label='滤波后的超声波信号')
plt.plot(t[peaks], filtered_signal[peaks], 'x', label='检测到的峰值')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.title('超声波信号峰值检测')
plt.legend()
plt.show()

# In[3]
'''最后，我们对提取的特征进行分析，并判断是否存在局部放电信号'''

# 计算峰值的频率
peak_intervals = np.diff(t[peaks])
peak_frequencies = 1 / peak_intervals

# 输出检测到的峰值频率
print("检测到的峰值频率 (Hz):")
print(peak_frequencies)

# 判断是否存在局部放电信号
pd_frequency_range = (frequency - 1000, frequency + 1000)  # 设定局部放电信号频率范围 ±1kHz
pd_detected = any((pd_frequency_range[0] <= f <= pd_frequency_range[1]) for f in peak_frequencies)

if pd_detected:
    print("检测到局部放电信号!")
else:
    print("未检测到局部放电信号.")

