import numpy as np
import librosa
import soundfile as sf
import random
from scipy.signal import butter, lfilter


class OptimizedFourierBirdSoundComposer:
    def __init__(self, target_duration: int = 10 * 1000, sample_rate: int = 16000):
        self.target_duration = target_duration  # 目标音频时长（毫秒）
        self.sample_rate = sample_rate

    def combine_bird_sounds(self, audio_files: list, output_file: str = 'combined_bird_sounds.wav'):
        """
        使用傅里叶变换将多个音频文件合并为一个 10 秒钟的多鸟鸣音频。

        Args:
            audio_files: 要合并的音频文件路径列表
            output_file: 输出合成后的音频文件路径
        """
        combined_freqs = None
        total_duration = 0

        # 为每个音频文件设置频域处理
        for audio_file in audio_files:
            try:
                # 加载音频
                audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                duration = len(audio) / sr
                total_duration += duration

                # 执行傅里叶变换
                freqs = np.fft.fft(audio)

                # 为每个音频频谱引入随机音量调节，以便每个鸟鸣音频的存在感不同
                volume_factor = random.uniform(0.8, 1.2)
                freqs *= volume_factor

                if combined_freqs is None:
                    combined_freqs = freqs
                else:
                    # 在频域上进行叠加
                    combined_freqs += freqs

            except Exception as e:
                print(f"❌ 错误: 无法加载音频文件 {audio_file}: {e}")

        # 处理合成后的频域信号
        if combined_freqs is not None:
            # 合成后的音频长度
            combined_audio = np.fft.ifft(combined_freqs).real

            # 对频谱进行平滑处理（例如使用低通滤波器）
            combined_audio = self._apply_lowpass_filter(combined_audio)

            # 确保最终音频的时长为 10 秒
            combined_audio = self._adjust_audio_length(combined_audio)

            # 保存合成后的音频文件
            sf.write(output_file, combined_audio, self.sample_rate)
            print(f"🎵 合成完成，文件保存为: {output_file}")

        else:
            print("❌ 无法进行傅里叶合成，未找到有效音频")

    def _adjust_audio_length(self, audio: np.ndarray) -> np.ndarray:
        """
        确保音频长度为目标时长
        """
        target_samples = self.target_duration * self.sample_rate // 1000
        audio_length = len(audio)

        if audio_length > target_samples:
            # 如果音频长度超过目标时长，进行裁剪
            return audio[:target_samples]
        elif audio_length < target_samples:
            # 如果音频长度不足目标时长，进行重复
            repeat_count = target_samples // audio_length + 1
            audio = np.tile(audio, repeat_count)
            return audio[:target_samples]
        return audio

    def _apply_lowpass_filter(self, audio: np.ndarray, cutoff: float = 3000.0, order: int = 6) -> np.ndarray:
        """
        对音频应用低通滤波器，平滑频谱
        """
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, audio)


# 使用示例
if __name__ == "__main__":
    # 创建合成器
    composer = OptimizedFourierBirdSoundComposer(target_duration=10 * 1000)

    # 输入音频文件路径列表
    audio_files = [
        "./audio_origin/birds/八哥.wav",
        "./audio_origin/birds/绿头鸭.wav",
    ]

    # 输出合成后的音频文件路径
    output_file = "./audio_origin/birds/output_birds.wav"

    # 合成多鸟鸣音频
    composer.combine_bird_sounds(audio_files, output_file)
