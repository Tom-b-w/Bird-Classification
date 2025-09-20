import os
import jieba
import time

import requests
from pydub import AudioSegment
import asyncio
from xenocanto import metadata, download, list_urls

'''
def extract_keywords(text):
    """使用 jieba 提取鸟类关键词"""
    print(f"text = {text}")
    words = set(jieba.lcut(text))  # 分词并去重
    bird_dict={'噪鹏', '鹤', '水', '鸦雀', '红脚鹬', '鹪莺', '隼', '麦鸡', '鸳鸯', '麻雀', '天鹅', '草鹬', '雷', '鸽', '雨燕',
     '绿头鸭', '苍鹰', '鸠', '雉鸡', '白鹭', '锦鸡', '苍鹭', '鹌鹑', '鸬鹚', '翠鸟', '燕雀', '骨顶鸡', '绿翅鸭', '莺',
     '潜鸟', '虫', '鹂', '叶', '鸥', '鹰雀', '风', '长脚鹬', '鵟', '喜鹊', '秧鸡', '戴胜', '八哥', '柳莺', '雀', '山雀',
     '鹎', '灰雁', '孔雀', '鹦鹉', '山鹑', '矶鹬', '林鹬'}

    return list(words & bird_dict)  # 取交集，筛选已知鸟类
'''
def extract_keywords(text):
    """遍历每个字，提取鸟类关键词"""
    print(f"text = {text}")

    # 已知的鸟类名称字典
    bird_dict = {
        '噪鹏', '鹤', '水', '鸦雀', '红脚鹬', '鹪莺', '隼', '麦鸡', '鸳鸯', '麻雀', '天鹅', '草鹬', '雷', '鸽', '雨燕',
        '绿头鸭', '苍鹰', '鸠', '雉鸡', '白鹭', '锦鸡', '苍鹭', '鹌鹑', '鸬鹚', '翠鸟', '燕雀', '骨顶鸡', '绿翅鸭', '莺',
        '潜鸟', '虫', '鹂', '叶', '鸥', '鹰雀', '风', '长脚鹬', '鵟', '喜鹊', '秧鸡', '戴胜', '八哥', '柳莺', '雀', '山雀',
        '鹎', '灰雁', '孔雀', '鹦鹉', '山鹑', '矶鹬', '林鹬','雨'
    }

    # 记录匹配到的鸟类关键词
    found_keywords = []

    # 遍历文本中的每个字
    for i in range(len(text)):
        # 以当前位置为起点，尝试取出每一个可能的子串（从当前位置到后面的每一个位置）
        for j in range(i + 1, len(text) + 1):
            word = text[i:j]
            if word in bird_dict and word not in found_keywords:  # 如果该子串是鸟类名称且没有重复添加
                found_keywords.append(word)

    return found_keywords  # 返回匹配到的所有鸟类关键词


def get_keywords_from_deepseek(text):
    prompt = f"从以下中文文本中提取鸟类名称，并将它们翻译成英文：\n\n{text}\n\n只输出这些名称的英文，不输出其他内容，并将结果用换行符隔开"

    payload = {
        "model": "Pro/deepseek-ai/DeepSeek-V3",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "stop": []
    }

    headers = {
        "Authorization": "Bearer YOUR_API_KEY",  # 请替换为你的 API 密钥
        "Content-Type": "application/json"
    }

    try:
        response = requests.post("https://api.siliconflow.cn/v1/chat/completions", json=payload, headers=headers)
        result = response.json()
        if 'choices' in result:
            keywords = result['choices'][0]['message']['content'].strip().split("\n")
            return keywords
        else:
            return []
    except Exception as e:
        print(f"Error in DeepSeek API: {e}")
        return []

# 使用 xeno-canto API 获取并下载音频
def download_audio_for_keywords(keywords):
    for keyword in keywords:
        print(f"开始下载与关键词 '{keyword}' 相关的音频...")
        # 下载音频
        asyncio.run(download([keyword, 'q:A', 'cnt:China']))  # 使用 xeno-canto 下载音频
'''
def mix_audio_files(file1_path, file2_path, destination_folder, aug_name, file2_volume=-20):
    """
    混合两个音频文件，将 file2 叠加到 file1（可调音量），输出合成后的音频。
    """
    try:
        # 加载音频文件
        sound1 = AudioSegment.from_file(file1_path, format="wav")
        sound2 = AudioSegment.from_file(file2_path, format="wav")

        # 调整 file2 的音量
        sound2 = sound2 + file2_volume

        # 计算循环次数并叠加 file2
        loop_count = len(sound1) // len(sound2) + 1
        sound2_looped = sound2 * loop_count  # 循环叠加
        sound2_looped = sound2_looped[:len(sound1)]  # 裁剪长度

        # 混音
        mixed = sound1.overlay(sound2_looped)

        # 生成输出文件路径
        file1_name = os.path.splitext(os.path.basename(file1_path))[0]
        destination_path = os.path.join(destination_folder, f"{file1_name}_{aug_name}.wav")

        # 导出混音文件
        mixed.export(destination_path, format="wav")
        print(f"✅ 混音文件已保存至: {destination_path}")
        return destination_path
    except Exception as e:
        print(f"❌ 混音过程中发生错误: {e}")
        return None


def process_text_and_mix(keywords, audio_folder, destination_folder):
    """
    提取文本中的鸟类关键词，找到对应音频文件，并进行混音。
    """
    print(f"关键词: {keywords}")
    print(f"音频文件夹: {audio_folder}")
    print(f"目标文件夹: {destination_folder}")

    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)

    # 选取第一个关键词作为基准音频
    base_bird = keywords[0]
    base_audio = os.path.join(audio_folder, f"{base_bird}.wav")
    print(f"基准音频: {base_audio}")

    if not os.path.exists(base_audio):
        print(f"⚠️ 基准音频 {base_bird}.wav 不存在，跳过混音")
        return None

    mixed_audio = base_audio  # 设定初始音频
    for bird in keywords[1:]:
        bird_audio = os.path.join(audio_folder, f"{bird}.wav")
        print(f"当前鸟类音频: {bird_audio}")

        if os.path.exists(bird_audio):
            # 生成唯一的混音名称
            timestamp = int(time.time())
            aug_name = f"{base_bird}_with_{bird}_{timestamp}"
            print(f"混音名称: {aug_name}")

            # 混音
            mixed_audio = mix_audio_files(mixed_audio, bird_audio, destination_folder, aug_name)
            print(f"生成的中间文件: {mixed_audio}")

            # 检查中间文件
            if not mixed_audio or not os.path.exists(mixed_audio):
                print(f"❌ 中间文件不存在: {mixed_audio}")
                return None
            try:
                sound = AudioSegment.from_file(mixed_audio, format="wav")
                print("✅ 中间文件格式有效")
            except Exception as e:
                print(f"❌ 中间文件格式无效: {e}")
                return None
        else:
            print(f"⚠️ 音频文件 {bird}.wav 不存在，跳过此鸟类")

    # 生成最终输出文件路径
    final_output = os.path.join(destination_folder, f"{'_'.join(keywords)}.wav")
    print(f"最终输出文件: {final_output}")

    try:
        # 导出最终混音文件
        sound = AudioSegment.from_file(mixed_audio, format="wav")
        sound.export(final_output, format="wav")
        print(f"✅ 最终混音文件已保存至: {final_output}")
        return final_output
    except Exception as e:
        print(f"❌ 导出最终音频时发生错误: {e}")
        return None
'''
import os
import time
from pydub import AudioSegment
import librosa
import soundfile as sf
import numpy as np
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


def mix_audio_files(file1_path, file2_path, destination_folder, aug_name, file2_volume=-20):
    """
    混合两个音频文件，将 file2 叠加到 file1（可调音量），输出合成后的音频。
    """
    try:
        # 加载音频文件
        sound1 = AudioSegment.from_file(file1_path, format="wav")
        sound2 = AudioSegment.from_file(file2_path, format="wav")

        # 调整 file2 的音量
        sound2 = sound2 + file2_volume

        # 计算循环次数并叠加 file2
        loop_count = len(sound1) // len(sound2) + 1
        sound2_looped = sound2 * loop_count  # 循环叠加
        sound2_looped = sound2_looped[:len(sound1)]  # 裁剪长度

        # 混音
        mixed = sound1.overlay(sound2_looped)

        # 生成输出文件路径
        file1_name = os.path.splitext(os.path.basename(file1_path))[0]
        destination_path = os.path.join(destination_folder, f"{file1_name}_{aug_name}.wav")

        # 导出混音文件
        mixed.export(destination_path, format="wav")
        print(f"✅ 混音文件已保存至: {destination_path}")
        return destination_path
    except Exception as e:
        print(f"❌ 混音过程中发生错误: {e}")
        return None


def process_text_and_mix(keywords, audio_folder, destination_folder):
    """
    提取文本中的鸟类关键词，找到对应音频文件，并进行混音。
    """
    print(f"关键词: {keywords}")
    print(f"音频文件夹: {audio_folder}")
    print(f"目标文件夹: {destination_folder}")

    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)

    # 选取第一个关键词作为基准音频
    base_bird = keywords[0]
    base_audio = os.path.join(audio_folder, f"{base_bird}.wav")
    print(f"基准音频: {base_audio}")

    if not os.path.exists(base_audio):
        print(f"⚠️ 基准音频 {base_bird}.wav 不存在，跳过混音")
        return None

    mixed_audio = base_audio  # 设定初始音频
    for bird in keywords[1:]:
        bird_audio = os.path.join(audio_folder, f"{bird}.wav")
        print(f"当前鸟类音频: {bird_audio}")

        if os.path.exists(bird_audio):
            # 生成唯一的混音名称
            timestamp = int(time.time())
            aug_name = f"{base_bird}_with_{bird}_{timestamp}"
            print(f"混音名称: {aug_name}")

            # 混音
            mixed_audio = mix_audio_files(mixed_audio, bird_audio, destination_folder, aug_name)
            print(f"生成的中间文件: {mixed_audio}")

            # 检查中间文件
            if not mixed_audio or not os.path.exists(mixed_audio):
                print(f"❌ 中间文件不存在: {mixed_audio}")
                return None
            try:
                sound = AudioSegment.from_file(mixed_audio, format="wav")
                print("✅ 中间文件格式有效")
            except Exception as e:
                print(f"❌ 中间文件格式无效: {e}")
                return None
        else:
            print(f"⚠️ 音频文件 {bird}.wav 不存在，跳过此鸟类")

    # 生成最终输出文件路径
    final_output = os.path.join(destination_folder, f"{'_'.join(keywords)}.wav")
    print(f"最终输出文件: {final_output}")

    try:
        # 导出最终混音文件
        sound = AudioSegment.from_file(mixed_audio, format="wav")
        sound.export(final_output, format="wav")
        print(f"✅ 最终混音文件已保存至: {final_output}")
        return final_output
    except Exception as e:
        print(f"❌ 导出最终音频时发生错误: {e}")
        return None
