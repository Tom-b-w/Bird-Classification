import os
from pydub import AudioSegment

# 设置文件夹路径
folder_path = './audio_origin/environment'

# 目标音频时长（毫秒）
target_duration = 10 * 1000  # 10秒转换为毫秒

# 遍历文件夹中的所有音频文件
for filename in os.listdir(folder_path):
    if filename.endswith('.wav') or filename.endswith('.mp3'):
        # 获取文件路径
        file_path = os.path.join(folder_path, filename)
        try:
            # 加载音频文件
            audio = AudioSegment.from_file(file_path)

            # 获取音频时长
            duration = len(audio)

            if duration > target_duration:
                # 如果音频超过10秒，裁剪音频
                audio = audio[:target_duration]
                print(f"{filename} 裁剪至 10 秒")
            elif duration < target_duration:
                # 如果音频不足10秒，重复播放直到达到10秒
                repeat_count = target_duration // duration + 1
                audio = audio * repeat_count
                audio = audio[:target_duration]
                print(f"{filename} 重复播放至 10 秒")

            # 将音频音量降低 10 分贝
            audio = audio - 10  # 通过减去 10 分贝来降低音量

            # 保存处理后的音频文件，使用原始文件名
            output_path = os.path.join(folder_path, filename)  # 这里去掉了 'processed_' 前缀
            audio.export(output_path, format='wav')  # 你可以根据需要调整格式

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
