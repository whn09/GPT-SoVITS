# conda create -n sovits python==3.10
# conda activate sovits
# git clone https://github.com/RVC-Boss/GPT-SoVITS.git
# cd GPT-SoVITS
# pip install -r requirements.txt
# sudo apt install -y ffmpeg
# rm -rf GPT_SoVITS/pretrained_models/.gitignore
# git lfs install
# git clone https://huggingface.co/lj1995/GPT-SoVITS GPT_SoVITS/pretrained_models
# wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip -O GPT_SoVITS/text/G2PWModel_1.1.zip
# unzip GPT_SoVITS/text/G2PWModel_1.1.zip -d GPT_SoVITS/text/

import io
import os
import wave
import numpy as np
import soundfile as sf

from inference_webui import i18n, dict_language, cut1, cut2, cut3, cut4, cut5, process_text, get_tts_wav


def text_to_speech_pcm_stream(text, ref_audio_path, language=i18n("中文"), 
                             prompt_text=None, prompt_language=i18n("中文"),
                             speed=1.0, cut_method=i18n("不切"), 
                             sample_rate=44100, bit_depth=16):
    """
    使用GPT-SoVITS进行文本到语音的PCM流式转换
    
    参数:
        text (str): 需要转换的文本
        ref_audio_path (str): 参考音频文件路径(3-10秒)
        language (str): 文本语言，默认"中文" 
        prompt_text (str): 参考音频的文本，可选
        prompt_language (str): 参考音频语言，默认"中文"
        speed (float): 语速,范围0.6-1.65,默认1.0
        cut_method (str): 文本切分方法,默认"不切"
        sample_rate (int): 采样率，默认44100
        bit_depth (int): 位深度，默认16
        
    返回:
        generator: 生成器对象,持续输出PCM音频数据
    """
    
    # 输入检查
    if not os.path.exists(ref_audio_path):
        raise ValueError("参考音频文件不存在")
    if not text or not text.strip():
        raise ValueError("请输入有效的文本")
        
    # 检查语言选项是否有效
    valid_languages = list(dict_language.keys())
    if language not in valid_languages:
        raise ValueError(f"不支持的语言选项: {language}")
    if prompt_language not in valid_languages:
        raise ValueError(f"不支持的参考音频语言选项: {prompt_language}")
    
    # 设置是否无参考文本模式
    ref_free = (prompt_text is None or len(prompt_text.strip()) == 0)
    
    try:
        generator = get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            text=text,
            text_language=language,
            how_to_cut=cut_method,
            top_k=15,
            top_p=1,
            temperature=1,
            ref_free=ref_free,
            speed=speed,
            if_freeze=False,
            inp_refs=None
        )
        
        # 处理每个生成的音频片段
        for sampling_rate, audio_data in generator:
            # print('sampling_rate:', sampling_rate)
            # print('audio_data:', np.max(audio_data), np.min(audio_data), np.mean(audio_data))
            
            # # 分块输出PCM数据
            # for i in range(0, len(audio_data), CHUNK_SIZE):
            #     chunk = audio_data[i:i + CHUNK_SIZE]
            #     yield chunk.tobytes()
            
            # 创建内存中的WAV文件
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(bit_depth // 8)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # 返回WAV数据
            yield wav_buffer.getvalue()
                            
            # # 保存为WAV文件
            # sf.write("output.wav", audio_data, 32000)
            # break
                
    except Exception as e:
        print(f"生成文本片段时出错: {text}")
        print(f"错误信息: {str(e)}")

def save_pcm_stream(generator, output_path):
    with open(output_path, 'wb') as f:
        for chunk in generator:
            f.write(chunk)

def save_wav_stream(generator, output_path):
    """
    保存WAV流到文件
    
    参数:
        generator: WAV数据生成器
        output_path: 输出文件路径
    """
    first_chunk = True
    wav_file = None
    
    try:
        for wav_data in generator:
            if first_chunk:
                # 第一个chunk，直接写入文件
                with open(output_path, 'wb') as f:
                    f.write(wav_data)
                first_chunk = False
            else:
                # 后续chunk，需要提取音频数据并追加
                with wave.open(io.BytesIO(wav_data), 'rb') as wav_read:
                    # 读取音频数据（跳过WAV头）
                    audio_data = wav_read.readframes(wav_read.getnframes())
                    
                    # 追加到输出文件
                    with wave.open(output_path, 'rb+') as wav_write:
                        # 移动到文件末尾
                        wav_write.setpos(wav_write.getnframes())
                        # 写入新数据
                        wav_write.writeframes(audio_data)
        
        print(f"Successfully saved WAV file to {output_path}")
        
    except Exception as e:
        print(f"Error saving WAV file: {str(e)}")
        raise
    
# 使用示例：
"""
# 1. 基础用法 - 直接写入PCM文件
def save_pcm_stream(generator, output_path):
    with open(output_path, 'wb') as f:
        for chunk in generator:
            f.write(chunk)

save_pcm_stream(
    text_to_speech_pcm_stream(
        text="这是一个测试文本",
        ref_audio_path="path/to/reference.wav"
    ),
    "output.pcm"
)

# 2. 实时播放PCM流
import pyaudio

def play_pcm_stream(generator, sample_rate=44100, channels=1, format=pyaudio.paInt16):
    p = pyaudio.PyAudio()
    
    # 打开音频流
    stream = p.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        output=True
    )
    
    try:
        # 逐块播放音频
        for chunk in generator:
            stream.write(chunk)
    finally:
        # 清理资源
        stream.stop_stream()
        stream.close()
        p.terminate()

# 播放示例
play_pcm_stream(
    text_to_speech_pcm_stream(
        text="这是一个测试文本",
        ref_audio_path="path/to/reference.wav",
        sample_rate=44100,
        bit_depth=16
    )
)

# 3. 网络传输示例 - 服务器端
from flask import Flask, Response

app = Flask(__name__)

@app.route('/stream_audio')
def stream_audio():
    def generate():
        for chunk in text_to_speech_pcm_stream(
            text="这是一个测试文本",
            ref_audio_path="path/to/reference.wav"
        ):
            yield chunk
    
    return Response(generate(), mimetype='audio/pcm')

if __name__ == '__main__':
    app.run()
"""

def main():
    # sovits_path="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    # gpt_path="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    # change_sovits_weights(sovits_path)
    # change_gpt_weights(gpt_path)

    # save_pcm_stream(
    #     text_to_speech_pcm_stream(
    #         text="喂，你好！歡迎致電安樂貸款公司，我係Amy，請問有咩可以幫到你？(Hello, welcome to Peace of Mind Lending Company. This is Amy, how may I help you?)",
    #         ref_audio_path="ref.wav",
    #         language=i18n("多语种混合(粤语)")
    #     ),
    #     "output.pcm"
    # )
    save_wav_stream(
        text_to_speech_pcm_stream(
            text="喂，你好！歡迎致電安樂貸款公司，我係Amy，請問有咩可以幫到你？(Hello, welcome to Peace of Mind Lending Company. This is Amy, how may I help you?)",
            ref_audio_path="ref.wav",
            language=i18n("多语种混合(粤语)"),
            sample_rate=32000
        ),
        "output.wav"
    )

if __name__ == "__main__":
    main()