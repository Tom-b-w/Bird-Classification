import hashlib
import smtplib
import time
from email.mime.text import MIMEText
from captcha.image import ImageCaptcha
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, session, \
    make_response
import os
import numpy as np
from flask_sqlalchemy import SQLAlchemy
import random
import string
from flask_mail import Mail, Message
from process_audio import extract_keywords, process_text_and_mix, get_keywords_from_deepseek
from utils import trainer
from utils.trainer import load_model
from models.model_utils import get_model
from datasets.dataset_utils import get_dataloaders
from utils.utils import get_optimizer
import torch
from email.utils import formataddr

# 添加多标签识别相关导入
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Flask(__name__, static_folder="static")
app.secret_key = 'your_secret_key'  # 设置密钥用于会话管理

# 多标签识别配置
MAX_DURATION = 3
SAMPLING_RATE = 16000
MAX_SEQ_LENGTH = MAX_DURATION * SAMPLING_RATE
MODEL_CHECKPOINT = "facebook/wav2vec2-base"
NUM_CLASSES = 10  # 多标签模型类别数

# 多标签模型的类别映射
cls_label = {
    '灰雁': 0, '大天鹅': 1, '绿头鸭': 2, '灰山鹑': 3,
    '雉鸡': 4, '红喉潜鸟': 5, '苍鹭': 6, '普通鸬鹚': 7, '红脚鹬': 8,
    '麻雀': 9
}

label_to_species = {
    0: '灰雁', 1: '大天鹅', 2: '绿头鸭', 3: '灰山鹑',
    4: '雉鸡', 5: '红喉潜鸟', 6: '苍鹭', 7: '普通鸬鹚', 8: '红脚鹬',
    9: '麻雀'
}


# 定义多标签模型类 - 与实际训练代码的层命名完全一致
class Wav2Vec2MultiLabelClassifier(torch.nn.Module):
    """基于Wav2Vec2的多标签分类器 - 使用实际训练时的层命名"""

    def __init__(self, model_checkpoint, num_classes, freeze_wav2vec=False):
        super(Wav2Vec2MultiLabelClassifier, self).__init__()

        # 加载预训练的Wav2Vec2模型
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_checkpoint)

        # 是否冻结Wav2Vec2参数
        if freeze_wav2vec:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        # 使用与训练代码完全一致的层命名
        self.dropout1 = torch.nn.Dropout(0.3)

        # 分类头层 - 使用训练时的确切命名
        self.dense1 = torch.nn.Linear(self.wav2vec2.config.hidden_size, 1024)
        self.dense2 = torch.nn.Linear(1024, 512)
        self.dense3 = torch.nn.Linear(512, 256)
        self.final_layer = torch.nn.Linear(256, num_classes)

    def forward(self, input_values):
        # 通过Wav2Vec2提取特征
        outputs = self.wav2vec2(input_values)

        # 全局平均池化
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)

        # 通过分类头 - 使用与训练代码一致的前向传播
        x = self.dropout1(pooled_output)

        x = torch.nn.functional.relu(self.dense1(x))
        x = torch.nn.functional.dropout(x, p=0.3, training=self.training)

        x = torch.nn.functional.relu(self.dense2(x))
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)

        x = torch.nn.functional.relu(self.dense3(x))

        logits = torch.sigmoid(self.final_layer(x))  # 多标签分类使用sigmoid

        return logits


# 加载多标签模型和特征提取器
def load_multilabel_model(model_path='final_wav2vec2_model.pth'):
    """加载多标签分类模型"""
    try:
        # 创建模型实例
        model = Wav2Vec2MultiLabelClassifier(MODEL_CHECKPOINT, NUM_CLASSES, freeze_wav2vec=False)

        # 加载状态字典
        if os.path.exists(model_path):
            # 加载检查点文件
            checkpoint = torch.load(model_path, map_location=device)

            # 检查是否是完整的检查点格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 如果是训练检查点格式，提取模型状态字典
                state_dict = checkpoint['model_state_dict']
                print(f"从训练检查点加载模型: {model_path}")
                print(f"训练历史信息: Epoch {checkpoint.get('epoch', 'N/A')}, "
                      f"验证准确率: {checkpoint.get('val_accuracy', 'N/A'):.4f}, "
                      f"验证损失: {checkpoint.get('val_loss', 'N/A'):.4f}")
            else:
                # 如果是纯模型权重格式
                state_dict = checkpoint
                print(f"从模型权重文件加载: {model_path}")

            # 加载模型权重
            model.load_state_dict(state_dict)
            print(f"✅ 多标签模型加载成功")
        else:
            print(f"❌ 多标签模型文件不存在: {model_path}")
            return None, None

        # 移动到设备并设置为评估模式
        model.to(device)
        model.eval()

        # 创建特征提取器
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

        return model, feature_extractor
    except Exception as e:
        print(f"❌ 加载多标签模型失败: {str(e)}")
        return None, None


# 音频预处理函数
def get_audio_for_multilabel(path, max_seq_length=MAX_SEQ_LENGTH, sampling_rate=SAMPLING_RATE):
    """多标签模型的音频加载函数"""
    try:
        y, sr = librosa.load(path, sr=sampling_rate)
        if len(y) > max_seq_length:
            y = y[:max_seq_length]
        else:
            y = np.pad(y, (0, max_seq_length - len(y)), mode='constant')
        return y
    except Exception as e:
        print(f"加载音频文件出错 {path}: {e}")
        return np.zeros(max_seq_length)


def preprocess_audio_for_multilabel(audio_data, feature_extractor):
    """使用Wav2Vec2特征提取器预处理音频"""
    try:
        inputs = feature_extractor(
            audio_data,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=True
        )
        return inputs['input_values']
    except Exception as e:
        print(f"音频预处理失败: {str(e)}")
        return None


# 多标签识别函数
def recognize_multiple_birds(audio_file_path, model, feature_extractor, threshold=0.5):
    """识别音频中的多种鸟类"""
    try:
        # 使用多标签模型的音频加载方式
        audio_data = get_audio_for_multilabel(audio_file_path)

        # 预处理音频
        processed = preprocess_audio_for_multilabel(audio_data, feature_extractor)
        if processed is None:
            return None

        # 移动到设备并推理
        processed = processed.to(device)
        with torch.no_grad():
            outputs = model(processed)

        # 获取结果
        outputs = outputs.cpu().numpy()[0]

        # 应用阈值筛选结果
        results = []
        for species_name, label_idx in cls_label.items():
            score = outputs[label_idx]
            if score >= threshold:
                results.append({
                    'species': species_name,
                    'confidence': float(score) * 100
                })

        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results
    except Exception as e:
        print(f"多标签识别过程出错: {str(e)}")
        return None


UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
SMTP_SERVER = 'smtp.qq.com'
SMTP_PORT = 465
EMAIL_SENDER = 'your-email'
EMAIL_PASSWORD = '**********'  # 注意：是授权码不是登录密码！

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///default.db'
mail = Mail(app)

# 配置数据库
app.config['SQLALCHEMY_BINDS'] = {
    'birds': 'sqlite:///birds_info.db',  # 存储鸟类信息
    'user': 'sqlite:///user.db'  # 存储用户信息
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
email_codes = {}
code_expiry = 5 * 60  # 验证码有效期（单位：秒）


# 数据库模型
class BirdInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lat = db.Column(db.Float, nullable=False)
    lon = db.Column(db.Float, nullable=False)
    bird_name = db.Column(db.String(100), nullable=False)
    audio_file = db.Column(db.String(200), nullable=False)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)


# 创建数据库表（如果没有）
with app.app_context():
    db.session.query(User).delete()  # 删除 User 表中的所有记录
    db.session.commit()
    db.create_all()

SUPPORTED_KEYWORDS = {'灰雁', '天鹅', '绿头鸭', '山鹑', '鹌鹑', '雉鸡', '潜鸟', '苍鹭', '鸬鹚', '苍鹰',
                      '鵟', '秧鸡', '麦鸡', '红脚鹬', '麻雀', '八哥', '山雀', '鸦雀',
                      '柳莺', '噪鹏', '鹎', '戴胜', '白鹭', '雨燕', "风", "雷", "雨", "虫", "水", "叶"}

# 完整的鸟类信息字典
COMPLETE_BIRD_INFO = {
    '灰雁': {
        'description': '灰雁是一种大型水鸟，主要栖息在湿地和湖泊中。',
        'image': 'images/gray_geese.png',
        'sound': 'sounds/gray_geese.wav'
    },
    '大天鹅': {
        'description': '大天鹅是北半球较大的水鸟，以其优美的姿态著称。',
        'image': 'images/swan.png',
        'sound': 'sounds/swan.wav'
    },
    '绿头鸭': {
        'description': '绿头鸭是常见的水鸟，雄性头部呈亮绿色，栖息在湖泊和湿地。',
        'image': 'images/mallard_duck.png',
        'sound': 'sounds/mallard_duck.wav'
    },
    '绿翅鸭': {
        'description': '绿翅鸭是一种栖息在湿地中的小型水鸟，雄性具有独特的绿色翅膀。',
        'image': 'images/green_winged_duck.png',
        'sound': 'sounds/green_winged_duck.wav'
    },
    '灰山鹑': {
        'description': '灰山鹑栖息在山地和干草原，是一种体型较小的鸟类。',
        'image': 'images/grey_partridge.png',
        'sound': 'sounds/grey_partridge.wav'
    },
    '西鹌鹑': {
        'description': '西鹌鹑是一种体型小巧、栖息在草地和灌丛中的鸟类。',
        'image': 'images/western_quail.png',
        'sound': 'sounds/western_quail.wav'
    },
    '雉鸡': {
        'description': '雉鸡是具有艳丽羽毛的鸟类，常见于森林和草原中。',
        'image': 'images/pheasant.png',
        'sound': 'sounds/pheasant.wav'
    },
    '红喉潜鸟': {
        'description': '红喉潜鸟栖息在湖泊和湿地中，具有鲜艳的红色喉部。',
        'image': 'images/red_throated_diver.png',
        'sound': 'sounds/red_throated_diver.wav'
    },
    '苍鹭': {
        'description': '苍鹭是大型水鸟，通常栖息在湿地和沼泽地带，以鱼类为食。',
        'image': 'images/grey_heron.png',
        'sound': 'sounds/grey_heron.wav'
    },
    '普通鸬鹚': {
        'description': '普通鸬鹚是常见的水鸟，栖息在湖泊和河流中，以捕鱼为主。',
        'image': 'images/cormorant.png',
        'sound': 'sounds/cormorant.wav'
    },
    '苍鹰': {
        'description': '苍鹰是一种大型猛禽，主要栖息在森林和山区，捕食其他小型鸟类。',
        'image': 'images/eurasian_hawk.png',
        'sound': 'sounds/eurasian_hawk.wav'
    },
    '欧亚鵟': {
        'description': '欧亚鵟是一种栖息在广阔草原和森林中的猛禽，主要捕食小型哺乳动物。',
        'image': 'images/eurasian_buzzard.png',
        'sound': 'sounds/eurasian_buzzard.wav'
    },
    '西方秧鸡': {
        'description': '西方秧鸡栖息在湿地和农田中，体型较小，动作敏捷。',
        'image': 'images/western_rail.png',
        'sound': 'sounds/western_rail.wav'
    },
    '骨顶鸡': {
        'description': '骨顶鸡栖息在湿地和水草丰富的区域，以水生植物为食。',
        'image': 'images/bonecrest_chicken.png',
        'sound': 'sounds/bonecrest_chicken.wav'
    },
    '黑翅长脚鹬': {
        'description': '黑翅长脚鹬栖息在湿地地区，具有较长的腿部和黑色的翅膀。',
        'image': 'images/black_winged_stilt.png',
        'sound': 'sounds/black_winged_stilt.wav'
    },
    '凤头麦鸡': {
        'description': '凤头麦鸡是一种栖息在草地和农田中的鸟类，拥有显著的头部羽冠。',
        'image': 'images/crested_lark.png',
        'sound': 'sounds/crested_lark.wav'
    },
    '白腰草鹬': {
        'description': '白腰草鹬栖息在湿地和草丛中，拥有鲜明的白色腰部羽毛。',
        'image': 'images/white_winged_snipe.png',
        'sound': 'sounds/white_winged_snipe.wav'
    },
    '红脚鹬': {
        'description': '红脚鹬是一种体型中等的鸟类，栖息在湿地和沼泽地带，常见于迁徙期。',
        'image': 'images/red_legged_snipe.png',
        'sound': 'sounds/red_legged_snipe.wav'
    },
    '林鹬': {
        'description': '林鹬栖息在森林和湿地中，擅长在湿地中觅食。',
        'image': 'images/wood_snipe.png',
        'sound': 'sounds/wood_snipe.wav'
    },
    '麻雀': {
        'description': '麻雀是一种小型鸟类，常见于城市和农田地区，栖息在建筑物附近。',
        'image': 'images/sparrow.jpg',
        'sound': 'sounds/sparrow.wav'
    },
    '八哥': {
        'description': '八哥是一种聪明的鸣禽，擅长模仿人类语言，常见于城市和乡村。',
        'image': 'images/myna.jpg',
        'sound': 'sounds/myna.wav'
    },
    '红头长尾山雀': {
        'description': '红头长尾山雀是一种小型鸟类，具有鲜艳的红色头部，活跃于森林和灌木丛。',
        'image': 'images/red_headed_tit.jpg',
        'sound': 'sounds/red_headed_tit.wav'
    },
    '棕头鸦雀': {
        'description': '棕头鸦雀是一种体型小巧的鸟类，主要分布在森林和灌木丛中。',
        'image': 'images/brown_parrotbill.jpg',
        'sound': 'sounds/brown_parrotbill.wav'
    },
    '黄腹柳莺': {
        'description': '黄腹柳莺是一种小型鸣禽，栖息在森林和灌木丛中，以昆虫为食。',
        'image': 'images/yellow_bellied_warbler.jpg',
        'sound': 'sounds/yellow_bellied_warbler.wav'
    },
    '褐头鹪莺': {
        'description': '褐头鹪莺是一种活跃的鸣禽，以昆虫为食，主要栖息在灌木丛和草地。',
        'image': 'images/brown_warbler.jpg',
        'sound': 'sounds/brown_warbler.wav'
    },
    '白颊噪鹏': {
        'description': '白颊噪鹏是一种生活在森林中的鸟类，以果实和昆虫为食，叫声响亮。',
        'image': 'images/noisy_pitta.jpg',
        'sound': 'sounds/noisy_pitta.wav'
    },
    '白头鹎': {
        'description': '白头鹎是一种常见的鸣禽，喜欢群居，主要分布在城市和农田附近。',
        'image': 'images/light_vented_bulbul.jpg',
        'sound': 'sounds/light_vented_bulbul.wav'
    },
    '矶鹬': {
        'description': '矶鹬是一种小型涉禽，栖息在沿海和淡水湿地，以小型水生动物为食。',
        'image': 'images/sandpiper.jpg',
        'sound': 'sounds/sandpiper.wav'
    },
    '戴胜': {
        'description': '戴胜是一种美丽的鸟类，头顶有醒目的羽冠，主要以昆虫为食。',
        'image': 'images/hoopoe.jpg',
        'sound': 'sounds/hoopoe.wav'
    },
    '小白鹭': {
        'description': '小白鹭是一种常见的涉禽，栖息在湿地和水域边缘，以鱼类和昆虫为食。',
        'image': 'images/little_egret.jpg',
        'sound': 'sounds/little_egret.wav'
    },
    '小白腰雨燕': {
        'description': '小白腰雨燕是一种高速飞行的鸟类，以昆虫为食，常见于空中滑翔。',
        'image': 'images/swift.jpg',
        'sound': 'sounds/swift.wav'
    }
}

# 游戏相关的鸟类数据
BIRD_GAME_DATA = COMPLETE_BIRD_INFO.copy()


@app.route('/show_users')
def user_list():
    users = User.query.all()  # 查询所有用户
    return render_template('show_users.html', users=users)


@app.route("/")
def home():
    return render_template("log.html")


@app.route('/send_code', methods=['POST'])
def send_code():
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({'error': 'Email cannot be empty'}), 400

    # 生成6位验证码
    code = ''.join(random.choices(string.digits, k=6))
    email_codes[email] = {
        'code': code,
        'time': time.time()
    }

    # 邮件的主题与内容
    subject = "注册验证码"
    body = f"{code}"
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['From'] = formataddr(('鸟鸣开发平台', EMAIL_SENDER))  # 设置发件人显示为匿名
    msg['To'] = email
    msg['Subject'] = subject

    # 使用SMTP_SSL发送邮件
    server = smtplib.SMTP_SSL("smtp.qq.com", 465)
    server.login(EMAIL_SENDER, EMAIL_PASSWORD)
    server.sendmail(EMAIL_SENDER, email, msg.as_string())
    server.quit()  # 关闭连接

    return jsonify({'message': '验证码已发送，请检查您的邮箱'})


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')
    code = data.get('code')

    if not all([email, username, password, code]):
        return jsonify({'error': '邮箱、用户名、密码和验证码不能为空'}), 400

    # 检查邮箱是否已经被注册
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({'error': '该邮箱已被注册'}), 400

    # 验证码验证部分
    stored_code = email_codes.get(email, {}).get('code')
    stored_time = email_codes.get(email, {}).get('time')

    if not stored_code or not stored_time:
        return jsonify({'error': '验证码已过期或未发送'}), 400

    if code != stored_code:
        return jsonify({'error': '验证码错误'}), 400

    if time.time() - stored_time > code_expiry:
        return jsonify({'error': '验证码已过期'}), 400

    # 加密密码
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    # 创建新用户
    new_user = User(email=email, username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': '注册成功', 'redirect': '/login'}), 201


@app.route('/login', methods=['GET'])
def login_page():
    return render_template("log.html")  # 渲染你的登录/注册页面


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')  # 获取用户名
    password = data.get('password')  # 获取密码
    captcha = data.get('captcha')  # 获取验证码

    if not username or not password or not captcha:
        return jsonify({'error': '用户名、密码和验证码不能为空'}), 400

    # 验证码校验逻辑（如果你有 session 或验证码校验机制）
    if captcha.lower() != session.get('captcha', '').lower():
        return jsonify({'error': '验证码错误'}), 400

    # 只通过用户名查找
    user = User.query.filter_by(username=username).first()

    if not user:
        return jsonify({'error': '用户名不存在'}), 404

    # 密码校验
    hashed_input = hashlib.sha256(password.encode()).hexdigest()
    if user.password != hashed_input:
        return jsonify({'error': '密码错误'}), 401

    return jsonify({'message': '登录成功', 'redirect': '/welcome'}), 200


@app.route('/welcome')
def welcome():
    return render_template('welcome.html')


@app.route('/index')
def index():
    return render_template('index.html')


from sqlalchemy import func


@app.route('/captcha')
def get_captcha():
    # 生成4个随机大写字母或数字
    captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    session['captcha'] = captcha_text  # 存储验证码到 session

    image = ImageCaptcha(width=140, height=60)
    data = image.generate(captcha_text)

    # 返回图片
    response = make_response(data.read())
    response.headers['Content-Type'] = 'image/png'
    return response


@app.route('/submit_bird_info_page')
def submit_bird_info_page():
    lat = float(request.args.get('lat', ''))
    lon = float(request.args.get('lon', ''))

    # 查找与用户点击经纬度差距小于1度的记录
    bird_info_records = BirdInfo.query.filter(
        func.abs(BirdInfo.lat - lat) <= 0.001,
        func.abs(BirdInfo.lon - lon) <= 0.001
    ).all()

    if bird_info_records:
        # 如果有相关记录，返回数据
        return render_template('submit_bird_info.html', bird_info_records=bird_info_records)
    else:
        # 如果没有相关记录，提示用户输入信息
        return render_template('submit_bird_info.html', lat=lat, lon=lon)


@app.route('/submit_bird_info', methods=['POST'])
def submit_bird_info():
    lat = request.form.get('lat')
    lon = request.form.get('lon')
    bird_name = request.form.get('bird_name')
    audio = request.files.get('audio')

    if audio:
        # 获取音频文件路径并保存
        audio_path = os.path.join(UPLOAD_FOLDER, audio.filename)
        audio.save(audio_path)
        print(f"音频已保存: {audio_path}")

    # 保存信息到数据库
    bird_info = BirdInfo(lat=lat, lon=lon, bird_name=bird_name, audio_file=audio.filename)
    db.session.add(bird_info)
    db.session.commit()

    # 跳转回鸟类信息页面
    return redirect(f'/submit_bird_info_page?lat={lat}&lon={lon}')


@app.route('/view_bird_info')
def view_bird_info():
    # 获取所有记录
    bird_info_records = BirdInfo.query.all()

    # 将记录传递给模板
    return render_template('view_bird_info.html', records=bird_info_records)


@app.route('/get_bird_data')
def get_bird_data():
    bird_spots = BirdInfo.query.all()  # 查询所有鸟类信息
    bird_data = [{
        'lat': bird.lat,
        'lon': bird.lon,
        'bird_name': bird.bird_name,
        'audio_file': bird.audio_file
    } for bird in bird_spots]
    return jsonify(bird_data)


from pathlib import Path
import shutil
import traceback
from flask import current_app, request, jsonify, render_template, url_for

@app.route('/synthesize', methods=['POST', 'GET'])
def synthesize():
    if request.method == 'GET':
        return render_template('synthesize.html')

    try:
        user_input = (request.form.get("message") or "").strip()
        if not user_input:
            return jsonify({"error": "Message cannot be empty"}), 400

        # 关键词
        keywords = extract_keywords(user_input) or get_keywords_from_deepseek(user_input) or ["默认鸟鸣"]

        # === 关键：所有路径都基于 app 根目录，使用 Path，避免相对路径与分隔符问题 ===
        app_root = Path(current_app.root_path).resolve()
        audio_folder = (app_root / "audio_origin" / "birds").resolve()         # 素材目录（绝对）
        static_root = (app_root / "static").resolve()                          # static 根（绝对）
        destination_folder = (static_root / "mixed_audio").resolve()           # 生成目录（绝对）
        destination_folder.mkdir(parents=True, exist_ok=True)

        # 生成音频（你的函数）
        final_audio = process_text_and_mix(keywords, str(audio_folder), str(destination_folder))

        # —— 统一把返回值转成“实际存在的文件路径” —— #
        # 如果返回 (path, ...)，取第一个
        if isinstance(final_audio, (tuple, list)):
            final_audio = final_audio[0]

        # 必须是字符串或 Path
        if not isinstance(final_audio, (str, Path)):
            raise ValueError(f"final_audio 不是路径，而是 {type(final_audio)}")

        final_path = Path(final_audio)
        # 如果是相对路径，按 destination_folder 解析；否则直接 resolve
        if not final_path.is_absolute():
            candidate = (destination_folder / final_path).resolve()
            final_path = candidate if candidate.exists() else (app_root / final_path).resolve()

        if not final_path.exists():
            raise FileNotFoundError(f"合成的音频文件不存在: {final_path}")

        # 确保文件在 static/ 下，不在就拷贝进去（便于 url_for('static', ...)）
        try:
            rel_to_static = final_path.relative_to(static_root)
        except ValueError:
            target = (destination_folder / final_path.name).resolve()
            if final_path != target:
                shutil.copy2(final_path, target)
            rel_to_static = target.relative_to(static_root)

        # 生成 URL：filename 一律用 POSIX 风格（/），url_for 会处理域名和前缀
        audio_url = url_for('static', filename=str(rel_to_static).replace("\\", "/"), _external=True)

        return jsonify({"audio_url": audio_url, "keywords": keywords}), 200

    except Exception as e:
        current_app.logger.error("synthesize failed: %s\n%s", e, traceback.format_exc())
        # 若没有 error.html 也不会再次报错
        try:
            return render_template("error.html", error_message=str(e), error_detail=traceback.format_exc()), 500
        except Exception:
            return jsonify({"error": str(e)}), 500



# 游戏路由
@app.route('/games')
def games_menu():
    """渲染游戏菜单页面"""
    return render_template('game_menu.html')


@app.route('/bird_game')
def bird_game():
    """渲染游戏页面"""
    return render_template('bird_game.html')


@app.route('/api/game/start', methods=['POST'])
def start_game():
    """开始新游戏，返回随机的鸟类数据"""
    try:
        # 随机选择6种鸟类进行游戏
        selected_birds = random.sample(list(BIRD_GAME_DATA.keys()), 6)

        game_data = {
            'birds': []
        }

        for bird_name in selected_birds:
            bird_info = BIRD_GAME_DATA[bird_name]
            game_data['birds'].append({
                'name': bird_name,
                'image': bird_info['image'],
                'sound': bird_info['sound'],
                'description': bird_info['description']
            })

        # 将游戏数据存储在session中
        session['current_game'] = game_data
        session['game_score'] = 0
        session['game_matches'] = []

        return jsonify({
            'success': True,
            'data': game_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/game/check_match', methods=['POST'])
def check_match():
    """检查连线是否正确"""
    try:
        data = request.get_json()
        sound_name = data.get('sound')
        bird_name = data.get('bird')

        if not sound_name or not bird_name:
            return jsonify({
                'success': False,
                'error': '参数不完整'
            }), 400

        # 检查当前游戏状态
        current_game = session.get('current_game')
        if not current_game:
            return jsonify({
                'success': False,
                'error': '游戏未开始'
            }), 400

        # 检查是否已经匹配过
        game_matches = session.get('game_matches', [])
        match_key = f"{sound_name}_{bird_name}"

        if match_key in game_matches:
            return jsonify({
                'success': False,
                'error': '已经匹配过了',
                'already_matched': True
            })

        # 检查匹配是否正确（音频文件名和鸟名应该对应）
        is_correct = False
        for bird in current_game['birds']:
            if bird['name'] == bird_name:
                # 检查音频文件名是否包含鸟名的拼音或对应关系
                if bird['sound'] == sound_name:
                    is_correct = True
                    break

        # 更新游戏状态
        if is_correct:
            session['game_score'] = session.get('game_score', 0) + 10
            game_matches.append(match_key)
            session['game_matches'] = game_matches

        # 检查游戏是否完成
        game_completed = len(game_matches) == len(current_game['birds'])

        return jsonify({
            'success': True,
            'correct': is_correct,
            'score': session.get('game_score', 0),
            'completed': game_completed,
            'total_matches': len(game_matches),
            'total_birds': len(current_game['birds'])
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/game/score')
def get_score():
    """获取当前得分"""
    return jsonify({
        'success': True,
        'score': session.get('game_score', 0),
        'matches': len(session.get('game_matches', [])),
        'total_birds': len(session.get('current_game', {}).get('birds', []))
    })


@app.route('/api/game/reset', methods=['POST'])
def reset_game():
    """重置游戏"""
    session.pop('current_game', None)
    session.pop('game_score', None)
    session.pop('game_matches', None)

    return jsonify({
        'success': True,
        'message': '游戏已重置'
    })


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/map')
def map_page():
    return render_template('map_page.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # 获取用户选择的鸟类数量
    bird_count = request.form.get('bird_count', '1')

    if file:
        file_path = os.path.join('./static/uploads', file.filename)
        file.save(file_path)

        # 根据用户选择决定使用单标签还是多标签识别
        if bird_count == '1':
            # 单标签识别 - 使用原有逻辑
            return predict_single_bird(file_path, file.filename)
        else:
            # 多标签识别 - 使用新的多标签模型
            return predict_multiple_birds(file_path, file.filename, bird_count)

    return redirect(url_for('index'))


def predict_single_bird(file_path, filename):
    """单鸟类识别"""
    try:
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 强制将模型加载到 CPU
        model_name = 'whisper'
        model = get_model(model_name='whisper', pre_trained=True)
        model = model.to(device)  # 确保模型在设备上运行

        criterion = torch.nn.CrossEntropyLoss()  # loss function
        train_dataloader, test_dataloader, labels_unique = get_dataloaders(
            './birdclef-preprocess/birdclef-preprocess/birdclef2023-dataset', True,
            spec_augment=True, seed=seed,
            batch_size=16, num_workers=4)
        optimizer = get_optimizer(model_name, model, 'fine-tuning', lr=0.01)

        # 加载预训练模型
        load_model(model, optimizer,
                   "./Training_MODEL_NAME/whisper/data-aug_spec-aug_fine-tuning/model_best.pt")
        print("\nEvaluating ...")
        predicted_label, probs, avg_loss = trainer.test(device, model, model_name, file_path,
                                                        labels_unique, criterion=None, true_label=None)

        bird_species = [
            '灰雁', '大天鹅', '绿头鸭', '绿翅鸭', '灰山鹑', '西鹌鹑', '雉鸡',
            '红喉潜鸟', '苍鹭', '普通鸬鹚', '苍鹰', '欧亚鵟', '西方秧鸡', '骨顶鸡',
            '黑翅长脚鹬', '凤头麦鸡', '白腰草鹬', '红脚鹬', '林鹬', '麻雀', '八哥',
            '红头长尾山雀', '棕头鸦雀', '黄腹柳莺', '褐头鹪莺', '白颊噪鹏',
            '白头鹎', '矶鹬', '戴胜', '小白鹭', '小白腰雨燕'
        ]

        predicted_species = bird_species[predicted_label]
        print(f"Predicted Label: {predicted_species}")

        # 获取鸟类的相关信息
        bird_data = COMPLETE_BIRD_INFO.get(predicted_species, {})

        # 构建结果数据
        bird_results = [{
            'species': predicted_species,
            'confidence': None,  # 单标签识别不显示置信度
            'image': bird_data.get('image', ''),
            'sound': bird_data.get('sound', ''),
            'description': bird_data.get('description', '')
        }]

        return render_template('result.html',
                               bird_results=bird_results,
                               is_multi_label=False,
                               bird_count='1',  # 单标签固定为1
                               filename=filename)

    except Exception as e:
        print(f"单标签识别错误: {str(e)}")
        return render_template('error.html', error_message=str(e))


def predict_multiple_birds(file_path, filename, bird_count):
    """多鸟类识别"""
    try:
        # 加载多标签模型
        multilabel_model, feature_extractor = load_multilabel_model()

        if multilabel_model is None or feature_extractor is None:
            # 如果多标签模型加载失败，回退到单标签识别
            print("多标签模型加载失败，回退到单标签识别")
            return predict_single_bird(file_path, filename)

        # 进行多标签识别
        threshold = 0.3  # 可以根据需要调整阈值
        results = recognize_multiple_birds(file_path, multilabel_model, feature_extractor, threshold)

        if not results:
            # 如果没有识别结果，返回提示
            return render_template('result.html',
                                   bird_results=[],
                                   is_multi_label=True,
                                   bird_count=bird_count,  # 传递用户选择的鸟类数量
                                   filename=filename)

        # 根据用户选择的数量过滤结果
        if bird_count == 'more':
            # 显示所有识别到的鸟类
            filtered_results = results
        else:
            # 显示指定数量的结果
            max_count = int(bird_count)
            filtered_results = results[:max_count]

        # 为每个识别结果添加详细信息
        bird_results = []
        for result in filtered_results:
            species = result['species']
            bird_data = COMPLETE_BIRD_INFO.get(species, {})

            bird_results.append({
                'species': species,
                'confidence': result['confidence'],
                'image': bird_data.get('image', ''),
                'sound': bird_data.get('sound', ''),
                'description': bird_data.get('description', '')
            })

        print(f"多标签识别结果: {len(bird_results)}种鸟类")

        return render_template('result.html',
                               bird_results=bird_results,
                               is_multi_label=True,
                               bird_count=bird_count,  # 传递用户选择的鸟类数量
                               filename=filename)

    except Exception as e:
        print(f"多标签识别错误: {str(e)}")
        # 出错时回退到单标签识别
        return predict_single_bird(file_path, filename)


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(host='0.0.0.0', port=5000, debug=True)
