"""
Comic Text Detection API Service
提供漫画文本检测的REST API接口
"""
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
import torch
import json
from inference import TextDetector
import base64
from utils.io_utils import NumpyEncoder
import urllib.request
import hashlib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 允许所有来源的跨域请求

# 配置
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 全局模型实例
model = None

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent

model_config = {
    'model_path': str(PROJECT_ROOT / 'data' / 'comictextdetector.pt'),
    'model_url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt',
    'model_md5': '5d0c4a7e1e7b8f5c6e8a2b3c4d5e6f7a',  # 需要根据实际下载的文件更新
    'input_size': 1024,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'half': False,
    'conf_thresh': 0.4,
    'nms_thresh': 0.35,
    'mask_thresh': 0.3
}


def download_model(url, dest_path):
    """下载模型文件并显示进度条"""
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = downloaded / total_size * 100 if total_size > 0 else 0
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        print(f'\rDownloading: {downloaded_mb:.1f}MB / {total_mb:.1f}MB ({percent:.1f}%)', end='', flush=True)

    print(f'Downloading model from {url}...')
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print('\nDownload completed!')
        return True
    except Exception as e:
        print(f'\nDownload failed: {e}')
        return False


def verify_model_md5(file_path, expected_md5):
    """验证模型文件的MD5哈希"""
    if not expected_md5:
        return True  # 如果没有提供MD5，跳过验证
    md5_hash = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5_hash.update(chunk)
        calculated_md5 = md5_hash.hexdigest()
        return calculated_md5.lower() == expected_md5.lower()
    except:
        return False


def ensure_model_exists():
    """确保模型文件存在，如果不存在则自动下载"""
    model_path = Path(model_config['model_path'])

    if model_path.exists():
        print(f'Model file found at {model_path}')
        return True

    # 确保data目录存在
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Model file not found at {model_path}')
    print('Starting automatic download...')

    if download_model(model_config['model_url'], model_path):
        if model_config.get('model_md5'):
            if verify_model_md5(model_path, model_config['model_md5']):
                print('Model MD5 verification passed!')
                return True
            else:
                print('Warning: MD5 verification failed, but proceeding anyway.')
                return True
        return True
    else:
        raise FileNotFoundError(
            f'Failed to download model. Please download manually from:\n'
            f'  {model_config["model_url"]}\n'
            f'  Or Google Drive: https://drive.google.com/drive/folders/1cTsXP5NYTCjhPVxwScdhxqJleHuIOyXG\n'
            f'And place it at: {model_path}'
        )


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_model():
    """初始化文本检测模型"""
    global model
    if model is None:
        # 确保模型文件存在
        ensure_model_exists()

        print(f"Loading model from {model_config['model_path']} on device {model_config['device']}...")
        model = TextDetector(
            model_path=model_config['model_path'],
            input_size=model_config['input_size'],
            device=model_config['device'],
            half=model_config['half'],
            conf_thresh=model_config['conf_thresh'],
            nms_thresh=model_config['nms_thresh'],
            mask_thresh=model_config['mask_thresh']
        )
        print("Model loaded successfully!")
    return model


def encode_image_to_base64(image):
    """将OpenCV图像编码为base64字符串"""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'device': model_config['device'],
        'model_loaded': model is not None
    })


@app.route('/detect', methods=['POST'])
def detect_text():
    """
    检测图片中的文本
    支持两种输入方式：
    1. multipart/form-data 上传图片文件
    2. application/json 发送base64编码的图片

    返回JSON格式的检测结果
    """
    try:
        # 初始化模型
        model = init_model()

        # 解析输入图片
        img = None

        # 方式1: 通过form-data上传文件
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return Response(json.dumps({'error': 'No file selected'}), status=400, mimetype='application/json')

            if not allowed_file(file.filename):
                return Response(json.dumps({'error': 'File type not allowed'}), status=400, mimetype='application/json')

            # 读取图片
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 方式2: 通过JSON发送base64编码图片
        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return Response(json.dumps({'error': 'No image data provided'}), status=400, mimetype='application/json')

            # 解码base64图片
            if data['image'].startswith('data:image'):
                # 移除data URL前缀
                base64_str = data['image'].split(',')[1]
            else:
                base64_str = data['image']

            img_data = base64.b64decode(base64_str)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        else:
            return Response(json.dumps({'error': 'Unsupported request format'}), status=400, mimetype='application/json')

        if img is None:
            return Response(json.dumps({'error': 'Failed to decode image'}), status=400, mimetype='application/json')

        # 运行检测
        mask, mask_refined, blk_list = model(img)

        # 准备结果
        result = {
            'success': True,
            'image_size': {
                'height': int(img.shape[0]),
                'width': int(img.shape[1])
            },
            'text_blocks': [],
            'text_lines_count': 0,
            'mask_size': mask.shape if mask is not None else None
        }

        # 处理文本块
        if blk_list:
            for blk in blk_list:
                blk_dict = blk.to_dict()
                result['text_blocks'].append(blk_dict)

            # 统计文本行数
            for blk in blk_list:
                if blk.lines:
                    result['text_lines_count'] += len(blk.lines)

        # 可选：返回mask的base64编码
        if 'return_mask' in request.form and request.form['return_mask'].lower() == 'true':
            result['mask_base64'] = encode_image_to_base64(mask)
            result['mask_refined_base64'] = encode_image_to_base64(mask_refined)

        # 使用NumpyEncoder处理numpy类型
        return Response(json.dumps(result, ensure_ascii=False, cls=NumpyEncoder, indent=2), mimetype='application/json')

    except Exception as e:
        return Response(json.dumps({
            'error': str(e),
            'type': type(e).__name__
        }, ensure_ascii=False), status=500, mimetype='application/json')


@app.route('/detect_visual', methods=['POST'])
def detect_text_visual():
    """
    检测文本并返回可视化结果图片
    """
    try:
        model = init_model()

        # 获取图片
        if 'image' not in request.files:
            return Response(json.dumps({'error': 'No image file provided'}), status=400, mimetype='application/json')

        file = request.files['image']
        if not allowed_file(file.filename):
            return Response(json.dumps({'error': 'File type not allowed'}), status=400, mimetype='application/json')

        # 读取图片
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return Response(json.dumps({'error': 'Failed to decode image'}), status=400, mimetype='application/json')

        # 运行检测
        mask, mask_refined, blk_list = model(img)

        # 可视化结果
        from utils.textmask import visualize_textblocks
        img_visualized = img.copy()
        visualize_textblocks(img_visualized, blk_list)

        # 保存到临时文件
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
        cv2.imwrite(output_path, img_visualized)

        return send_file(output_path, mimetype='image/jpeg')

    except Exception as e:
        return Response(json.dumps({'error': str(e)}), status=500, mimetype='application/json')


@app.route('/config', methods=['GET', 'POST'])
def config():
    """获取或更新模型配置"""
    global model_config, model

    if request.method == 'GET':
        return Response(json.dumps(model_config), mimetype='application/json')

    elif request.method == 'POST':
        try:
            data = request.get_json()
        except:
            return Response(json.dumps({'error': 'Invalid JSON data'}), status=400, mimetype='application/json')

        # 更新配置
        if 'conf_thresh' in data:
            model_config['conf_thresh'] = float(data['conf_thresh'])
        if 'nms_thresh' in data:
            model_config['nms_thresh'] = float(data['nms_thresh'])
        if 'mask_thresh' in data:
            model_config['mask_thresh'] = float(data['mask_thresh'])
        if 'input_size' in data:
            model_config['input_size'] = int(data['input_size'])

        # 如果配置改变，需要重新加载模型
        if 'model_path' in data:
            model_config['model_path'] = data['model_path']
            model = None  # 触发重新加载

        return Response(json.dumps({
            'success': True,
            'config': model_config
        }, ensure_ascii=False, cls=NumpyEncoder), mimetype='application/json')


if __name__ == '__main__':
    print("=" * 60)
    print("Comic Text Detection API Service")
    print("=" * 60)
    print(f"Device: {model_config['device']}")
    print(f"Model path: {model_config['model_path']}")
    print(f"Listening on http://0.0.0.0:5000")
    print("\nAvailable endpoints:")
    print("  GET  /health           - Health check")
    print("  POST /detect           - Detect text (returns JSON)")
    print("  POST /detect_visual    - Detect text (returns image)")
    print("  GET  /config           - Get model config")
    print("  POST /config           - Update model config")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=False)
