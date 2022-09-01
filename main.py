# 서버용 #
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import uuid as id
import time, random
import natsort, shutil
# 서버용 #

# 모델용 #
import argparse, os, sys, torch
import numpy as np
from PIL import Image
sys.path.append('.')
from EleGANt.training.config import get_config
from EleGANt.training.inference import Inference
from EleGANt.training.utils import create_logger, print_args # 로그 생성용

import cv2 # 다중인식용
import glob
# 모델용 #

# 로그인 #
from flask import session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

from flask_bcrypt import Bcrypt # 암호관련 패키지
# 로그인 #

app = Flask(__name__)
mysql = MySQL(app)
bcrypt = Bcrypt(app)

RESULT_FOLDER = 'static/results/'
UPLOAD_FOLDER = 'static/uploads/'
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 # 업로드 5MB 용량 제한

app.secret_key = "secret key"

app.config['BCRYPT_LEVEL'] = 10
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'PJ_user_system'

@app.route('/')
def home():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']

@app.route('/', methods=['POST'])
def upload_image():
    # if 'file' not in request.files:
    #     flash('파일이 존재하지 않습니다.')
    #     redirect(request.url)
    file = request.files['file']
    # if file.filename == '':
    #     flash('업로드할 이미지를 선택하세요.')
    #     redirect(request.url)
    if file and allowed_file(file.filename):
        filename = time.strftime("[%Y-%m-%d-%H:%M]")+id.uuid4().hex + "_" + secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if request.form.get('gen_num', type=int):
            generating_image = request.form.get('gen_num', type=int)
        else:
            generating_image = 1

        ### model ###
        try:
            ts = time.time()

            parser = argparse.ArgumentParser("argument for training")
            parser.add_argument("--name", type=str, default='')
            parser.add_argument("--save_path", type=str, default='static/results', help="path to save model")
            parser.add_argument("--load_path", type=str, help="folder to load model", default="./EleGANt/ckpts/sow_pyramid_a5_e3d2_remapped.pth")
            parser.add_argument("--source-dir", type=str, default="static/uploads")
            parser.add_argument("--reference-dir", type=str, default="static/references")
            parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

            args = parser.parse_args()
            args.gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
            args.device = torch.device(args.gpu)
            args.save_folder = os.path.join(args.save_path, args.name)

            config = get_config()
            refer_list = main(config, args, filename, generating_image)

            te = time.time()
            print(f' * 사진 생성에 {te - ts:.2f}초 걸렸습니다.\n')

            flash('사진을 클릭하면 다운로드가 진행됩니다.')


            ### 레퍼런스 ###
            if len(refer_list) < 9:
                left_refer_list = refer_list

            left_refer_list = [f"references/{refer_img}" for refer_img in refer_list][:9]
            right_refer_list = [f"references/{refer_img}" for refer_img in refer_list][9:]
            ### 레퍼런스 ###

        except:
            flash('얼굴 사진을 업로드 하세요!')
        ### model ###

        ### 생성사진 ###
        img_list = os.listdir(f'static/results/{os.path.splitext(filename)[0]}')
        img_list = [f'results/{os.path.splitext(filename)[0]}/{img}' for img in img_list]
        img_list = natsort.natsorted(img_list)
        ### 생성사진 ###


        try:
            if session['loggedin']:
                return render_template('user.html', filename=filename, img_list=img_list,
                                       left_refer_list=left_refer_list, right_refer_list=right_refer_list)
        except:
            return render_template('index.html', filename=filename, img_list=img_list)

    else:
        flash('png, jpg, jpeg 유형의 파일만 가능합니다.')

    return redirect(request.url)

# upload_image_display
@app.route('/display/<filename>')
def display_upload(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

class FaceCropper(object):
    CASCADE_PATH = "haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, filename):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
        if (faces is None):
            print('Failed to detect face')
            return 0
        # facecnt = len(faces)
        # print("Detected faces: %d" % facecnt)
        i = 0
        height, width = img.shape[:2]

        tmp_res=[]
        save_path = f"./static/uploads/{os.path.splitext(filename)[0]}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (361, 361))
            i += 1
            tmp_res.append(lastimg)
            cv2.imwrite(f"{save_path}/src_{i}.png", lastimg)
        return save_path


def main(config, args, filename, generating_image):
    ### log ###
    # logger = create_logger(args.save_folder, args.name, 'info', console=True)
    # print_args(args, logger)
    # logger.info(config)
    ### log ###

    global result_path
    result_path = f"{args.save_folder}/{os.path.splitext(filename)[0]}"

    if os.path.exists(result_path):
        shutil.rmtree(result_path)
        os.makedirs(result_path)
    else:
        os.makedirs(result_path)

    inference = Inference(config, args, args.load_path)
    refer_list = random.sample(os.listdir(args.reference_dir), generating_image)

    print(f"\n * {filename}\n * {generating_image}장 생성시작")
    for i, refer_img in enumerate(refer_list, 1):
        src_img = Image.open(os.path.join(args.source_dir, filename)).convert('RGB')
        ref_img = Image.open(os.path.join(args.reference_dir, refer_img)).convert('RGB')


        ### 다중인식 ###
        FC = FaceCropper();
        img_path = f'./static/uploads/{filename}'
        save_path = FC.generate(img_path, filename)
        src_list = os.listdir(save_path)
        src_num = len(src_list)
        merged_img = []
        if src_num>1:
            for j in range(src_num):
                img = cv2.imread(f"{save_path}/src_{j+1}.png")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img).convert('RGB')

                result = inference.transfer(img, ref_img, postprocess=True)
                result = result.resize((361, 361))
                result = np.array(result)

                merged_img.append(result)
            result = np.concatenate(merged_img, axis=1)
        ### 다중인식 ###
        else:
            result = inference.transfer(src_img, ref_img, postprocess=True)
            result = result.resize((361, 361))
            result = np.array(result)

        if result is None:
            continue

        save_path = os.path.join(args.save_folder + f"/{os.path.splitext(filename)[0]}", f"{i}.png")
        Image.fromarray(result.astype(np.uint8)).save(save_path)

        if i%3==0:
            print(f'{i:>5}/{generating_image}장 생성완료')
    print(f" * {filename}\n * {generating_image}장 생성완료")
    return refer_list


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']; password = request.form['password']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s', (email, ))
        user = cursor.fetchone()

        pw_hash = bcrypt.generate_password_hash(password, 10)
        pw_check = bcrypt.check_password_hash(pw_hash, password)

        if user and pw_check:
            session['loggedin'] = True
            session['userid'] = user['userid']
            session['name'] = user['nickname']
            session['email'] = user['email']
            return render_template('user.html')
        else:
            flash("이메일 또는 비밀번호를 확인하세요.")
    return render_template('login.html')

@app.route('/dashboard/', methods=['GET'])
def dashboard():
    return render_template('user.html')

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('loggedin', None)
    session.pop('userid', None)
    session.pop('email', None)
    return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    message = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form:
        nickname = request.form['name']
        email = request.form['email']
        password = request.form['password']
        password_check = request.form['password-check']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s', (email, ))
        account = cursor.fetchone()
        if account:
            message = '이미 존재하는 계정입니다.'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            message = '유효하지 않은 이메일입니다.'
        elif not password == password_check:
            message = '비밀번호를 다시 입력하세요.'
        elif not nickname or not password or not email:
            message = '양식을 모두 작성해 주세요.'
        else:
            password = bcrypt.generate_password_hash(password, 10)
            cursor.execute('INSERT INTO user VALUES (NULL, % s, % s, % s)', (nickname, email, password, ))
            mysql.connection.commit()
            flash('회원가입이 완료되었습니다.')
            return redirect(url_for('login'))
    elif request.method == 'POST':
        message = '양식을 모두 작성해 주세요.'
    return render_template('register.html', message = message)

# generating image download(click)
@app.route('/static/<path:filename>', methods=['GET', 'POST'])
def download_file(filename):
    return send_from_directory(directory=result_path, path=filename, as_attachment=True)


if __name__ == "__main__":
    app.run()















