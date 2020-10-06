import _pickle
import base64
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template, Response, session, make_response, jsonify
from pyts.classification import SAXVSM
from sklearn.model_selection import GridSearchCV
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sax import visualization, distance_measure, euclidean_distance, znorm, draw_distance, sax_via_window, paa_distance
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from flask_sqlalchemy import SQLAlchemy
from flask_cors import *
from flask_redis import FlaskRedis
import msgpack
from pyecharts.globals import CurrentConfig
from pyecharts.options import *
from pyecharts.charts import Line
from pyecharts.charts import Bar
from pyecharts.globals import ThemeType

CurrentConfig.ONLINE_HOST = "http://127.0.0.1:5000/static/assets/"

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = 'sax'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/1'
app.config['REDIS_URL'] = 'redis://localhost:6379/0'
rd = FlaskRedis(app)

db = SQLAlchemy(app)
ALLOWED_EXTENSIONS = {'csv', 'tsv'}
# global chart

executor = ThreadPoolExecutor(4)


class User(db.Model):
    __tablename__ = 'user'
    username = db.Column(db.String(40), primary_key=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    files = db.Column(db.Integer, nullable=False)

    def __init__(self, username, password, files):
        self.username = username
        self.password = password
        self.files = files


class History(db.Model):
    __tablename__ = 'history'
    order = db.Column(db.Integer, autoincrement=True, primary_key=True)
    no = db.Column(db.Integer, nullable=False)
    username = db.Column(db.String(40), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    idx = db.Column(db.String(10), nullable=False)
    pairwise_size = db.Column(db.String(3), nullable=False)
    alphabet_size = db.Column(db.String(3), nullable=False)
    window_size = db.Column(db.String(5), nullable=False)

    # 单个对象方法
    def single_to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    # 多个对象
    def double_to_dict(self):
        result = {}
        for key in self.__mapper__.c.keys():
            if getattr(self, key) is not None:
                result[key] = str(getattr(self, key))
            else:
                result[key] = getattr(self, key)
        return result


class Model_history(db.Model):
    __tablename__ = 'model_history'
    order = db.Column(db.Integer, autoincrement=True, primary_key=True)
    no = db.Column(db.Integer, nullable=False)
    username = db.Column(db.String(40), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    paa_size1 = db.Column(db.String(3), nullable=False)
    paa_size2 = db.Column(db.String(3), nullable=False)
    alphabet_size1 = db.Column(db.String(3), nullable=False)
    alphabet_size2 = db.Column(db.String(3), nullable=False)
    window_size1 = db.Column(db.String(5), nullable=False)
    window_size2 = db.Column(db.String(5), nullable=False)

    # 单个对象方法
    def single_to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    # 多个对象
    def double_to_dict(self):
        result = {}
        for key in self.__mapper__.c.keys():
            if getattr(self, key) is not None:
                result[key] = str(getattr(self, key))
            else:
                result[key] = getattr(self, key)
        return result


# sql query to json
def to_json(all_vendors):
    v = [ven.double_to_dict() for ven in all_vendors]
    return v


@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    # if request.method == 'GET':
    #     if session.get('error') is not None:
    #         session.pop('error')  # delete the username/password validation information
    #     name = request.cookies.get('username')
    #     if name:
    #         session['username'] = name
    #         return redirect(url_for('profile'))
    #     else:
    #         return render_template('new-login.html')
    # else:
    if request.method == 'POST':
        name = request.form.get('user')
        psw = request.form.get('psw')

        if name and psw:
            user = User.query.filter_by(username=name).first()
            print(user)
            # if username does not exist
            if not user:
                session['error'] = 'username'
                return jsonify({"status": 'exist', 'info': 'Username does not exist!'})

            print(psw)
            if not check_password_hash(user.password, psw):
                session['error'] = 'password'
                return jsonify({"status": 'wrong', 'info': 'Password is wrong!'})

            # if session.get('error') is not None:
            #     session.pop('error')  # delete the username/password validation information
            session['username'] = name
            # god = God()
            # rd.set(session['username'], _pickle.dumps(god))
            response = make_response(jsonify({"status": 'success', 'info': 'Success!'}))
            # response.response = render_template('welcome.html')
            response.set_cookie('username', name, max_age=3600 * 24)
            return response
        else:
            # session['error'] = 'null'
            return jsonify({"status": 'null', 'info': 'Input value is null!'})


@app.route('/register', methods=['GET', 'POST'])
def register():
    # if request.method == 'GET':
    #     if session.get('error') is not None:
    #         session.pop('error')  # delete the username/password validation information on login page
    #     return render_template('new-register.html')
    # else:
    if request.method == 'POST':
        name = request.form.get('username')
        psw = request.form.get('password')
        if name and psw:
            if User.query.filter_by(username=name).first():
                # session['error'] = 'username'
                # flash('This ID has been registered!')
                return jsonify({"status": 'change', 'info': 'Username already exists!'})
            else:
                psw = generate_password_hash(psw)
                db.session.add(User(name, psw, 0))
                db.session.commit()
                # session['error'] = 'success'
                # flash('Registry Success! Now get started!')
                # return redirect(url_for('login'))
                return jsonify({"status": 'success', 'info': 'Register success!'})
        else:
            # session['error'] = 'null'
            # flash('Illegal username or password!')
            return jsonify({"status": 'null', 'info': 'Input value is null!'})


@app.route('/logout')
def logout():
    rd.delete(session['username']+'error')
    rd.delete(session['username']+'training_status')
    rd.delete(session['username']+'models')
    rd.delete(session['username']+'datasets')
    session.clear()
    response = Response(render_template('logout.html'))
    response.delete_cookie('username')
    return response


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'POST':
        if request.form.get('record'):
            history = History.query.filter_by(no=request.form.get('record')).first()
            print(history)
            session['filename'] = history.filename
            session['index'] = history.idx
            session['pairwise_size'] = history.pairwise_size
            session['alphabet_size'] = history.alphabet_size
            session['window_size'] = history.window_size
            return redirect(url_for('visualize'))
        else:
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                user = db.session.query(User).filter_by(username=request.cookies.get('username')).first()

                # file numbers of current user
                file_number = user.files + 1
                user.files = file_number
                db.session.commit()

                filename = request.cookies.get('username') + '_' + str(file_number) + '_' + secure_filename(
                    file.filename)
                session['filename'] = filename[len(session['username']) + 1:]
                file.save('dataset/' + filename)

                return redirect(url_for('initialize'))
            return 'Uploading file is wrong'
    else:

        # print(history)
        # print(to_json(history))
        # if history:
        #     return render_template('new-profile.html', history=history)
        return render_template('new-profile.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/initialize', methods=['GET', 'POST'])
def initialize():
    if request.method == 'POST':
        try:
            if request.form.get('index') or request.form.get('window_size'):
                # sax visualize initialize
                if request.form.get('index'):
                    session['index'] = request.form.get('index')
                if request.form.get('pairwise_size'):
                    session['pairwise_size'] = request.form.get('pairwise_size')
                if request.form.get('alphabet_size'):
                    session['alphabet_size'] = request.form.get('alphabet_size')
                if request.form.get('window_size'):
                    session['window_size'] = request.form.get('window_size')

                # to inquire the no of the current user
                history = History.query.filter(History.username == session['username']).order_by(
                    History.no.desc()).first()
                try:
                    no = history.no + 1
                except AttributeError:
                    no = 1

                db.session.add(History(username=session['username'],
                                       filename=session['filename'],
                                       idx=session['index'],
                                       pairwise_size=session['pairwise_size'],
                                       alphabet_size=session['alphabet_size'],
                                       window_size=session['window_size'],
                                       no=no))
                db.session.commit()

                return redirect(url_for('visualize'))
            else:

                return redirect(url_for('distance'))
        except KeyError:  # user may not input the right parameter
            return redirect(request.url)
    else:
        try:
            '''
            here we want to know the size of the dataset
            in order to inform the ranges of the parameters inside which users should input the values
            '''
            filename = session["username"] + '_' + session["filename"]
            # firstly load the data
            if session["filename"][-3] == "c":
                ts = pd.read_csv("dataset/" + filename, header=None, index_col=False).iloc[:, 1:]
            elif session["filename"][-3] == "t":
                ts = pd.read_csv("dataset/" + filename, sep="\t", header=None, index_col=False).iloc[:, 1:]
            else:
                raise FileNotFoundError
            # then get the size of dataset, and put them into the session
            session['row'] = ts.shape[0] - 1
            session['col'] = ts.shape[1] - 1

            return render_template('new-initialize.html')
        except Exception as e:  # when user get into initialization without uploading a file
            print(e)
            session['error'] = 'filename'
            return redirect(url_for('profile'))


@app.route("/visualize", methods=['GET', 'POST'])
def visualize():
    # global chart

    try:
        if request.method == 'POST':
            if request.form.get('index') or request.form.get('window_size'):
                # sax visualize initialize
                if request.form.get('index'):
                    session['index'] = request.form.get('index')
                if request.form.get('pairwise_size'):
                    session['pairwise_size'] = request.form.get('pairwise_size')
                if request.form.get('alphabet_size'):
                    session['alphabet_size'] = request.form.get('alphabet_size')
                if request.form.get('window_size'):
                    session['window_size'] = request.form.get('window_size')

                # to inquire the no of the current user
                history = History.query.filter(History.username == session['username']).order_by(
                    History.no.desc()).first()
                try:
                    no = history.no + 1
                except AttributeError:
                    no = 1

                db.session.add(History(username=session['username'],
                                       filename=session['filename'],
                                       idx=session['index'],
                                       pairwise_size=session['pairwise_size'],
                                       alphabet_size=session['alphabet_size'],
                                       window_size=session['window_size'],
                                       no=no))
                db.session.commit()
            else:
                # distance measurement initialize
                if request.form.get('length'):
                    session['length'] = request.form.get('length')
                if request.form.get('a_id'):
                    session['a_id'] = request.form.get('a_id')
                if request.form.get('start'):
                    session['start'] = request.form.get('start')
                if request.form.get('b_id'):
                    session['b_id'] = request.form.get('b_id')
                if request.form.get('pairwise_size'):
                    session['pairwise_size'] = request.form.get('pairwise_size')
                if request.form.get('alphabet_size'):
                    session['alphabet_size'] = request.form.get('alphabet_size')

                return redirect(url_for('distance'))

    except NameError:  # input parameters when no file is upload
        return redirect(url_for('profile'))
    except KeyError:  # user may not input the right parameter
        return redirect(request.url)

    # chart = visualization(filename, index, pairwise_size, alphabet_size, window_size)

    return render_template('new-visualize.html', random=str(random.randint(0, 20)))


@app.route("/distance", methods=['GET', 'POST'])
def distance():
    if request.method == 'GET':
        eu, paa, dtw_fig, string = set_distance()
        return render_template('new-distance.html', eu=eu, paa=paa, dtw_fig=dtw_fig, string=string,
                               random=str(random.randint(0, 20)))
    else:
        try:
            if request.form.get('index') or request.form.get('window_size'):
                # sax visualize initialize
                if request.form.get('index'):
                    session['index'] = request.form.get('index')
                if request.form.get('pairwise_size'):
                    session['pairwise_size'] = request.form.get('pairwise_size')
                if request.form.get('alphabet_size'):
                    session['alphabet_size'] = request.form.get('alphabet_size')
                if request.form.get('window_size'):
                    session['window_size'] = request.form.get('window_size')

                # to inquire the no of the current user
                history = History.query.filter(History.username == session['username']).order_by(
                    History.no.desc()).first()

                try:
                    no = history.no + 1
                except AttributeError:
                    no = 1

                db.session.add(History(username=session['username'],
                                       filename=session['filename'],
                                       idx=session['index'],
                                       pairwise_size=session['pairwise_size'],
                                       alphabet_size=session['alphabet_size'],
                                       window_size=session['window_size'],
                                       no=no))
                db.session.commit()

                return redirect(url_for('visualize'))
            else:
                eu, paa, dtw_fig, string = set_distance()

                return render_template('new-distance.html', eu=eu, paa=paa, dtw_fig=dtw_fig, string=string, random=str(random.randint(0, 20)))
        except KeyError:  # user may not input the right parameter
            return redirect(request.url)


def set_distance():
    # distance measurement initialize
    if request.form.get('length'):
        session['length'] = request.form.get('length')
    if request.form.get('a_id'):
        session['a_id'] = request.form.get('a_id')
    if request.form.get('start'):
        session['start'] = request.form.get('start')
    if request.form.get('b_id'):
        session['b_id'] = request.form.get('b_id')
    if request.form.get('pairwise_size'):
        session['pairwise_size'] = request.form.get('pairwise_size')
    if request.form.get('alphabet_size'):
        session['alphabet_size'] = request.form.get('alphabet_size')

    filename = session["username"] + '_' + session["filename"]
    if session["filename"][-3] == "c":
        ts = pd.read_csv("dataset/" + filename, header=None, index_col=False).iloc[:, 1:]
    elif session["filename"][-3] == "t":
        ts = pd.read_csv("dataset/" + filename, sep="\t", header=None, index_col=False).iloc[:, 1:]
    else:
        raise FileNotFoundError

    ts1 = np.array(ts)[int(session['a_id'])]
    ts2 = np.array(ts)[int(session['b_id'])]

    ts1_1d = ts1[int(session['start']):int(session['start']) + int(session['length'])]
    ts2_1d = ts2[int(session['start']):int(session['start']) + int(session['length'])]

    # ts1_2d = np.array(pd.concat([pd.DataFrame(ts.index), ts1],
    #                             axis=1))[int(session['a_start']):int(session['a_start']) + int(session['length']), :].tolist()
    # ts2_2d = np.array(pd.concat([pd.DataFrame(ts.index), ts2],
    #                             axis=1))[int(session['b_start']):int(session['b_start']) + int(session['length']), :].tolist()

    # DTW based on shape
    s1 = znorm(np.array(ts1_1d, dtype=np.double))
    s2 = znorm(np.array(ts2_1d, dtype=np.double))
    path = dtw.warping_path(s1, s2)
    temp_filename = 'static/images/' + session['username'] + "dtw.png"
    dtwvis.plot_warping(s1, s2, path, filename=temp_filename)  # draw DTW distance
    with open(temp_filename, "rb") as f:
        base64_data = base64.b64encode(f.read())
    # 文字转图片
    dtw_fig = 'data:image/png;base64,' + base64_data.decode()


    # mindist: {sax_string a, sax_string b, distance}
    print('-----user set paa & sax size for distance measuring-----')
    print(session['pairwise_size'])
    print(session['alphabet_size'])

    # calculate sax distance
    mindist = distance_measure(ts1_1d, ts2_1d, int(session['pairwise_size']), int(session['alphabet_size']))
    eu, paa, string = draw_distance(s1, s2, int(session['pairwise_size']), int(session['alphabet_size']))  # draw SAX distance diagram

    # store 3 distance results into session
    session['string_a'] = mindist[0]
    session['string_b'] = mindist[1]
    session['mindist'] = round(mindist[2], 2)
    session['eu'] = round(euclidean_distance(ts1_1d, ts2_1d), 2)
    session['dtw'] = round(dtw.distance(s1, s2), 2)
    session['paa'] = round(paa_distance(s1,s2,int(session['pairwise_size'])),2)

    return eu, paa, dtw_fig, string


@app.route('/train', methods=['GET', 'POST'])
def train():
    # if request is GET
    if request.method == "GET":
        return render_template('train.html')
    # if request is POST
    else:
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"code": 500, "status": 'No file is uploaded!'})

        file = request.files['file']
        # if user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({"code": 500, "status": 'Filename is null!'})
        # if get the correct fil e
        if file and allowed_file(file.filename):
            print('-----user upload training data-----')
            print(file.filename)

            if file.filename[-3] == "c":
                ts = np.array(pd.read_csv(file, header=None, index_col=False))
            elif file.filename[-3] == "t":
                ts = np.array(pd.read_csv(file, sep="\t", header=None, index_col=False))
            else:
                raise FileNotFoundError

            # get the number of columns in training set, in order to verify if the predict set is valid
            session['training_set_length'] = ts.shape[1]

            x_train = ts[:, 1:]
            y_train = ts[:, 0]

            is_null = True
            if request.form.get('word1'):
                session['word1'] = request.form.get('word1')
                is_null = False
            if request.form.get('word2'):
                session['word2'] = request.form.get('word2')
                is_null = False
            if request.form.get('bin1'):
                session['bin1'] = request.form.get('bin1')
                is_null = False
            if request.form.get('bin2'):
                session['bin2'] = request.form.get('bin2')
                is_null = False
            if request.form.get('win1'):
                session['win1'] = request.form.get('win1')
                is_null = False
            if request.form.get('win2'):
                session['win2'] = request.form.get('win2')
                is_null = False

            if not is_null:
                word = list(range(int(session['word1']), int(session['word2']) + 1))
                nbin = list(range(int(session['bin1']), int(session['bin2']) + 1))
                win = list(range(int(session['win1']), int(session['win2']) + 1))

                para = {'window_size': win, 'word_size': word, 'n_bins': nbin}

                print('-----user----train-----parameter------')
                print(para)

                username = session['username']

                rd.set(username+'training_status', 'training')
                executor.submit(training, x_train, y_train, para, username)

                # to inquire the no of the current user
                model_history = Model_history.query.filter(Model_history.username == session['username']).order_by(
                    Model_history.no.desc()).first()
                print(type(model_history))

                try:
                    no = model_history.no + 1
                except AttributeError:
                    no = 1

                filename = session['username'] + '_' + str(no) + '_' + secure_filename(file.filename)
                session['filename_train'] = str(no) + '_' + file.filename
                file.seek(0)  # put the cursor of file object in the begining
                file.save('dataset/training/' + filename)

                db.session.add(Model_history(username=session['username'],
                                             filename=session['filename_train'],
                                             paa_size1=session['word1'],
                                             alphabet_size1=session['bin1'],
                                             window_size1=session['win1'],
                                             paa_size2=session['word2'],
                                             alphabet_size2=session['bin2'],
                                             window_size2=session['win2'],
                                             no=no))
                db.session.commit()

                # task = modeling_task.delay()
                print('-----server start training-----')

                # return jsonify({"code": 200, "status": 'model training', "task_id": task.id})
                return jsonify({"code": 200, "status": 'model training'})

            # if one of the parameter is null, return mistake
            else:
                return jsonify({"code": 500, "status": 'Input value is wrong!'})

        return 'Uploading file is wrong'


def training(x_train, y_train, para, username):
    try:
        saxvsm = SAXVSM()
        clf = GridSearchCV(saxvsm, para)
        clf.fit(x_train, y_train)
        #
        # session['estimator'] = clf.best_estimator_
        # session['para'] = clf.best_params_
        # session['score'] = clf.best_score_

        # session['is_train'] = 'success'
        print(clf.best_estimator_)
        print(clf.best_params_)
        print(clf.best_score_)
        # models[username] = [clf.best_estimator_, clf.best_params_, clf.best_score_]
        # training_status[username] = 'success'

        rd.set(username + 'training_status', 'success')
        rd.set(username + 'models', _pickle.dumps([clf.best_estimator_, clf.best_params_, clf.best_score_, clf.cv_results_]))

    except Exception as e:
        print(e)
        # session['is_train'] = 'failure'
        # training_status[username] = 'failure'

        rd.set(username + 'training_status', 'failure')
        rd.set(username+'error', str(e))
        # error[username] = str(e)


@app.route("/chart_train")
def get_chart_train():
    # get model info
    username = session['username']
    models = _pickle.loads(rd.get(username + 'models'))

    res = models[3]

    times = res['mean_fit_time']
    times = ['{:.2f}'.format(i) for i in times]
    scores = res['mean_test_score']
    scores = ['{:.1f}'.format(i) for i in scores]
    x = np.arange(0, len(times), 1).tolist()

    '''echart of training model'''
    line = (
        Line(init_opts=InitOpts(theme=ThemeType.DARK))
            .add_xaxis(x)
            .add_yaxis(
            "Mean time",
            times,
            is_symbol_show=True,
            linestyle_opts=LineStyleOpts(
                width=2,
                opacity=0.9
            ),
            markpoint_opts=MarkPointOpts(
                data=[
                    MarkPointItem(type_="max", name="Max",
                                       itemstyle_opts=ItemStyleOpts(color='#E6E6FA', opacity=0.9), ),
                    MarkPointItem(type_="min", name="Min",
                                       itemstyle_opts=ItemStyleOpts(color='#E6E6FA', opacity=0.9), ),
                ]
            ),
            markline_opts=MarkLineOpts(
                data=[MarkLineItem(type_="average", name="Average")],
                symbol_size=2,
            ),
        )
            .set_global_opts(
            title_opts=TitleOpts(title="Training and tuning"),
            tooltip_opts=TooltipOpts(trigger='axis', axis_pointer_type="cross"),
            datazoom_opts=[
                DataZoomOpts(xaxis_index=0, range_start=0, range_end=100),
                DataZoomOpts(yaxis_index=0, type_='inside', range_start=0, range_end=100)
                # DataZoomOpts(type_="slider", orient='vertical',is_realtime=False,range_start=0,range_end=100),
                # DataZoomOpts(type_="inside", yaxis_index=0,is_realtime=False)
            ],
            toolbox_opts=ToolboxOpts(pos_left='right',
                                          feature=ToolBoxFeatureOpts(
                                              magic_type=ToolBoxFeatureMagicTypeOpts(is_show=False))),
            brush_opts=BrushOpts(),
            yaxis_opts=AxisOpts(split_number=3)
        )
    )
    bar = (
        Bar()
            .add_xaxis(x)
            .add_yaxis("Mean score", scores, category_gap='70%',
                       itemstyle_opts=ItemStyleOpts(color='#BDB76B', opacity=0.4))

    )
    line.overlap(bar)

    return jsonify(line.dump_options_with_quotes())


@app.route('/model', methods=['GET', 'POST'])
def model():
    if request.method == "GET":
        # get model info
        username = session['username']
        models = _pickle.loads(rd.get(username+'models'))

        estimator = models[0]
        para = models[1]
        score = models[2]


        # store the optimal parameter in session
        session['pairwise_size'] = para['word_size']
        session['alphabet_size'] = para['n_bins']
        session['window_size'] = para['window_size']

        return render_template('model.html',
                               estimator=estimator,
                               score=str(round(score, 3) * 100) + "%",
                               word=para['word_size'],
                               nbin=para['n_bins'],
                               win=para['window_size'])

    elif request.method == "POST":
        # test
        if request.files.get('test'):
            file = request.files['test']
            filename = request.cookies.get('username') + '_Test_' + secure_filename(file.filename)

            print(file.filename)
            print('--------model-------filename')
            print(filename)
            print('--------model-------file')
            print(file)

            # read test dataset
            if file.filename[-3] == "c":
                ts = pd.read_csv(file, sep=",", header=None, index_col=False)
            elif file.filename[-3] == "t":
                ts = pd.read_csv(file, sep="\t", header=None, index_col=False)
            else:
                raise FileNotFoundError

            # if the testing set length is not valid
            if ts.shape[1] != int(session['training_set_length']):
                session['error'] = 'invalid dataset'
                return redirect(request.url)

            x_train = np.array(ts)[:, 1:]
            y_train = np.array(ts)[:, 0]

            # get model, parameters, score transfer to render_template
            username = session['username']

            rd.set(username + 'datasets', _pickle.dumps(ts))
            models = _pickle.loads(rd.get(username+'models'))
            estimator = models[0]
            para = models[1]
            score = models[2]

            # store the uploaded dataset as DataFrame
            # datasets[user] = ts

            return render_template('model_test.html',
                                   estimator=estimator,
                                   score=str(round(score, 3) * 100) + "%",
                                   word=para['word_size'],
                                   nbin=para['n_bins'],
                                   win=para['window_size'],
                                   accuracy=str(round(estimator.score(x_train, y_train), 4) * 100) + "%")

        # predict
        elif request.files.get('predict'):
            file = request.files['predict']

            # read predict dataset
            if file.filename[-3] == "c":
                ts = pd.read_csv(file, header=None, index_col=False)
            elif file.filename[-3] == "t":
                ts = pd.read_csv(file, sep="\t", header=None, index_col=False)
            else:
                raise FileNotFoundError

            # if the predict set length is not valid
            if ts.shape[1] != int(session['training_set_length']) - 1:
                session['error'] = 'invalid dataset'
                return redirect(request.url)

            # dataset to be predicted
            x = np.array(ts)

            # get model, parameters, score transfer to render_template
            username = session['username']
            models = _pickle.loads(rd.get(username + 'models'))
            estimator = models[0]
            para = models[1]
            score = models[2]

            # predicted label and dataset in pandas DataFrame
            y = pd.DataFrame(estimator.predict(x))
            data = pd.concat([y, ts], axis=1)

            # store the uploaded dataset as DataFrame
            # datasets[user] = data
            rd.set(username+'datasets', _pickle.dumps(data))

            return render_template('model_predict.html',
                                   estimator=estimator,
                                   score=str(round(score, 3) * 100) + "%",
                                   word=para['word_size'],
                                   nbin=para['n_bins'],
                                   win=para['window_size'],
                                   )

        else:
            raise FileNotFoundError


@app.route('/get_model', methods=["GET"])
def get_model():
    # task = AsyncResult(task_id)
    print("accessing model")
    username = session['username']

    status = rd.get(username+'training_status').decode()

    if status == 'training':
        return jsonify({"code": 204, "status": 'model is still training'})
    elif status == 'success':
        return jsonify({"code": 200, "status": 'model is trained'})
    elif status == 'failure':
        return jsonify({"code": 500, "status": rd.get(username+'error')})
    else:
        print(status)
        return jsonify({"code": 500, "status": 'model training fails'})

    # if task.state == 'PENDING':
    #     return jsonify({"code": 204, "status": 'model training'})
    # elif task.state != 'FAILURE':
    #     return jsonify({"code": 200, "status": 'success'})


@app.route('/history')
def get_history():
    history = History.query.filter_by(username=request.cookies.get('username')).all()
    return jsonify(to_json(history))


@app.route('/choose')
def choose_history():
    data = request.args.to_dict()
    session['filename'] = data['filename']
    session['index'] = data['idx']
    session['pairwise_size'] = data['pairwise_size']
    session['alphabet_size'] = data['alphabet_size']
    session['window_size'] = data['window_size']

    '''
    here we want to know the size of the dataset
    in order to inform the ranges of the parameters inside which users should input the values
    '''
    filename = session["username"] + '_' + session["filename"]
    # firstly load the data
    if session["filename"][-3] == "c":
        ts = pd.read_csv("dataset/" + filename, header=None, index_col=False).iloc[:, 1:]
    elif session["filename"][-3] == "t":
        ts = pd.read_csv("dataset/" + filename, sep="\t", header=None, index_col=False).iloc[:, 1:]
    else:
        raise FileNotFoundError
    # then get the size of dataset, and put them into the session
    session['row'] = ts.shape[0] - 1
    session['col'] = ts.shape[1] - 1

    return jsonify({"code": 200, "status": 'success'})


@app.route('/delete', methods=["DELETE"])
def delete_history():
    data = request.get_data().decode()[3:]
    record = db.session.query(History).filter_by(username=session['username'], no=data).first()
    db.session.delete(record)
    db.session.commit()
    print(request.get_data().decode())
    print(record)
    return jsonify({"code": 200, "status": 'success'})


@app.route('/model_history')
def get_model_history():
    history = Model_history.query.filter_by(username=request.cookies.get('username')).all()
    return jsonify(to_json(history))


@app.route('/choose_model')
def choose_model_history():
    data = request.args.to_dict()
    username = session['username']
    filename = username + '_' + data['filename']  # recover the filename
    session['no'] = data['no']
    session['word1'] = data['paa_size1']
    session['bin1'] = data['alphabet_size1']
    session['win1'] = data['window_size1']
    session['word2'] = data['paa_size2']
    session['bin2'] = data['alphabet_size2']
    session['win2'] = data['window_size2']

    word = list(range(int(session['word1']), int(session['word2']) + 1))
    nbin = list(range(int(session['bin1']), int(session['bin2']) + 1))
    win = list(range(int(session['win1']), int(session['win2']) + 1))

    para = {'window_size': win, 'word_size': word, 'n_bins': nbin}

    if filename[-3] == "c":
        ts = np.array(pd.read_csv('dataset/training/' + filename, header=None, index_col=False))
    elif filename[-3] == "t":
        ts = np.array(pd.read_csv('dataset/training/' + filename, sep="\t", header=None, index_col=False))
    else:
        raise FileNotFoundError

    # get the number of columns in training set, in order to verify if the predict set is valid
    session['training_set_length'] = ts.shape[1]

    x_train = ts[:, 1:]
    y_train = ts[:, 0]

    rd.set(username + 'training_status', 'training')

    executor.submit(training, x_train, y_train, para, username)

    return jsonify({"code": 200, "status": 'success'})


@app.route('/delete_model', methods=["DELETE"])
def delete_model_history():
    data = request.get_data().decode()[3:]
    print(data)
    record = db.session.query(Model_history).filter_by(username=session['username'], no=int(data)).first()
    db.session.delete(record)
    db.session.commit()
    print('-----user delete------')
    print(request.get_data().decode())
    print(record)
    return jsonify({"code": 200, "status": 'success'})


@app.route('/prediction')
def get_prediction():  # the same as get_dataset() except that the first column name is "Prediction"
    ts = _pickle.loads(rd.get(session['username']+'datasets'))

    if ts.shape[1] > 41:
        # make header of ts include one white space to be str '0 ','1 ','2 ', ...
        header = np.array(range(ts.shape[1]), dtype='<U10')  # '<U4' means string with at most 4 characters
        for i in range(len(header)):
            header[i] = header[i] + " "
        header[0] = 'Prediction'  # rename the first column as "class"
        ts.columns = header

        # the first column "ID"
        idColumn = pd.DataFrame(ts.index)
        idColumn.columns = ['Instance ID']

        # dataset column 0~20
        left = ts.iloc[:, :21]
        left = pd.concat([idColumn, left], axis=1)  # concatenate "ID" + dataset column 0~20

        # column "..."
        strg = "21 —— " + str(ts.shape[1] - 21)
        left[strg] = '......'

        # dataset column -20 ~ -1
        right = ts.iloc[:, -20:]

        # concatenate "ID" + dataset column 0~20 + "..." + dataset column -20 ~ -1
        data = pd.concat([left, right], axis=1).to_json(orient='records')

        return jsonify({"code": 200, "status": 'success', "data": data})

    # when the dataset has less than 41 columns
    else:
        # make header of ts include one white space to be str '0 ','1 ','2 ', ...
        header = np.array(range(ts.shape[1]), dtype='<U7')  # '<U4' means string with at most 4 characters
        for i in range(len(header)):
            header[i] = header[i] + " "
        header[0] = 'Prediction'  # rename the first column as "class"
        ts.columns = header

        # the first column "ID"
        idColumn = pd.DataFrame(ts.index)
        idColumn.columns = ['Instance ID']

        # concatenate "ID" + dataset
        data = pd.concat([idColumn, ts], axis=1).to_json(orient='records')
        return jsonify({"code": 200, "status": 'success', "data": data})


@app.route('/test')
def get_test():  # the same as get_dataset()
    # ts = datasets[session['username']]
    ts = _pickle.loads(rd.get(session['username']+'datasets'))

    if ts.shape[1] > 41:
        # make header of ts include one white space to be str '0 ','1 ','2 ', ...
        header = np.array(range(ts.shape[1]), dtype='<U7')  # '<U4' means string with at most 4 characters
        for i in range(len(header)):
            header[i] = header[i] + " "
        header[0] = ' Class '  # rename the first column as "class"
        ts.columns = header

        # the first column "ID"
        idColumn = pd.DataFrame(ts.index)
        idColumn.columns = ['Instance ID']

        # dataset column 0~20
        left = ts.iloc[:, :21]
        left = pd.concat([idColumn, left], axis=1)  # concatenate "ID" + dataset column 0~20

        # column "..."
        strg = "21 —— " + str(ts.shape[1] - 21)
        left[strg] = '......'

        # dataset column -20 ~ -1
        right = ts.iloc[:, -20:]

        # concatenate "ID" + dataset column 0~20 + "..." + dataset column -20 ~ -1
        data = pd.concat([left, right], axis=1).to_json(orient='records')

        return jsonify({"code": 200, "status": 'success', "data": data})

    # when the dataset has less than 41 columns
    else:
        # make header of ts include one white space to be str '0 ','1 ','2 ', ...
        header = np.array(range(ts.shape[1]), dtype='<U7')  # '<U4' means string with at most 4 characters
        for i in range(len(header)):
            header[i] = header[i] + " "
        header[0] = ' Class '  # rename the first column as "class"
        ts.columns = header

        # the first column "ID"
        idColumn = pd.DataFrame(ts.index)
        idColumn.columns = ['Instance ID']

        # concatenate "ID" + dataset
        data = pd.concat([idColumn, ts], axis=1).to_json(orient='records')
        return jsonify({"code": 200, "status": 'success', "data": data})


@app.route('/dataset')
def get_dataset():
    print('------get dataset-----filename')
    print(session["filename"])
    filename = session["username"] + '_' + session["filename"]
    if session["filename"][-3] == "c":
        ts = pd.read_csv("dataset/" + filename, header=None, index_col=False)
    elif session["filename"][-3] == "t":
        ts = pd.read_csv("dataset/" + filename, sep="\t", header=None, index_col=False)
    else:
        raise FileNotFoundError

    if ts.shape[1] > 41:
        # make header of ts include one white space to be str '0 ','1 ','2 ', ...
        header = np.array(range(ts.shape[1]), dtype='<U7')  # '<U4' means string with at most 4 characters
        for i in range(len(header)):
            header[i] = header[i] + " "
        header[0] = ' Class '  # rename the first column as "class"
        ts.columns = header

        # the first column "ID"
        idColumn = pd.DataFrame(ts.index)
        idColumn.columns = ['Instance ID']

        # dataset column 0~20
        left = ts.iloc[:, :21]
        left = pd.concat([idColumn, left], axis=1)  # concatenate "ID" + dataset column 0~20

        # column "..."
        strg = "21 —— " + str(ts.shape[1] - 21)
        left[strg] = '......'

        # dataset column -20 ~ -1
        right = ts.iloc[:, -20:]

        # concatenate "ID" + dataset column 0~20 + "..." + dataset column -20 ~ -1
        data = pd.concat([left, right], axis=1).to_json(orient='records')

        return jsonify({"code": 200, "status": 'success', "data": data})

    # when the dataset has less than 41 columns
    else:
        # make header of ts include one white space to be str '0 ','1 ','2 ', ...
        header = np.array(range(ts.shape[1]), dtype='<U7')  # '<U4' means string with at most 4 characters
        for i in range(len(header)):
            header[i] = header[i] + " "
        header[0] = ' Class '  # rename the first column as "class"
        ts.columns = header

        # the first column "ID"
        idColumn = pd.DataFrame(ts.index)
        idColumn.columns = ['Instance ID']

        # concatenate "ID" + dataset
        data = pd.concat([idColumn, ts], axis=1).to_json(orient='records')
        return jsonify({"code": 200, "status": 'success', "data": data})


# @app.route("/chart_mtpl")
# def get_chart_mtpl():
#     god = _pickle.loads(rd.get(session['username']))
#
#     return god.chart[3]


@app.route("/chart_sax")
def get_chart_sax():
    # c = bar_base()ss
    # god = _pickle.loads(rd.get(session['username']))
    filename = session["username"] + '_' + session["filename"]
    index = int(session['index'])
    pairwise_size = int(session['pairwise_size'])
    alphabet_size = int(session['alphabet_size'])
    window_size = int(session['window_size'])

    if filename[-3] == "c":
        ts = pd.read_csv("dataset/" + filename, header=None, index_col=False).iloc[:, 1:]
    elif filename[-3] == "t":
        ts = pd.read_csv("dataset/" + filename, sep="\t", header=None, index_col=False).iloc[:, 1:]
    else:
        raise FileNotFoundError

    t = np.array(ts)[index]
    dat_znorm = znorm(t)

    vocab = sax_via_window(dat_znorm, paa_size=pairwise_size, alphabet_size=alphabet_size, win_size=window_size)[1]
    vocab = pd.DataFrame.from_dict(vocab, orient='index').reset_index()
    vocab.columns = ['n', 'i']

    chart = visualization(t, dat_znorm, pairwise_size, alphabet_size, window_size)

    return jsonify({"0": chart[0].dump_options_with_quotes(),
                    "1": chart[1].dump_options_with_quotes(),
                    "2": chart[2].dump_options_with_quotes(),
                    "3": chart[3],
                    "4": chart[4],
                    "5": chart[5].dump_options_with_quotes(),
                    "6": vocab.to_json(orient='records')})


# @app.route("/chart_original")
# def get_chart_original():
#     # c = bar_base()
#     god = _pickle.loads(rd.get(session['username']))
#     return god.chart[1].dump_options_with_quotes()
#
#
# @app.route("/chart_sliding")
# def get_chart_sliding():
#     # c = bar_base()
#     god = _pickle.loads(rd.get(session['username']))
#     return god.chart[2].dump_options_with_quotes()


#
@app.errorhandler(404)
def page_not_found(e):
    flash(str(e))
    return render_template('404.html')


@app.errorhandler(KeyError)
def value_error(e):
    flash(str(e))
    return render_template('value_error.html')


@app.errorhandler(Exception)
def value_error(e):
    flash(str(e))
    return render_template('error.html')


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')
    app.run(debug=False  )
    # app.run(debug=False)
