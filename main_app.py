# coding=utf-8
import json
import os

import rq
from flask import Flask, Response
from flask import request
from flask_cors import CORS

import task_handler
from classifier.model import DeepLearningModel
from db.database import modelDB
from enums import model
from worker import conn

FLASK_APP = Flask(__name__)
CORS(FLASK_APP)  # allowing request from different urls... (localhost in another port)

# just to avoid a windows error... o.ó
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')

tasksQueue = rq.Queue(connection=conn, default_timeout=3600)


@FLASK_APP.route("/task/<task_id>", methods=["GET"])
def get_task_status(task_id):
    task = tasksQueue.fetch_job(task_id)

    if task:
        response_object = {
            "status": "success",
            "data": {
                "task_id": task.get_id(),
                "task_status": task.get_status(),
                "task_result": task.result,
            },
        }
    else:
        response_object = {"status": "error"}

    response = json.dumps(response_object)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/models')
def get_models():
    models = modelDB.get_models()

    models_names = []
    for m in models:
        models_names.append(m[model.ORIGINAL_NAME])

    respjs = json.dumps({"models": models_names})
    return Response(respjs, status=200, mimetype='application/json')


@FLASK_APP.route('/model/<model_name>', methods=['GET', 'POST'])
def get_model_info(model_name):
    if request.method == 'POST':  # create model
        model_info = json.loads(request.form.get('model_info'))
        model_isnew = model_info["model_isnew"]
        model_typedata = model_info["model_type_data"]

        if modelDB.exist_model(model_name):
            respjs = json.dumps({"message": "That model already exists..."})
            return Response(respjs, status=300, mimetype='application/json')

        DeepLearningModel(name=model_name, newModel=model_isnew, dataType=model_typedata)
        return Response(status=200, mimetype='application/json')
    else:  # get model information
        model_id = model_name.lower().replace(" ", "_")
        model_info = modelDB.get_model(model_id)
        respjs = json.dumps({"model_info": model_info})
        return Response(respjs, status=200, mimetype='application/json')


@FLASK_APP.route('/train', methods=['POST'])
def train_model():
    model_info = json.loads(request.form.get('model_info'))
    relevant_docs = request.form.get('relevant_docs').split()
    irrelevant_docs = request.form.get('irrelevant_docs').split()

    job = tasksQueue.enqueue(task_handler.train_model, model_info, relevant_docs, irrelevant_docs)
    jobId_js = json.dumps(job.get_id())
    return Response(jobId_js, status=200, mimetype='application/json')


@FLASK_APP.route('/search/google', methods=['POST'])
def search_news():
    query = request.form.get('query')
    model_info = json.loads(request.form.get('model_info'))

    job = tasksQueue.enqueue(task_handler.search_news, model_info, query)
    jobId_js = json.dumps(job.get_id())
    return Response(jobId_js, status=200, mimetype='application/json')


@FLASK_APP.route('/relevant/google/<model_name>')
def get_relevant_news(model_name):
    result = task_handler.relevant_google(model_name)
    resultjs = json.dumps(result)
    return Response(resultjs, status=200, mimetype='application/json')


@FLASK_APP.route('/classify/google/<model_name>', methods=['POST'])
def classify_news(model_name):
    new_documents = json.loads(request.form.get('documents'))

    job = tasksQueue.enqueue(task_handler.classify_news, model_name, new_documents)
    jobId_js = json.dumps(job.get_id())
    return Response(jobId_js, status=200, mimetype='application/json')


if __name__ == '__main__':
    FLASK_APP.run()
