import json

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from clip_server import get_human_score_for_fooling_ai

app = Flask(__name__)
CORS(app)

gvlab_practice_and_qualification = pd.read_csv('assets/gvlab_practice_and_qualification.csv')
gvlab_game_split_5_6 = pd.read_csv('assets/gvlab_game_split_5_6.csv')
gvlab_game_split_10_12 = pd.read_csv('assets/gvlab_game_split_10_12.csv')
gvlab_game_split_5_6['ID'] = gvlab_game_split_5_6['ID'].apply(lambda x: f'id_5_6_{x}')
gvlab_game_split_10_12['ID'] = gvlab_game_split_10_12['ID'].apply(lambda x: f'id_10_12_{x}')
created_data_nitzan = pd.read_csv('assets/create_hit_type_id_33KTOXRB2MHEDTK23QD1IIHZSACFRH_random_indices_100_250_candidates_10_12.csv')

def get_score_by_cue_association(r):
    if r['cue'] == r['cue1']:
        return r['score_fooling_ai_1']
    elif r['cue'] == r['cue2']:
        return r['score_fooling_ai_2']
    else:
        Exception(f'Unknown Score')


for df in [gvlab_practice_and_qualification, gvlab_game_split_5_6, gvlab_game_split_10_12, created_data_nitzan]:
    for col in ['associations', 'distractors', 'candidates', 'labels']:
        if col in df:
            df[col] = df[col].apply(json.loads)

all_game_data = pd.concat([gvlab_game_split_5_6, gvlab_game_split_10_12])
all_game_data_5_candidates = all_game_data[all_game_data['num_candidates'] == 5]
all_game_data_6_candidates = all_game_data[all_game_data['num_candidates'] == 6]
all_game_data_10_candidates = all_game_data[all_game_data['num_candidates'] == 10]
all_game_data_12_candidates = all_game_data[all_game_data['num_candidates'] == 12]
num_candidates_to_df_map = {'5': all_game_data_5_candidates, '6': all_game_data_6_candidates,
                            '10': all_game_data_10_candidates, '12': all_game_data_12_candidates}

@app.route('/task/mturk/create/<id>', methods=['GET'])
def get_task_mturk_create(id):
    """ Returns the relevant instance of the dataset corresponding to the received id """
    print(f"get_task_mturk_create (id={id})")
    # id_number = id.split("_")[-1]
    id_rows = all_game_data[all_game_data['ID'] == id]
    id_row_dict = take_columns_to_dict(id, id_rows, ['candidates'])
    return corsify(id_row_dict)

@app.route('/task/example/create/<id>', methods=['GET'])
def get_task_example_create(id):
    """ Returns the relevant instance of the dataset corresponding to the received id """
    print(f"get_task_example_create (id={id})")
    id_rows = gvlab_practice_and_qualification.query(f'ID=={id}')
    id_row_dict = take_columns_to_dict(id, id_rows, ['candidates'])
    return corsify(id_row_dict)

@app.route('/task/mturk/solve/<id>', methods=['GET'])
def get_task_mturk_solve(id):
    """ Returns the relevant instance of the dataset corresponding to the received id """
    print(f"get_task_mturk_solve (id={id})")
    if 'solve_create_' in id:
        print(f"Taking row from solve_create")
        id_number = id.split("_")[-1]
        id_rows = created_data_nitzan.query(f'annotation_index=={id_number}')
    elif 'solve_game_test_5_6' in id:
        print(f"Taking row from gvlab_game_split")
        id_number = id.split("solve_game_test_5_6")[-1]
        id_rows = gvlab_game_split_5_6.query(f'ID=={id_number}')
        # id_rows = gvlab_game_split_10_12.query(f'ID=={id_number}')
    elif 'solve_game_test_' in id:
        print(f"Taking row from gvlab_game_split")
        id_number = id.split("solve_game_test_")[-1]
        # id_rows = gvlab_game_split.query(f'ID=={id_number}')
        id_rows = gvlab_game_split_10_12.query(f'ID=={id_number}')
    else:
        raise Exception(f"Unknown request")
    id_row_dict = take_columns_to_dict(id, id_rows, ['cue', 'num_associations', 'associations', 'candidates'])
    return corsify(id_row_dict)


@app.route('/task/mturk/solve_create/<annotation_index>', methods=['GET'])
def get_task_mturk_solve_create(annotation_index):
    """ Returns the relevant instance of the dataset corresponding to the received id """
    print(f"get_task_mturk_solve_create (annotation_index={annotation_index})")
    if 'solve_create_nitzan_' in annotation_index:
        print(f"Taking row from solve_create")
        id_number = annotation_index.split("_")[-1]
        id_rows = created_data_nitzan.query(f'annotation_index=={id_number}')
    else:
        id_rows = all_game_data[all_game_data['ID'] == annotation_index]
    id_row_dict = take_columns_to_dict(annotation_index, id_rows, ['cue', 'num_associations', 'associations', 'candidates'])
    print(id_row_dict)
    return corsify(id_row_dict)


@app.route('/task/example/solve/<id>', methods=['GET'])
def get_task_example_solve(id):
    """ Returns the relevant instance of the dataset corresponding to the received id """
    print(f"get_task_example_solve (id={id})")
    id_rows = gvlab_practice_and_qualification.query(f'ID=={id}')
    id_row_dict = take_columns_to_dict(id, id_rows, ['cue', 'num_associations', 'associations', 'candidates'])
    return corsify(id_row_dict)


@app.route('/task/example/random_solve/<num_candidates>', methods=['GET'])
def get_task_example_random_solve(num_candidates):
    """ Returns the relevant instance of the dataset corresponding to the received num_candidates """
    print(f"get_task_example_random_solve (num_candidates={num_candidates})")
    if num_candidates != 'random':
        id_rows = num_candidates_to_df_map[num_candidates].sample(1).iloc[0]
    else:
        id_rows = all_game_data.sample(1).iloc[0]
    id_row_dict = id_rows[['ID', 'cue', 'num_associations', 'associations', 'candidates']].to_dict()
    return corsify(id_row_dict)

@app.route('/task/example/random_create/<num_candidates>', methods=['GET'])
def get_task_example_random_create(num_candidates):
    """ Returns the relevant instance of the dataset corresponding to the received num_candidates """
    print(f"get_task_example_random_create (num_candidates={num_candidates})")
    if num_candidates != 'random':
        id_rows = num_candidates_to_df_map[num_candidates].sample(1).iloc[0]
    else:
        id_rows = all_game_data.sample(1).iloc[0]
    id_row_dict = id_rows[['ID', 'cue', 'num_associations', 'associations', 'candidates']].to_dict()
    return corsify(id_row_dict)

@app.route('/create', methods=['POST'])
def create():
    """ Receives one cue to do the clip prediction """
    data = request.json
    create_answers = get_human_score_for_fooling_ai(data)
    return corsify(create_answers)


@app.route('/create_game', methods=['POST'])
def create_game():
    """ create function for the game (with the database) """
    data = request.json
    print(data)
    return corsify(data)


@app.route('/solve_game', methods=['POST'])
def solve_game():
    """ solve function for the game (with the database) """
    data = request.json
    print(data)
    return corsify(data)


def corsify(data):
    response = jsonify(data)
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


def take_columns_to_dict(id, id_rows, columns):
    assert len(id_rows) == 1
    id_row = id_rows.iloc[0]
    id_row_dict = id_row[columns].to_dict()
    id_row_dict['id'] = id
    return id_row_dict
