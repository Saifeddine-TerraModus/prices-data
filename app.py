from flask import Flask, Response, render_template, request, url_for, redirect, jsonify, session
from passlib.hash import pbkdf2_sha256
import uuid
from pymongo import MongoClient
import json
from bson.objectid import ObjectId 
from functools import wraps
import pandas as pd
import pickle
import math
import geopy.distance
from shapely.geometry import Point,LineString,Polygon
import shapely.wkt
import numpy as np
import asyncio
from fastapi import FastAPI
from starlette.responses import JSONResponse
import xgboost as xgb

import os

print(os.getcwd())  # print the current working directory

# Load the pickled models
model_path = "Models/model_rabat8.pkl"
pipe3 = xgb.Booster()
pipe3.load_model(model_path)
pipe = pickle.load(open("Models/Terramodus_module.pkl",'rb'))
pipe2 = pickle.load(open("Models/model_casablanca2.pkl",'rb'))
model_path2 = "Models/model_marrakech1.pkl"
pipe4 = xgb.Booster()
pipe4.load_model(model_path2)

# Load dataframes
temara_coordinates = pd.read_excel('coordonnes_CI.xlsx')
Quartiers_temara_df = pd.read_excel('Quartiers_points.xlsx')
Quartiers_casablanca_df = pd.read_excel('Quartier_points_casablanca.xlsx')
Arrondissement_casablanca_df = pd.read_excel('Casa_arrondissement.xlsx')

app = Flask(__name__)  
app.secret_key = b'\xcc^\x91\xea\x17-\xd0W\x03\xa7\xf8J0\xac8\xc5'



# load the pickled model from file

try:
    #launch the data base client
    client = MongoClient('localhost', 27017)
    db = client.user_login_system
    print("system")
    records = db.users
    print("register")
except:
    print("Error cannot connect to mongodb!!!")



# Decorators
def login_required(f):
  @wraps(f)
  def wrap(*args, **kwargs):
    if 'logged_in' in session:
      return f(*args, **kwargs)
    else:
      return redirect('/')
  return wrap

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/estimation')
def estimation():
  return render_template('index.html')
  #return render_template('index.html')  

@app.route('/Chatbot')
def realestimation():
  return render_template('Loading.html')
  #return render_template('index.html')

@app.route('/dashboard/')
@login_required
def dashboard():
    return render_template('dashboard.html')

def start_session(user):
  del user['password']
  session['logged_in'] = True
  session['user'] = user
  return jsonify(user), 200    

#right_function
@app.route('/user/signup', methods=['POST'])
def create_user():
    # Create the user object
    user = {
      "_id": uuid.uuid4().hex,
      "name": request.form.get('name'),
      "email": request.form.get('email'),
      "telephone":request.form.get('telephone'),
      "password": request.form.get('password')
    }
    user['password'] = pbkdf2_sha256.hash(user['password'])

    #check if the email is already used

    if db.users.find_one({ "email": user['email'] }):
        return jsonify({ "error": "this email is already used" }), 400
    if db.users.insert_one(user):
        return start_session(user)
    return jsonify({ "error": "Signup failed" }), 400


@app.route('/user/signout')
def signout():
  session.clear()
  return redirect('/')

@app.route('/user/login', methods=['POST'])
def login():
  try:
    user = db.users.find_one({"email": request.form.get('email')})
    if user and pbkdf2_sha256.verify(request.form.get('password'), user['password']):
      return start_session(user)
    
    return jsonify({ "error": "Wrong Credentials : The username or password is incorrect" }), 401

  except Exception as ex:
    print(ex)

async def load_data(filepath):
  df = pd.read_excel(filepath)
  return df
async def main():
  print("---------------------Loading data----------------------------")
  filepaths = ["bidonsvilles.xlsx", "NodeFin.xlsx", "casapoints_withattributes.xlsx","rabatpoints_withattributes.xlsx","marrakechpoints_withattributes.xlsx"]
  tasks = [asyncio.ensure_future(load_data(filepath)) for filepath in filepaths]
  dataframes = await asyncio.gather(*tasks)
  bidonevilles_temara_df, temara_nodes_df, casablanca_nodes_df, rabat_nodes_df, marrakech_nodes= dataframes
  return bidonevilles_temara_df, temara_nodes_df, casablanca_nodes_df, rabat_nodes_df, marrakech_nodes

#loop = asyncio.get_event_loop()
#db1, gdf_nodes1, gdf_nodes2, gdf_nodes3, marrakech_nodes= loop.run_until_complete(main())

@app.route('/predict', methods=['POST'])
def predict():
  # Get input values from the request
  Longitude = request.form.get('a')
  Latitude = request.form.get('b')
  Superficie = request.form.get('c')

  # Check if any input value is missing
  if not Longitude or not Latitude or not Superficie:
      return jsonify({'error': 'Missing input values'}), 400

  # Validate input values
  try:
      Longitude = float(Longitude)
      Latitude = float(Latitude)
      Superficie = int(Superficie)
  except ValueError:
      return jsonify({'error': 'Invalid input values'}), 400

  # Create a Point object from the input values
  point = Point(Latitude, Longitude)

  # Perform prediction
  if Longitude >= 33.4988269019536 and Longitude <= 33.62416420545495 and Latitude >= -7.712288563385707 and Latitude <= -7.422723604704283 :
    # Casablanca prediction
    """quartier_casablanca_df = Quartiers_points_f2.apply(lambda row: point.hausdorff_distance(Point(row.X, row.Y)), axis=1)
    quartier = quartier_distances2.idxmin()
    Arrondissement_diss = Arrondissement_casablanca_df.apply(lambda row: point.hausdorff_distance(Point(row.X, row.Y)), axis=1)
    Arrondissement = Arrondissement_diss.idxmin()
    gdf_nodes = gdf_nodes2.copy()
    features = [Superficie]
    node_distances = np.apply_along_axis(lambda x: point.hausdorff_distance(Point(x[0], x[1])), 1, gdf_nodes[['x', 'y']].values)
    closest_node = gdf_nodes.iloc[np.argmin(node_distances)]
    features += list(closest_node[3:].values)
    features.append(quartier)
    features.append(Arrondissement)
    prediction = int(pipe2.predict(np.array(features).reshape(1,-1)))"""
    gdf_nodes = casablanca_nodes_df.copy()
    features = [Superficie]
    node_distances = np.apply_along_axis(lambda x: point.hausdorff_distance(Point(x[0], x[1])), 1, gdf_nodes[['x', 'y']].values)
    closest_node = gdf_nodes.iloc[np.argmin(node_distances)]
    features += list(closest_node[2:].values)
    prediction = int(pipe2.predict(np.array(features).reshape(1,-1)))

  elif Longitude <= 33.9442616274894 and Longitude >= 33.86847250864682 and Latitude >= -6.876605831192356 and Latitude <= -6.9331121018731015 :
    # Temara prediction
    quartier_distances = Quartiers_temara_df.apply(lambda row: point.hausdorff_distance(Point(row.X, row.Y)), axis=1)
    quartier = quartier_distances.idxmin()
    bd = bidonevilles_temara_df.copy()
    gdf_nodes = temara_nodes_df.copy()
    features = [Superficie]
  
    bd.geometry = bd.geometry.apply(lambda x: shapely.wkt.loads(x))
    node_distances = np.apply_along_axis(lambda x: point.hausdorff_distance(Point(x[0], x[1])), 1, gdf_nodes[['x', 'y']].values)
    closest_node = gdf_nodes.iloc[np.argmin(node_distances)]
    features += list(closest_node[3:].values)
    features.append(min([point.hausdorff_distance(row) for row in bd.geometry]))
    features.append(quartier)
    
    prediction = int(pipe.predict(np.array(features).reshape(1,-1)))
  elif Longitude <= 34.040636 and Longitude >= 33.897702 and Latitude >= -6.891597 and Latitude <= -6.759761 :
    # Rabat prediction
    gdf_nodes = rabat_nodes_df.copy()
    features = [Superficie]
    node_distances = np.apply_along_axis(lambda x: point.hausdorff_distance(Point(x[0], x[1])), 1, gdf_nodes[['x', 'y']].values)
    closest_node = gdf_nodes.iloc[np.argmin(node_distances)]
    features += list(closest_node[2:].values)
    d_features = xgb.DMatrix(np.array(features).reshape(1, -1))
    prediction = int(pipe3.predict(d_features))
  else :
    # marrakech prediction

    gdf_nodes = marrakech_nodes.copy()
    features = [Superficie]
    node_distances = np.apply_along_axis(lambda x: point.hausdorff_distance(Point(x[0], x[1])), 1, gdf_nodes[['x', 'y']].values)
    closest_node = gdf_nodes.iloc[np.argmin(node_distances)]
    features += list(closest_node[2:].values)
    d_features = xgb.DMatrix(np.array(features).reshape(1, -1))
    prediction = int(pipe4.predict(d_features))
  
  return '{:,}'.format(prediction).replace(',', ' ') + " MAD"

@app.route('/location', methods=['POST'])
def predict_location():
  # Get input values from the request 
  Longitude = request.form.get('a')
  Latitude = request.form.get('b')
  Superficie = request.form.get('c')

  # Check if any input value is missing
  if not Longitude or not Latitude or not Superficie:
      return jsonify({'error': 'Missing input values'}), 400

  # Validate input values
  try:
      Longitude = float(Longitude)
      Latitude = float(Latitude)
      Superficie = int(Superficie)
  except ValueError:
      return jsonify({'error': 'Invalid input values'}), 400

  # Create a Point object from the input values
  point = Point(Latitude, Longitude)
  # Perform prediction
  if Longitude >= 33.4988269019536 and Longitude <= 33.62416420545495 and Latitude >= -7.712288563385707 and Latitude <= -7.422723604704283 :
    # Casablanca prediction

    gdf_nodes = casablanca_nodes_df.copy()
    features = [Superficie]
    node_distances = np.apply_along_axis(lambda x: point.hausdorff_distance(Point(x[0], x[1])), 1, gdf_nodes[['x', 'y']].values)
    closest_node = gdf_nodes.iloc[np.argmin(node_distances)]
    features += list(closest_node[2:].values)
    prediction = int(pipe2.predict(np.array(features).reshape(1,-1)))
    location = int((prediction*0.05)/12)

  elif Longitude <= 33.9442616274894 and Longitude >= 33.86847250864682 and Latitude >= -6.876605831192356 and Latitude <= -6.9331121018731015 :
    # Temara prediction
    quartier_distances = Quartiers_temara_df.apply(lambda row: point.hausdorff_distance(Point(row.X, row.Y)), axis=1)
    quartier = quartier_distances.idxmin()

    features = [Superficie]
    bd = bidonevilles_temara_df.copy()
    gdf_nodes = temara_nodes_df.copy()
    bd.geometry = bd.geometry.apply(lambda x: shapely.wkt.loads(x))
    node_distances = np.apply_along_axis(lambda x: point.hausdorff_distance(Point(x[0], x[1])), 1, gdf_nodes[['x', 'y']].values)
    closest_node = gdf_nodes.iloc[np.argmin(node_distances)]
    features += list(closest_node[3:].values)
    features.append(min([point.hausdorff_distance(row) for row in bd.geometry]))
    features.append(quartier)
    prediction = pipe.predict(np.array(features).reshape(1,-1))
    location = int((prediction*0.05)/12)
  
  elif Longitude <= 34.040636 and Longitude >= 33.897702 and Latitude >= -6.891597 and Latitude <= -6.759761 :
    # Temara prediction
    gdf_nodes = rabat_nodes_df.copy()
    features = [Superficie]
    node_distances = np.apply_along_axis(lambda x: point.hausdorff_distance(Point(x[0], x[1])), 1, gdf_nodes[['x', 'y']].values)
    closest_node = gdf_nodes.iloc[np.argmin(node_distances)]
    features += list(closest_node[2:].values)
    d_features = xgb.DMatrix(np.array(features).reshape(1, -1))
    prediction = int(pipe3.predict(d_features))
    location = int((prediction*0.05)/12)
  else :
    # Casablanca prediction

    gdf_nodes = marrakech_nodes.copy()
    features = [Superficie]
    node_distances = np.apply_along_axis(lambda x: point.hausdorff_distance(Point(x[0], x[1])), 1, gdf_nodes[['x', 'y']].values)
    closest_node = gdf_nodes.iloc[np.argmin(node_distances)]
    features += list(closest_node[2:].values)
    d_features = xgb.DMatrix(np.array(features).reshape(1, -1))
    prediction = int(pipe4.predict(d_features))
    location = int((prediction*0.05)/12)
  

  return jsonify('{:,}'.format(location ).replace(',', ' ') + " MAD"), 200
@app.route('/Heat Map')
def Heatmap():
  
  return render_template('map_with_bar_chart (2).html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

    

  

    