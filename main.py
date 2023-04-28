from flask import Flask, render_template, request
import pandas as pd
import pickle
import folium
from folium.plugins import MarkerCluster

app = Flask(__name__, static_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Get user input values
    month = int(request.form['month'])
    year = int(request.form['year'])
    date = pd.to_datetime('{}-{}-01'.format(year, month))
  
    with open('xg2.pickle', 'rb') as f:
        xg = pickle.load(f)

        data = pd.read_csv("eda_lat_lon.csv")
    # Create a dictionary to store the predictions and coordinates for all cities
    predictions = {}

    # Iterate over all city-encoded values
    for city_encoded in range(1, 27):
        # Create a user input dataframe for the current city
        user_input = pd.DataFrame({

            'pm2.5': [0],
            'pm10': [0],
            'no': [0],
            'no2': [0],
            'nox': [0],
            'nh3': [0],
            'co': [0],
            'so2': [0],
            'o3': [0],
            'benzene': [0],
            'toluene': [0],
            'month': [month],
            'year': [year],
            'city_encoded': [city_encoded]
        })

        # Calculate the mean values for the other features for the specified city and month
        city_month_data = data[(data['city_encoded'] == city_encoded) & (data['month'] == month)]
        mean_pm2_5 = city_month_data['pm2.5'].mean()
        mean_pm10 = city_month_data['pm10'].mean()
        mean_no = city_month_data['no'].mean()
        mean_no2 = city_month_data['no2'].mean()
        mean_nox = city_month_data['nox'].mean()
        mean_nh3 = city_month_data['nh3'].mean()
        mean_co = city_month_data['co'].mean()
        mean_so2 = city_month_data['so2'].mean()
        mean_o3 = city_month_data['o3'].mean()
        mean_benzene = city_month_data['benzene'].mean()
        mean_toluene = city_month_data['toluene'].mean()

        # Update the user input dataframe with the mean values for the other features
        user_input['pm2.5'] = mean_pm2_5
        user_input['pm10'] = mean_pm10
        user_input['no'] = mean_no
        user_input['no2'] = mean_no2
        user_input['nox'] = mean_nox
        user_input['nh3'] = mean_nh3
        user_input['co'] = mean_co
        user_input['so2'] = mean_so2
        user_input['o3'] = mean_o3
        user_input['benzene'] = mean_benzene
        user_input['toluene'] = mean_toluene

        # Use the model to make prediction
        prediction = xg.predict(user_input)

        # Add the prediction and coordinates to the dictionary
        city_data = data[(data['city_encoded'] == city_encoded)].iloc[0]
        predictions[city_data['city']] = {'prediction': float(prediction[0]), 'lat': city_data['lat'],
                                          'lon': city_data['lon']}

    # Create a map centered at India
    map = folium.Map(location=[22.5726, 88], zoom_start=4)
    heatmap_data = [[data['lat'], data['lon'], data['prediction']] for city, data in predictions.items()]
    heatmap = folium.plugins.HeatMap(heatmap_data)
    heatmap.add_to(map)

    # Save the map to a file
    map.save('templates/map.html')
    # Add markers for each city with prediction and coordinate

    return render_template('home.html', predictions=predictions)


if __name__ == '__main__':
    app.run(debug=True)
