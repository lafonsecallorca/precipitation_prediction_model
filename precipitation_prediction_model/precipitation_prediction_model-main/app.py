from flask import Flask, render_template, request, redirect, url_for, session
from geopy.geocoders import GoogleV3
import requests
from flask_wtf import FlaskForm
from wtforms import Form, StringField, validators
from wtforms.validators import DataRequired
import secrets

app = Flask(__name__)
# Generate a secret key
secret_key = secrets.token_hex(16)
app.config['SECRET_KEY'] = secret_key

MODEL_API_URL = "http://127.0.0.1:8000/"


# Function to get latitude and longitude from ZIP code using Google Maps API
def get_coordinates_from_zip(zip_code):
    api_key = 'AIzaSyCHy3jGmG7yN0EECyKhTtfSRLgXqHZ894M'
    geolocator = GoogleV3(api_key=api_key)
    location = geolocator.geocode(zip_code)
    if location:
        return location.latitude, location.longitude
    return None, None


def helper_request_method(endpoint, lat, lon):
    api_url = MODEL_API_URL + endpoint
    json_data = {"latitude": lat,
                 "longitude": lon}
    return requests.get(api_url, json=json_data)


class LocationForm(FlaskForm):
    location = StringField('Zip Code/City, State', validators=[DataRequired()])


@app.route("/")
def home():
    form = LocationForm()
    result = session.get('result', {})
    daily = session.get('daily', {})
    trihourly = session.get('trihourly', {})
    return render_template("index.html", form=form, result=result, daily=daily, trihourly=trihourly)


@app.route("/current", methods=['GET', 'POST'])
def current():
    form = LocationForm()
    daily = session.get('daily', {})
    if form.validate_on_submit():
        location_value = form.location.data
        latitude, longitude = get_coordinates_from_zip(location_value)
        end_point = "predict_hourlyML/current_weather"
        response = helper_request_method(end_point, latitude, longitude)
        if response.status_code == 200:
            result = response.json()
            session['result'] = result  # Store 'result' in session
            return redirect(url_for('home'))  # Redirect to home to update the template
        else:
            return render_template("error.html")

    return render_template('index.html', form=form)


@app.route("/current_daily", methods=['GET', 'POST'])
def current_daily():
    form = LocationForm()
    result = session.get('result', {})
    if form.validate_on_submit():
        location_value = form.location.data
        latitude, longitude = get_coordinates_from_zip(location_value)
        end_point = "predict_dailyML/current_weather"
        response = helper_request_method(end_point, latitude, longitude)
        if response.status_code == 200:
            daily = response.json()
            session['daily'] = daily  # Store 'daily' in session
            return redirect(url_for('home'))  # Redirect to home to update the template
        else:
            return render_template("error.html")

    return render_template('index.html', form=form)

@app.route("/trihourly", methods=['GET', 'POST'])
def trihourly():
    form = LocationForm()
    result = session.get('trihourly', {})
    if form.validate_on_submit():
        location_value = form.location.data
        latitude, longitude = get_coordinates_from_zip(location_value)
        end_point = "predict_hourlyML/fiveday/trihourly_forecast"
        response = helper_request_method(end_point, latitude, longitude)
        if response.status_code == 200:
            trihourly = response.json()
            session['trihourly'] = trihourly  # Store 'daily' in session
            return redirect(url_for('home'))  # Redirect to home to update the template
        else:
            return render_template("error.html")

    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
