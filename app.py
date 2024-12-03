from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
API_KEY = '2f8f64a3fcef4abdad73a00e71e19ee5'  # Replace with your Football-Data API key
BASE_URL = 'https://api.football-data.org/v2'  # Correct API URL


@app.route('/live_score', methods=['GET'])
def live_score():
    url = f"{BASE_URL}/matches?status=LIVE"
    headers = {"X-Auth-Token": API_KEY}

    # Make the API request
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code != 200:
        return jsonify({"response": f"Error: Unable to fetch data (Status Code {response.status_code})"})

    # Parse the response to JSON
    data = response.json()

    # Check if 'matches' key exists
    if "matches" not in data or len(data["matches"]) == 0:
        return jsonify({"response": "No live matches right now."})

    matches = data["matches"]
    results = []
    for match in matches:
        results.append(f"{match['homeTeam']['name']} {match['score']['fullTime']['homeTeam']} - "
                       f"{match['awayTeam']['name']} {match['score']['fullTime']['awayTeam']}")
    
    return jsonify({"response": results})


@app.route('/schedule', methods=['POST'])
def match_schedule():
    data = request.json
    team_name = data.get('team', '').lower()

    url = f"{BASE_URL}/teams"
    headers = {"X-Auth-Token": API_KEY}
    response = requests.get(url, headers=headers).json()

    for team in response.get("teams", []):
        if team_name in team["name"].lower():
            return jsonify({"response": f"Next match for {team['name']} is on 2024-12-01."})

    return jsonify({"response": "Team not found."})


@app.route('/player_info', methods=['POST'])
def player_info():
    data = request.json
    player_name = data.get('player', '').lower()

    # Mock response for now
    if player_name == "messi":
        return jsonify({"response": "Lionel Messi is a footballer who plays for Inter Miami."})
    return jsonify({"response": f"No info available for player: {player_name}."})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
