import json
from flask import Flask, render_template, request
from flask_socketio import SocketIO
import redis
import os
import openai as gpt
import pandas

##############################################Initialization######################################################
os.environ["OPENAI_API_KEY"] = "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"

app = Flask(__name__)
socketio = SocketIO(app)
###########################################REDIS PART######################################################
# Connect to your Redis instance
r = redis.Redis(host="localhost", port=6379, db=0)
#store plans from xlsx to redis
df = pandas.read_excel('demo.xlsx')

# iterate over the DataFrame rows
for _, row in df.iterrows():
    # construct the plan dict
    selling_points = json.dumps(row["Marketing Selling Points"])
    plan = {
        "name": str(row["Product Name"]),
        "active": str(row["Active"]),
        "type": str(row["Product Type"]),
        "price": str(row["Price"]),
        "market": str(row["Market"]),
        "upload_speed": str(row["Upload Speed"]),
        "download_speed": str(row["Download Speed"]),
        "selling_points": selling_points
    }

    # store the plan in Redis
    plan_key = f"plan:{row['Product ID']}"
    market = str(row["Market"])
    for field, value in plan.items():
        r.hset(plan_key, field, value)
    r.sadd(f"market:{market}", plan_key)

###########################################Methods#################################################################

def get_all_plans(area):
    plan_keys = r.smembers(f"market:{area}")
    plans = [r.hgetall(key) for key in plan_keys]
    return plans


@app.route('/')
def index():
    return render_template('consult.html')

@socketio.on('consult_with_gpt4')
def handle_message(message):
    gpt.api_key="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"
    area = message.get('area')
    plans = get_all_plans(area)
    user_input = message.get('user_input')
    plans_text = "\n".join([f"Plan {i+1}: {p[b'name'].decode()} offers {p[b'upload_speed'].decode()} Mbps upload speed and {p[b'download_speed'].decode()} Mbps download speed for ${p[b'price'].decode()} per month. Selling points: {json.loads(p[b'selling_points'].decode())}." for i, p in enumerate(plans)])

    # Prepare the messages for the chat model
    messages = [
        {"role": "system", "content":"You are an AI assistant that helps users find the best internet plans,you can only pick plans from the options that user gave you." },
        {"role":"user", "content":"you should throughly introduce all the matching plans with their selling points,Please give me a detailed explanation of your recommendations"},
        {"role": "user", "content": f"Given the following internet plans:\n{plans_text}"},
        {"role": "user", "content": user_input},
        {"role": "user", "content": "Please be friendly and talk to me like a person, don't just give me a list of recommendations"},
        {"role": "assistant", "content": "Sure! Let me find the best plan for you."},
    ]

    for response in gpt.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        n=1,
        temperature=0.8,
        stream=True,
    ):
        content = response.choices[0].get("delta", {}).get("content")
        if content is not None:
            for word in content.split():
                socketio.emit('new_recommendation', word)
                socketio.sleep(0)  
    socketio.emit('recommendation_complete', {'status': 'complete'})

if __name__ == '__main__':
    socketio.run(app)
