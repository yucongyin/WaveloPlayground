import json
from flask import render_template, request
from env import load_env
from gpt import get_gpt_response
from app_socket import emit_recommendation
from conn import create_postgres_conn, create_redis_conn
from psycopg2.extras import DictCursor
from app import app, socketio
load_env()


socketio.init_app(app)

pg_conn = create_postgres_conn()
redis_conn = create_redis_conn()

cur = pg_conn.cursor(cursor_factory=DictCursor)
cur.execute('SELECT "NAME", "IS_ACTIVE", "TYPE", "RATE_RECUR_PRICE", "MARKET_NAME", "UPLOAD_SPEED", "DOWNLOAD_SPEED", "MARKETING_SELLING_POINTS", "ID" FROM products')
rows = cur.fetchall()

# iterate over the rows
for row in rows:
    selling_points = json.dumps(row["MARKETING_SELLING_POINTS"])
    plan = {
        "name": str(row["NAME"]),
        "active": str(row["IS_ACTIVE"]),
        "type": str(row["TYPE"]),
        "price": str(row["RATE_RECUR_PRICE"]),
        "market": str(row["MARKET_NAME"]),
        "upload_speed": str(row["UPLOAD_SPEED"]),
        "download_speed": str(row["DOWNLOAD_SPEED"]),
        "selling_points": selling_points
    }

    # store the plan in Redis
    plan_key = f"plan:{row['ID']}"
    market = str(row["MARKET_NAME"])
    for field, value in plan.items():
        redis_conn.hset(plan_key, field, value)
    redis_conn.sadd(f"market:{market}", plan_key)

cur.close()
pg_conn.close()



def get_all_plans(area):
    plan_keys = redis_conn.smembers(f"market:{area}")
    plans = [redis_conn.hgetall(key) for key in plan_keys]
    return plans

@app.route('/')
def index():
    return render_template('consult.html')

@socketio.on('consult_with_gpt4')
def handle_message(message):
    area = message.get('area')
    # Add code here to get all plans from Postgres and format them for GPT-3
    plans = get_all_plans(area)
    user_input = message.get('user_input')
    plans_text = "\n".join([f"Plan {i+1} from the {p[b'market'].decode()} market: {p[b'name'].decode()} offers {p[b'upload_speed'].decode()} Mbps upload speed and {p[b'download_speed'].decode()} Mbps download speed for ${p[b'price'].decode()} per month. Selling points: {json.loads(p[b'selling_points'].decode())}." for i, p in enumerate(plans)])


    # Prepare the messages for the chat model
    messages = [
        {"role": "system", "content":"You are an AI assistant that helps users find the best internet plans,you can only pick plans from the options that user gave you." },
        {"role":"user", "content":"you should throughly introduce all the matching plans with their selling points,Please give me a detailed explanation of your recommendations"},
        {"role": "user", "content": f"Given the following internet plans:\n{plans_text}"},
        {"role": "user", "content": user_input},
        {"role": "user", "content": "Please be friendly and talk to me like a person, don't just give me a list of recommendations"},
        {"role": "user", "content": 
         '''Your recomended internet plan should be in the following format:
         In [market_name], the following plan best serves your requirements
            Plan name: XXXXX
            Feature 1: XXXXX
            Feature 2: xxxxx
            Feature 3: xxxxx
            Short description of why you recommend it.
         '''},
        {"role": "assistant", "content": "Sure! Let me find the best plan for you."},
    ]

    response = get_gpt_response(messages)
    emit_recommendation(response)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=15001,)
