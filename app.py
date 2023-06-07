import json
from flask import Flask, render_template, request
import redis
import os
import openai as gpt
import pandas
import timeit

##############################################Initialization######################################################
os.environ["OPENAI_API_KEY"] = "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"

app = Flask(__name__)

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

def process_input_with_gpt4(user_input,area):
    start_time = timeit.default_timer()
    plans = get_all_plans(area)
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Get All Plan Execution time: {execution_time} seconds")
    start_time = timeit.default_timer()
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
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Arrange Messaging Execution time: {execution_time} seconds")
    start_time = timeit.default_timer()
    # Use the v1/chat/completions endpoint
    response = gpt.ChatCompletion.create(
        model="gpt-3.5-turbo", # Replace with the appropriate chat model
        messages=messages,
        max_tokens=1000,
        n=1,
        temperature=0.8,
    )
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Get response from server Execution time: {execution_time} seconds")
    query = response.choices[0].message["content"].strip()
    print(response)
    print(query)
    return query


def consult_with_gpt4(user_input,area):
    start_time = timeit.default_timer()
    plans = get_all_plans(area)
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Get All Plan Execution time: {execution_time} seconds")
    start_time = timeit.default_timer()
    plans_text = "\n".join([f"Plan {i+1}: {p[b'name'].decode()} offers {p[b'upload_speed'].decode()} Mbps upload speed and {p[b'download_speed'].decode()} Mbps download speed for ${p[b'price'].decode()} per month. Selling points: {json.loads(p[b'selling_points'].decode())}." for i, p in enumerate(plans)])

    # Prepare the messages for the chat model
    messages = [
        {"role": "system", "content":"You are an AI assistant that helps users find the best internet plans,you can only pick plans from the options that user gave you, and you should throughly introduce all the matching plans." },
        {"role":"user", "content":"you should throughly introduce all the matching plans with their selling points"},
        {"role": "user", "content": f"Given the following internet plans:\n{plans_text},could you give me your best recommendation?"},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": "Sure! Let me find the best plan for you."},
    ]
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Arrange Messaging Execution time: {execution_time} seconds")
    start_time = timeit.default_timer()
    #print(messages)
    # Use the v1/chat/completions endpoint
    response = gpt.ChatCompletion.create(
        model="gpt-3.5-turbo", # Replace with the appropriate chat model
        messages=messages,
        max_tokens=1000,
        n=1,
        temperature=0.8,
    )
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Get response from server Execution time: {execution_time} seconds")
    query = response.choices[0].message["content"].strip()
    print(response)
    print(query)
    return query

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/consult")
def consult():
    return render_template("consult-origin.html")

@app.route("/consult-result", methods=["POST"])
def consultresult():
    area = request.form["area"]
    user_input = request.form["user_input"]
    recommendations = consult_with_gpt4(user_input,area)
    
    return render_template("result.html", recommendations=recommendations)

@app.route("/search", methods=["POST"])
def search():
    speed = request.form["speed"]
    budget = request.form["budget"]
    area = request.form["area"]

    # Combine the inputs into a single user requirement string
    user_input = f"I want internet speed of at least {speed} Mbps. My budget is ${budget} per month. I live in {area}, provide all the plans matches."
    
    recommendations = process_input_with_gpt4(user_input,area)
    
    return render_template("result.html", recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)
