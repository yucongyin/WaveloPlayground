from app import socketio



def emit_recommendation(response):
    for resp in response:
        content = resp.choices[0].get("delta", {}).get("content")
        if content is not None:
            socketio.emit('new_recommendation', content)
            socketio.sleep(0)
    socketio.emit('recommendation_complete', {'status': 'complete'})
