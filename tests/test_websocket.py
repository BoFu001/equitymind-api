"""
test_websocket.py

Quick test for the WebSocket streaming endpoint.
Tests both Layer 1 (progress events) and Layer 2 (token streaming).

Run with server already started:
    uvicorn api.main:app --reload --port 8000

Then in a second terminal:
    python test_websocket.py
"""

import asyncio
import json
import websockets


async def test():
    uri = "ws://localhost:8000/api/v1/query/stream"
    question = input("Enter your question: ")

    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "question": question,
            "session_id": "test-session-001"
        }))
        print(f"Question sent: {question}")
        print("-" * 60)

        token_count = 0
        answer = ""

        async for message in ws:
            data = json.loads(message)
            event_type = data.get("type")

            if event_type == "connected":
                print(f"[connected]  job_id={data.get('job_id')}")

            elif event_type == "progress":
                print(f"[progress]   node={data.get('node')} | {data.get('message')}")

            elif event_type == "sub_progress":
                print(f"  ↳ {data.get('message')}")
                
            elif event_type == "token":
                token_count += 1
                answer += data.get("text", "")
                # Print a dot for each token to show streaming is working
                print(data.get("text", ""), end="", flush=True)

            elif event_type == "done":
                print(f"\n[done]       job_id={data.get('job_id')} | ticker={data.get('ticker')} | intent={data.get('intent')}")
                break

            elif event_type == "error":
                print(f"\n[error]      {data.get('message')}")
                break

        print("-" * 60)
        print(f"Tokens received: {token_count}")
        print(f"Answer length:   {len(answer)} chars")
        print(f"Full answer:\n{answer}")


asyncio.run(test())