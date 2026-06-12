"""
test_websocket.py

Multi-turn WebSocket test for EquityMind.
Tests conversation history and session memory across multiple turns.

Run with server already started:
    uvicorn api.main:app --reload --port 8000

Then in a second terminal:
    python tests/test_websocket.py
"""

import asyncio
import json
import websockets

URI = "ws://localhost:8000/api/v1/query/stream"
HEADERS = {"Authorization": "Bearer test_key"}


async def send_question(question: str, messages: list, session_memory: dict) -> tuple:
    """Send one question and return updated messages and session_memory."""

    async with websockets.connect(URI, additional_headers=HEADERS) as ws:
        await ws.send(json.dumps({
            "question":       question,
            "messages":       messages,
            "session_memory": session_memory,
        }))

        print(f"\nQuestion: {question}")
        print("-" * 60)

        answer = ""

        async for message in ws:
            data = json.loads(message)
            event_type = data.get("type")

            if event_type == "connected":
                print(f"[connected] job_id={data.get('job_id')}")

            elif event_type == "progress":
                print(f"[progress] {data.get('message')}")

            elif event_type == "sub_progress":
                print(f"  ↳ {data.get('message')}")

            elif event_type == "token":
                answer += data.get("text", "")
                print(data.get("text", ""), end="", flush=True)

            elif event_type == "done":
                print(f"\n[done] tickers={data.get('tickers')} | intent={data.get('intent')}")
                updated_messages       = data.get("messages") or messages
                updated_session_memory = data.get("session_memory") or session_memory
                narrative = (updated_session_memory.get("narrative") or "")
                last_tickers = (updated_session_memory.get("structured") or {}).get("last_tickers", [])
                print(f"[memory] last_tickers={last_tickers}")
                print(f"[memory] narrative={narrative}...")
                return updated_messages, updated_session_memory

            elif event_type == "error":
                print(f"\n[error] {data.get('message')}")
                return messages, session_memory

    return messages, session_memory


async def main():
    messages       = []
    session_memory = {}

    print("EquityMind multi-turn test. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        messages, session_memory = await send_question(question, messages, session_memory)


asyncio.run(main())