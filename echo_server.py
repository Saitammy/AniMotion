import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        print(f"Received: {message}")
        await websocket.send(f"Echo: {message}")

async def main():
    async with websockets.serve(echo, "localhost", 8001):
        print("WebSocket server running on ws://localhost:8001")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())