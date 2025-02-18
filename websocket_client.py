import asyncio
import websockets
import json

AUTH_TOKEN = "your_auth_token"
VTS_WS_URL = "ws://localhost:8001"

async def websocket_task():
    global ear_left, ear_right, mar
    while True:
        try:
            async with websockets.connect(VTS_WS_URL) as websocket:
                print("Connected to VTube Studio!")

                # Authenticate
                auth_message = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": "AuthRequest",
                    "messageType": "AuthenticationRequest",
                    "data": {
                        "pluginName": "FacialTracker",
                        "pluginDeveloper": "YourName",
                        "authenticationToken": AUTH_TOKEN
                    }
                }
                await websocket.send(json.dumps(auth_message))
                response = await websocket.recv()
                print("Authentication Response:", response)

                while True:
                    if ear_left is not None and ear_right is not None and mar is not None:
                        expression_data = {
                            "apiName": "VTubeStudioPublicAPI",
                            "apiVersion": "1.0",
                            "requestID": "ExpressionRequest",
                            "messageType": "ExpressionActivationRequest",
                            "data": {
                                "expressions": [
                                    {"id": "EyeLeftX", "value": ear_left},
                                    {"id": "EyeRightX", "value": ear_right},
                                    {"id": "MouthOpen", "value": mar}
                                ]
                            }
                        }
                        await websocket.send(json.dumps(expression_data))
                    await asyncio.sleep(0.05)
        except Exception as e:
            print(f"WebSocket connection error: {e}, retrying in 3 seconds...")
            await asyncio.sleep(3)

def start_websocket():
    asyncio.run(websocket_task())
