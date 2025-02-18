# websocket_client.py

import asyncio
import websockets
import json

AUTH_TOKEN = "your_auth_token"
VTS_WS_URL = "ws://localhost:8001"

async def websocket_task():
    global ear_left, ear_right, mar, ebr_left, ebr_right, lip_sync_value, yaw, pitch, roll
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
                    if None not in (ear_left, ear_right, mar, ebr_left, ebr_right, lip_sync_value):
                        expression_data = {
                            "apiName": "VTubeStudioPublicAPI",
                            "apiVersion": "1.0",
                            "requestID": "ExpressionRequest",
                            "messageType": "ExpressionActivationRequest",
                            "data": {
                                "expressions": [
                                    {"id": "EyeLeftX", "value": ear_left},
                                    {"id": "EyeRightX", "value": ear_right},
                                    {"id": "MouthOpen", "value": mar},
                                    {"id": "BrowLeftY", "value": ebr_left},
                                    {"id": "BrowRightY", "value": ebr_right},
                                    {"id": "LipSync", "value": lip_sync_value},
                                    {"id": "HeadYaw", "value": yaw},
                                    {"id": "HeadPitch", "value": pitch},
                                    {"id": "HeadRoll", "value": roll}
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
