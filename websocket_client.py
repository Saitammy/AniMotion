import asyncio
import websockets
import logging
import configparser

async def websocket_handler(shared_vars, data_lock, uri):
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                logging.info(f"Connected to WebSocket server at {uri}")
                while True:
                    # Example: packaging shared variables into a message (can be extended to JSON).
                    with data_lock:
<<<<<<< HEAD
                        # For this example, we send a simple message with the latest lip sync value.
                        message = f"LipSync: {shared_vars.lip_sync_value}"
                    await websocket.send(message)
                    await asyncio.sleep(1)  # Adjust the interval as needed.
        except Exception as e:
            logging.error(f"WebSocket connection error: {e}")
            # Back off before retrying the connection.
            await asyncio.sleep(10)

def start_websocket(shared_vars, data_lock):
=======
                        # For this complete code, we're sending a simple test message.
                        message = f"LipSync: {shared_vars.lip_sync_value}"
                    await websocket.send(message)
                    await asyncio.sleep(1)  # Adjust the sending interval as needed.
        except Exception as e:
            logging.error(f"WebSocket connection error: {e}")
            # Wait before retrying the connection.
            await asyncio.sleep(10)

def start_websocket(shared_vars, data_lock):
    # Read WebSocket configuration.
>>>>>>> ef7be31febf199688a6915a3627b4b256863f376
    config = configparser.ConfigParser()
    config.read('config.ini')
    uri = config.get('WebSocket', 'VTS_WS_URL', fallback="ws://localhost:8001")
    asyncio.run(websocket_handler(shared_vars, data_lock, uri))