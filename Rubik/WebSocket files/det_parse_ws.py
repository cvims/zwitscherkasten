#!/usr/bin/env python3
import asyncio
import sys
import re
import os
import signal
import json
import websockets

clients = set()

# kompletter ObjectDetection-Block (wie in deinem ursprünglichen Regex)
OBJ_RE = re.compile(r'\{\s*\(structure\)"ObjectDetection.*?;"\s*\}', re.DOTALL)

# timestamp=(guint64)...
TS_RE = re.compile(r'timestamp\\=\\\(guint64\\\)(\d+)')

# einzelnes bbox-entry (deine ursprüngliche Regex, unverändert)
BOX_RE = re.compile(
    r'\\"(?P<label>[^\\"]+?)\\\\,\\\\\s*id\\\\=\\\\\(uint\\\\\)(?P<id>\d+)'
    r'.*?confidence\\\\=\\\\\(double\\\\\)(?P<conf>[0-9.]+)'
    r'.*?rectangle\\\\=\\\\\(float\\\\\)\\\\\<\\\\\s*(?P<x>[0-9.]+)\\\\,\\\\\s*(?P<y>[0-9.]+)\\\\,\\\\\s*(?P<w>[0-9.]+)\\\\,\\\\\s*(?P<h>[0-9.]+)',
    re.DOTALL
)


# websockets library: deine Version ruft handler(ws) (oder ws,path).
# Wir definieren handler(ws, path=None) um kompatibel zu sein.
# ersetze die bestehende ws_handler(...) Definition durch folgendes:

async def ws_handler(websocket, path=None):
    """
    Akzeptiert eingehende WebSocket-Verbindungen.
    - Fügt websocket zu clients hinzu.
    - Läuft einen Empfangs-Loop: alle eingehenden Messages werden an
      die anderen verbundenen Clients weitergereicht.
    - Wenn die Connection schließt, wird sie entfernt.
    """
    clients.add(websocket)
    try:
        async for msg in websocket:
            # Optional: falls dein Sender newline-terminierte JSON schickt,
            # strip() kann helfen — entferne falls nicht gewünscht.
            if isinstance(msg, str):
                payload = msg.strip()
            else:
                # binary data: wir versuchen, es unverändert weiterzureichen
                payload = msg

            # Broadcast an alle *anderen* Clients
            dead = []
            for ws in list(clients):
                if ws is websocket:
                    continue
                try:
                    await ws.send(payload)
                except Exception:
                    dead.append(ws)

            for ws in dead:
                clients.discard(ws)

    except websockets.ConnectionClosed:
        pass
    finally:
        clients.discard(websocket)
        # Ensure closed
        try:
            await websocket.close()
        except Exception:
            pass


async def broadcast_json(obj: dict):
    if not clients:
        return

    data = json.dumps(obj)
    dead = []
    for ws in list(clients):
        try:
            await ws.send(data)
        except Exception:
            dead.append(ws)

    for ws in dead:
        clients.discard(ws)


def parse_detections_from_buf(buf: bytes):
    """
    Nimmt bytes-Buffer, decodiert (ignore errors),
    extrahiert alle kompletten ObjectDetection-Blöcke,
    liefert (messages:list[dict], rest_bytes).
    """
    # Buffer begrenzen (Schutz gegen Memory-Wachstum)
    if len(buf) > 200_000:
        buf = buf[-50_000:]

    text = buf.decode("utf-8", errors="ignore")
    matches = list(OBJ_RE.finditer(text))
    if not matches:
        return [], buf

    out = []
    last_end = 0

    for m in matches:
        last_end = m.end()
        block = m.group(0)

        ts_m = TS_RE.search(block)
        ts = int(ts_m.group(1)) if ts_m else "?"

        for b in BOX_RE.finditer(block):
            label = b.group("label")
            cid = int(b.group("id"))
            try:
                conf = float(b.group("conf"))
            except Exception:
                conf = 0.0
            try:
                x = float(b.group("x"))
                y = float(b.group("y"))
                w = float(b.group("w"))
                h = float(b.group("h"))
            except Exception:
                x = y = w = h = 0.0

            msg = {
                "ts": ts,
                "label": label,
                "class_id": cid,
                "conf": conf,                 # as float (0..1 if source uses that)
                "conf_pct": conf * 100.0,     # convenience field
                "bbox": [x, y, w, h],
                "raw": block[:1000]           # truncated original block for debugging
            }
            out.append(msg)

    # Buffer nach dem letzten vollständigen Match kürzen
    rest = text[last_end:].encode("utf-8", errors="ignore")
    return out, rest


async def main():
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    # sauberes Shutdown bei SIGINT/SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop.set())

    # WebSocket server starten
    server = await websockets.serve(ws_handler, "0.0.0.0", 9001)

    q = asyncio.Queue(maxsize=200)
    buf = b""

    # stdin non-blocking + reader
    fd = sys.stdin.fileno()
    try:
        os.set_blocking(fd, False)
    except Exception:
        pass

    def on_stdin_readable():
        try:
            data = os.read(fd, 4096)
        except BlockingIOError:
            return
        except OSError:
            try:
                loop.remove_reader(fd)
            except Exception:
                pass
            stop.set()
            return

        if not data:
            try:
                loop.remove_reader(fd)
            except Exception:
                pass
            stop.set()
            return

        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            # Drop on overflow
            pass

    loop.add_reader(fd, on_stdin_readable)

    async def consumer():
        nonlocal buf
        while not stop.is_set():
            try:
                data = await asyncio.wait_for(q.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            buf += data
            msgs, buf = parse_detections_from_buf(buf)
            for msg in msgs:
                await broadcast_json(msg)

    consumer_task = asyncio.create_task(consumer())

    try:
        await stop.wait()
    finally:
        try:
            loop.remove_reader(fd)
        except Exception:
            pass

        consumer_task.cancel()
        try:
            await consumer_task
        except Exception:
            pass

        for ws in list(clients):
            try:
                await ws.close()
            except Exception:
                pass
        clients.clear()

        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
