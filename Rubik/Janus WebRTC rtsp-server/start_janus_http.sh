HTTP_PORT=8080
JANUS_LOGLEVEL=4

echo "🚀 Starte Janus WebRTC Gateway (Log-Level ${JANUS_LOGLEVEL}) im Hintergrund..."

# Prüfen ob Janus vorhanden ist
if ! command -v janus >/dev/null 2>&1; then
    echo "❌ Janus wurde nicht gefunden!"
    exit 1
fi

# Janus im Hintergrund starten
janus -L "$JANUS_LOGLEVEL" > janus.log 2>&1 &

JANUS_PID=$!
sleep 2

if ps -p $JANUS_PID > /dev/null; then
    echo "✅ Janus läuft (PID: $JANUS_PID)"
else
    echo "❌ Janus konnte nicht gestartet werden"
    exit 1
fi

echo "📂 Wechsel in ${JANUS_HTML_DIR}"
cd "$JANUS_HTML_DIR" || {
    echo "❌ Verzeichnis nicht gefunden!"
    exit 1
}

echo "🌐 Starte Python HTTP Server auf Port ${HTTP_PORT}"
python3 -m http.server "$HTTP_PORT"
