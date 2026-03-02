#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Door Lock Kiosk — start.sh
#  Launches the standalone face-recognition door kiosk.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

KIOSK_PID=""

# ── Auto-install Python dependencies ─────────────────────────────────────────
if [ -f requirements.txt ]; then
  echo "📦  Checking / installing Python dependencies…"
  pip install -q -r requirements.txt
  echo "✅  Dependencies ready."
fi

# ── Graceful teardown ─────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "🧹  Stopping Door Lock Kiosk…"
  if [ -n "$KIOSK_PID" ] && kill -0 "$KIOSK_PID" 2>/dev/null; then
    echo "  ⏹  Sending SIGTERM to kiosk (PID $KIOSK_PID)…"
    kill -TERM "$KIOSK_PID" 2>/dev/null || true
    # Wait up to 5 s for graceful exit
    for i in $(seq 1 5); do
      kill -0 "$KIOSK_PID" 2>/dev/null || break
      sleep 1
    done
    # Force-kill if still alive
    if kill -0 "$KIOSK_PID" 2>/dev/null; then
      echo "  💀  Force-killing PID $KIOSK_PID…"
      kill -KILL "$KIOSK_PID" 2>/dev/null || true
    fi
  fi
  echo "✅  Kiosk stopped."
}
trap 'cleanup' SIGINT SIGTERM EXIT

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "   🔐  Door Lock Kiosk — Face Access"
echo "══════════════════════════════════════════"
echo ""

# ── Detect OS & auto-find Arduino serial port ─────────────────────────────────
SERIAL_PORT=""
OS_TYPE="$(uname -s)"

if [ "$OS_TYPE" = "Darwin" ]; then
  # ── macOS: Arduino shows up as /dev/tty.usbmodem* ──────────────────────────
  USB_PORT=$(ls /dev/tty.usbmodem* 2>/dev/null | head -1 || true)
  if [ -n "$USB_PORT" ]; then
    echo "🍎  macOS detected — Arduino on $USB_PORT"
    SERIAL_PORT="--port $USB_PORT"
  else
    echo "ℹ️   macOS: No Arduino found on /dev/tty.usbmodem* — running in no-hardware mode."
  fi
elif [ "$OS_TYPE" = "Linux" ]; then
  # ── Ubuntu / Linux: try ttyUSB0 first, then ttyACM0 ───────────────────────
  if [ -e "/dev/ttyUSB0" ]; then
    USB_PORT="/dev/ttyUSB0"
  elif [ -e "/dev/ttyACM0" ]; then
    USB_PORT="/dev/ttyACM0"
  else
    USB_PORT=""
  fi

  if [ -n "$USB_PORT" ]; then
    echo "🐧  Ubuntu/Linux detected — Arduino on $USB_PORT"
    SERIAL_PORT="--port $USB_PORT"
  else
    echo "ℹ️   Ubuntu/Linux: No Arduino found on /dev/ttyUSB0 or /dev/ttyACM0 — running in no-hardware mode."
  fi
else
  echo "⚠️   Unknown OS ($OS_TYPE) — skipping serial auto-detect. Pass --port manually if needed."
fi

echo ""
echo "🚀  Launching Qt kiosk… (Press Q or Esc in the window to stop)"
echo ""

# ── Ensure DISPLAY is set (required by Qt on Linux/Ubuntu) ───────────────────
if [ "$OS_TYPE" = "Linux" ] && [ -z "${DISPLAY:-}" ]; then
  export DISPLAY=:0
  echo "🖥️   DISPLAY not set — defaulting to :0"
fi

# ── Launch Qt kiosk ───────────────────────────────────────────────────────────
"$PYTHON_BIN" door_kiosk_qt.py $SERIAL_PORT &
KIOSK_PID=$!
echo "    Kiosk PID: $KIOSK_PID"

wait $KIOSK_PID || true
