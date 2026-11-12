// WebSocket connection with automatic reconnect.
export function connect(handlers) {
  let ws = null;
  const api = {
    send(obj) {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
      else handlers.error?.({ message: "Not connected to the server." });
    },
  };
  function open() {
    handlers.status?.("connecting");
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.onopen = () => handlers.status?.("connected");
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      handlers[m.type]?.(m);
    };
    ws.onclose = () => {
      handlers.status?.("disconnected");
      setTimeout(open, 1500);
    };
  }
  open();
  return api;
}
