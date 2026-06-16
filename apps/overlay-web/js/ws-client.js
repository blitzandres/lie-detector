export class WsClient {
  constructor(url, onConsensus, onStatus) {
    this.url = url; this.onConsensus = onConsensus; this.onStatus = onStatus;
    this.ws = null; this.connected = false; this._reconnectTimer = null;
  }

  connect() {
    this.ws = new WebSocket(this.url);
    this.ws.onopen = () => { this.connected = true; this.onStatus("engine-online"); };
    this.ws.onmessage = (e) => this.onConsensus(JSON.parse(e.data));
    this.ws.onclose = () => {
      this.connected = false; this.onStatus("engine-offline");
      this._reconnectTimer = setTimeout(() => this.connect(), 1500);  // auto-reconnect (spec §8)
    };
    this.ws.onerror = () => this.ws.close();
  }

  send(frame) {
    if (this.connected && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(frame));
    }
  }
}
