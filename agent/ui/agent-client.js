/**
 * GeminiClient: Handles WebSocket communication
 */
class GeminiClient {
  constructor(config) {
    this.websocket = null;
    this.onOpen = config.onOpen;
    this.onMessage = config.onMessage;
    this.onClose = config.onClose;
    this.onError = config.onError;
  }

  connect(params = {}) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let wsUrl = `${protocol}//${window.location.host}/ws`;

    // Append query parameters if provided
    const queryParts = [];
    for (const [key, value] of Object.entries(params)) {
      if (value != null && value !== '') {
        queryParts.push(
            `${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
      }
    }
    if (queryParts.length > 0) {
      wsUrl += '?' + queryParts.join('&');
    }


    this.websocket = new WebSocket(wsUrl);
    this.websocket.binaryType = 'arraybuffer';

    this.websocket.onopen = () => {
      if (this.onOpen) this.onOpen();
    };

    this.websocket.onmessage = (event) => {
      if (this.onMessage) this.onMessage(event);
    };

    this.websocket.onclose = (event) => {
      if (this.onClose) this.onClose(event);
    };

    this.websocket.onerror = (event) => {
      if (this.onError) this.onError(event);
    };
  }

  send(data) {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(data);
    }
  }

  sendText(text) {
    this.send(text);
  }

  sendImage(base64Data, mimeType = 'image/jpeg') {
    this.send(JSON.stringify({
      type: 'image',
      mime_type: mimeType,
      data: base64Data,
    }));
  }

  disconnect() {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }

  isConnected() {
    return this.websocket && this.websocket.readyState === WebSocket.OPEN;
  }
}
