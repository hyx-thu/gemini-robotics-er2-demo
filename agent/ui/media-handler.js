/**
 * MediaHandler: Manages Audio/Video capture and playback
 */
class MediaHandler {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.audioWorkletNode = null;
    this.videoStream = null;
    this.videoInterval = null;
    this.nextStartTime = 0;
    this.scheduledSources = [];
    this.isRecording = false;
    this.onBargeIn = null;
    this._loudFrames = 0;
    this.videoCanvas = document.createElement('canvas');
    this.canvasCtx = this.videoCanvas.getContext('2d');
  }

  async initializeAudio() {
    if (!this.audioContext) {
      this.audioContext =
          new (window.AudioContext || window.webkitAudioContext)();
      await this.audioContext.audioWorklet.addModule(
          '/static/pcm-processor.js');
    }
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  async startAudio(onAudioData) {
    await this.initializeAudio();
    // Never run two capture pipelines at once (each would stream its own
    // chunks and the server would receive N× real-time audio).
    if (this.audioWorkletNode || this.mediaStream) {
      this.stopAudio();
    }

    try {
      // TTS is played through this same page, so ask the browser for echo
      // cancellation explicitly: without it the mic re-captures the
      // speaker output and the server VAD treats it as the user barging in.
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
      const source =
          this.audioContext.createMediaStreamSource(this.mediaStream);
      const node = new AudioWorkletNode(this.audioContext, 'pcm-processor');
      this.audioWorkletNode = node;
      this.audioSource = source;

      node.port.onmessage = (event) => {
        // Only the CURRENT pipeline may forward audio; a stale node from a
        // previous session must stay silent even after isRecording flips
        // back to true for the new one.
        if (!this.isRecording || this.audioWorkletNode !== node) return;
        {
          const downsampled = this.downsampleBuffer(
              event.data, this.audioContext.sampleRate, 16000);
          // Local barge-in detection: if TTS is currently playing and the
          // echo-cancelled mic signal shows sustained user speech energy,
          // cut off TTS playback immediately (<100ms) and notify the server.
          if (this.scheduledSources.length > 0 && downsampled.length > 0) {
            let sumSq = 0;
            for (let i = 0; i < downsampled.length; i++) {
              sumSq += downsampled[i] * downsampled[i];
            }
            const rms = Math.sqrt(sumSq / downsampled.length);
            if (rms > 0.085) {
              this._loudFrames += 1;
              if (this._loudFrames >= 4) {
                this._loudFrames = 0;
                this.stopAudioPlayback();
                if (typeof this.onBargeIn === 'function') {
                  this.onBargeIn();
                }
              }
            } else {
              this._loudFrames = 0;
            }
          } else {
            this._loudFrames = 0;
          }
          const pcm16 = this.convertFloat32ToInt16(downsampled);
          onAudioData(pcm16);
        }
      };

      source.connect(node);
      // Mute local feedback
      const muteGain = this.audioContext.createGain();
      muteGain.gain.value = 0;
      node.connect(muteGain);
      muteGain.connect(this.audioContext.destination);
      this.audioMuteGain = muteGain;

      this.isRecording = true;
    } catch (e) {
      console.error('Error starting audio:', e);
      throw e;
    }
  }

  stopAudio() {
    this.isRecording = false;
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
      this.mediaStream = null;
    }
    if (this.audioSource) {
      try { this.audioSource.disconnect(); } catch (e) {}
      this.audioSource = null;
    }
    if (this.audioWorkletNode) {
      const node = this.audioWorkletNode;
      node.port.onmessage = null;
      try { node.port.close(); } catch (e) {}
      try { node.disconnect(); } catch (e) {}
      this.audioWorkletNode = null;
    }
    if (this.audioMuteGain) {
      try { this.audioMuteGain.disconnect(); } catch (e) {}
      this.audioMuteGain = null;
    }
  }

  async startVideo(videoElement, onFrame) {
    try {
      this.videoStream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });
      videoElement.srcObject = this.videoStream;

      this.videoInterval = setInterval(() => {
        this.captureFrame(videoElement, onFrame);
      }, 1000);  // 1 FPS
    } catch (e) {
      console.error('Error starting video:', e);
      throw e;
    }
  }

  /**
   * Start video and also upload frames to the robot backend at 5 FPS.
   * This feeds the fake backend so the robot camera pipeline works.
   */
  async startVideoWithRobotFeed(videoElement, onFrame) {
    try {
      this.videoStream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });
      videoElement.srcObject = this.videoStream;

      // 1 FPS for Gemini direct image input
      this.videoInterval = setInterval(() => {
        this.captureFrame(videoElement, onFrame);
      }, 1000);

      // 5 FPS upload to fake backend via proxy
      const uploadCanvas = document.createElement('canvas');
      uploadCanvas.width = 384;
      uploadCanvas.height = 384;
      const uploadCtx = uploadCanvas.getContext('2d');

      this._robotFeedInterval = setInterval(() => {
        if (!this.videoStream) return;
        uploadCtx.drawImage(videoElement, 0, 0, 384, 384);
        uploadCanvas.toBlob(async (blob) => {
          if (!blob) return;
          try {
            await fetch('/api/upload_frame', {
              method: 'POST',
              headers: {'Content-Type': 'image/jpeg'},
              body: blob,
            });
          } catch (e) {
            // silent — backend may not be ready yet
          }
        }, 'image/jpeg', 0.8);
      }, 200);  // 5 FPS

    } catch (e) {
      console.error('Error starting video:', e);
      throw e;
    }
  }

  async startScreen(videoElement, onFrame, onEnded) {
    try {
      this.videoStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
      });
      videoElement.srcObject = this.videoStream;

      // Handle stream ending (e.g. user clicks "Stop sharing" in browser UI)
      this.videoStream.getVideoTracks()[0].onended = () => {
        this.stopVideo(videoElement);
        if (onEnded) onEnded();
      };

      this.videoInterval = setInterval(() => {
        this.captureFrame(videoElement, onFrame);
      }, 1000);  // 1 FPS
    } catch (e) {
      console.error('Error starting screen share:', e);
      throw e;
    }
  }

  stopVideo(videoElement) {
    if (this.videoStream) {
      this.videoStream.getTracks().forEach((t) => t.stop());
      this.videoStream = null;
    }
    if (this.videoInterval) {
      clearInterval(this.videoInterval);
      this.videoInterval = null;
    }
    if (this._robotFeedInterval) {
      clearInterval(this._robotFeedInterval);
      this._robotFeedInterval = null;
    }
    if (videoElement) {
      videoElement.srcObject = null;
    }
  }

  captureFrame(videoElement, onFrame) {
    if (!this.videoStream) return;
    // Keep the source aspect ratio (640 px wide) so the model sees an
    // undistorted frame and its normalized 0-1000 coordinates line up with
    // the preview; fall back to 4:3 before metadata is available.
    const vw = videoElement.videoWidth || 640;
    const vh = videoElement.videoHeight || 480;
    const w = 640;
    const h = Math.max(1, Math.round((w * vh) / vw));
    if (this.videoCanvas.width !== w || this.videoCanvas.height !== h) {
      this.videoCanvas.width = w;
      this.videoCanvas.height = h;
    }
    this.canvasCtx.drawImage(videoElement, 0, 0, w, h);
    const base64 = this.videoCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];
    onFrame(base64);
  }

  playAudio(arrayBuffer) {
    if (!this.audioContext) return;
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }

    let bufferToUse = arrayBuffer;
    if (arrayBuffer.byteLength % 2 !== 0) {
      console.warn(`Audio buffer length is odd (${
          arrayBuffer.byteLength}), slicing to even length.`);
      bufferToUse = arrayBuffer.slice(0, arrayBuffer.byteLength - 1);
    }
    const pcmData = new Int16Array(bufferToUse);
    const float32Data = new Float32Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      float32Data[i] = pcmData[i] / 32768.0;
    }

    const buffer = this.audioContext.createBuffer(1, float32Data.length, 24000);
    buffer.getChannelData(0).set(float32Data);

    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);

    const now = this.audioContext.currentTime;
    this.nextStartTime = Math.max(now, this.nextStartTime);
    source.start(this.nextStartTime);
    this.nextStartTime += buffer.duration;

    this.scheduledSources.push(source);
    source.onended = () => {
      const idx = this.scheduledSources.indexOf(source);
      if (idx > -1) this.scheduledSources.splice(idx, 1);
    };
  }

  stopAudioPlayback() {
    this.scheduledSources.forEach((s) => {
      try {
        s.stop();
      } catch (e) {
      }
    });
    this.scheduledSources = [];
    if (this.audioContext) {
      this.nextStartTime = this.audioContext.currentTime;
    }
  }

  // Utils
  downsampleBuffer(buffer, sampleRate, outSampleRate) {
    if (outSampleRate === sampleRate) return buffer;
    const ratio = sampleRate / outSampleRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
      let accum = 0, count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length;
           i++) {
        accum += buffer[i];
        count++;
      }
      result[offsetResult] = accum / count;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }

  convertFloat32ToInt16(buffer) {
    let l = buffer.length;
    const buf = new Int16Array(l);
    while (l--) {
      buf[l] = Math.min(1, Math.max(-1, buffer[l])) * 0x7fff;
    }
    return buf.buffer;
  }
}
