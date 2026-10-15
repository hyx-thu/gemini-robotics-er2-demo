// --- Main Application Logic (Proactive Agent) ---
let localStorage = window.localStorage;

let statusLabel = document.getElementById('status-label');
let micBtn = document.getElementById('micBtn');
let textInput = document.getElementById('textInput');
let sendBtn = document.getElementById('sendBtn');
let videoPreview = document.getElementById('video-preview');
let videoPlaceholder = document.getElementById('video-placeholder');
let robotCamImg = document.getElementById('robot-cam-img');
let connectBtn = document.getElementById('connectBtn');
let chatLog = document.getElementById('chat-log');
let overlayCanvas = document.getElementById('overlay-canvas');
let chatToggle = document.getElementById('chatToggle');
let chatSidebar = document.getElementById('chat-sidebar');

// --- Model Selector & Runtime Configs ---
let modelSelectorBtn = document.getElementById('modelSelectorBtn');
let modelSelectorLabel = document.getElementById('modelSelectorLabel');
let modelDropdown = document.getElementById('modelDropdown');
let customModelModal = document.getElementById('custom-model-modal');
let customModelInput = document.getElementById('custom-model-input');
let customModalitySelect = document.getElementById('custom-modality-select');
let customModelCancel = document.getElementById('custom-model-cancel');
let customModelApply = document.getElementById('custom-model-apply');

// Endpoint Selector DOM
let endpointSelectorBtn = document.getElementById('endpointSelectorBtn');
let endpointSelectorLabel = document.getElementById('endpointSelectorLabel');
let endpointDropdown = document.getElementById('endpointDropdown');

// Audio Toggle DOM
let audioToggleBtn = document.getElementById('audioToggleBtn');

// Agent Selector DOM
let agentSelectorBtn = document.getElementById('agentSelectorBtn');
let agentSelectorLabel = document.getElementById('agentSelectorLabel');
let agentDropdown = document.getElementById('agentDropdown');

const ENDPOINT_MODELS = {
  gemini_live_api: [
    { model: 'models/gemini-robotics-er-2-streaming-preview', label: 'models/gemini-robotics-er-2-streaming-preview', modality: 'TEXT', textOnly: true },
    { model: 'models/gemini-3.8-live', label: 'models/gemini-3.8-live', modality: 'AUDIO', textOnly: false },
    { model: 'models/gemini-3.8-live-extended-thinking', label: 'models/gemini-3.8-live-extended-thinking', modality: 'AUDIO', textOnly: false },
    // Cascade: Gemini ASR -> gemini-4-argon (generateContent) -> server TTS.
    { model: 'models/gemini-4-argon', label: 'gemini-4-argon (ASR + TTS cascade)', modality: 'TEXT', textOnly: true },
    // Gemini 3.8 Live for voice + light tools; ER2 (non-streaming) plans robot tasks.
    { model: 'models/gemini-3.8-live+er2', label: 'gemini-3.8-live + ER2 planner', modality: 'AUDIO', textOnly: false }
  ]
};

let selectedEndpointType =
  localStorage.getItem('lite_endpoint_type') || 'gemini_live_api';

let selectedModel =
  localStorage.getItem('lite_model') ||
  ENDPOINT_MODELS[selectedEndpointType][0].model;

let selectedAudioEnabled = localStorage.getItem('lite_audio_enabled') !==
  'false';  // speaker defaults to ON
// Mic defaults to ON (auto-starts on connect). Persisted when the user
// manually toggles it so their choice sticks across reloads.
let micAutoStart = localStorage.getItem('lite_mic_auto') !== 'false';
// Per-model thinking capability from /api/server_defaults:
//   { "<model>": { default: "low", supported: true }, ... }
let modelThinkingInfo = {};

// Keep the Thinking Level dropdown consistent with the selected model:
// reset to the model's default and disable it for models that reject
// thinkingConfig (otherwise the Live API closes the socket with 1007).
function modelKey(model) {
  return (model || selectedModel || '').replace(/^models\//, '');
}

// User's chosen thinking level, remembered per model across reloads.
function savedThinkingLevel(model) {
  return localStorage.getItem('lite_thinking_level:' + modelKey(model));
}
function saveThinkingLevel(model, level) {
  localStorage.setItem('lite_thinking_level:' + modelKey(model), level);
}

// The level that will be sent on the next Connect for the given model:
// unsupported → 'none'; else user's saved choice; else model default.
function pendingThinkingLevel(model) {
  const info = modelThinkingInfo[modelKey(model)];
  if (info && info.supported === false) return 'none';
  return savedThinkingLevel(model) || (info && info.default) || 'low';
}

// Level the *current* live session was started with (null when offline).
let activeThinkingLevel = null;

function syncThinkingLevelForModel(model) {
  const sel = document.getElementById('config-thinking-level');
  if (sel) {
    const info = modelThinkingInfo[modelKey(model)];
    const help = sel.parentElement && sel.parentElement.previousElementSibling;
    if (info && info.supported === false) {
      sel.value = 'none';
      sel.disabled = true;
      sel.title = 'This model does not support thinking levels';
      if (help) help.textContent = 'Not supported by this model (always off).';
    } else {
      sel.disabled = false;
      sel.title = '';
      sel.value = pendingThinkingLevel(model);
      if (help) help.textContent =
        'Controls how much the model thinks before responding. ' +
        'Saved immediately; takes effect on the next Connect.';
    }
  }
  renderThinkingStatus();
}

// Footer status: 🧠 <active level>, with "→ <pending>*" when the user has
// picked a different level that needs a reconnect to apply.
function renderThinkingStatus() {
  const el = document.getElementById('think-val');
  if (!el) return;
  const pending = pendingThinkingLevel(selectedModel).toUpperCase();
  const active = activeThinkingLevel ? activeThinkingLevel.toUpperCase() : null;
  if (!active) {
    el.textContent = pending;
    el.title = 'Thinking level for the next session';
    el.classList.remove('stale');
  } else if (active === pending) {
    el.textContent = active;
    el.title = 'Thinking level of the current session';
    el.classList.remove('stale');
  } else {
    el.textContent = active + ' \u2192 ' + pending + '*';
    el.title = 'Session is running at ' + active + '; reconnect to apply ' + pending;
    el.classList.add('stale');
  }
}
let customBaseModality =
  localStorage.getItem('lite_custom_modality') || 'TEXT';

// Re-validate selectedModel against available options for the selected endpoint
let availableModels = ENDPOINT_MODELS[selectedEndpointType] || [];
let modelExists = availableModels.some(m => m.model === selectedModel) || selectedModel === 'custom';
if (!modelExists && availableModels.length > 0) {
  selectedModel = availableModels[0].model;
  localStorage.setItem('lite_model', selectedModel);
}

// Populate models dropdown dynamically
function populateModels(endpoint) {
  if (!modelDropdown) return;
  modelDropdown.innerHTML = '';
  const models = ENDPOINT_MODELS[endpoint] || [];
  models.forEach(m => {
    const opt = document.createElement('div');
    opt.className = 'model-option';
    if (m.model === selectedModel) {
      opt.classList.add('active');
      if (modelSelectorLabel) modelSelectorLabel.textContent = m.label;
    }
    opt.dataset.model = m.model;
    opt.dataset.modality = m.modality;
    opt.dataset.textOnly = m.textOnly ? 'true' : 'false';
    opt.textContent = m.label;
    bindModelOption(opt);
    modelDropdown.appendChild(opt);
  });

  // Add Custom Option
  const customOpt = document.createElement('div');
  customOpt.className = 'model-option model-option-custom';
  if (selectedModel === 'custom') {
    customOpt.classList.add('active');
  }
  customOpt.dataset.model = 'custom';
  customOpt.textContent = 'Custom...';
  bindModelOption(customOpt);
  modelDropdown.appendChild(customOpt);
}

// Binds the click handler to a model option element.
function bindModelOption(opt) {
  opt.onclick = () => {
    const model = opt.dataset.model;
    if (model === 'custom') {
      modelDropdown.classList.add('hidden');
      if (customModelInput) customModelInput.value = selectedModel;
      if (customModalitySelect) customModalitySelect.value = customBaseModality;
      customModelModal.classList.remove('hidden');
      if (customModelInput) customModelInput.focus();
      return;
    }
    selectedModel = model;
    localStorage.setItem('lite_model', selectedModel);
    modelSelectorLabel.textContent = opt.textContent;

    modelDropdown.querySelectorAll('.model-option')
      .forEach(o => o.classList.remove('active'));
    opt.classList.add('active');
    modelDropdown.classList.add('hidden');
    updateAudioUI();
    syncThinkingLevelForModel(selectedModel);
  };
}

// Initialize endpoint and model selectors
if (endpointSelectorLabel) {
  if (endpointDropdown) {
    const activeOpt = Array.from(endpointDropdown.querySelectorAll('.model-option'))
      .find(opt => opt.dataset.endpoint === selectedEndpointType);
    if (activeOpt) {
      endpointSelectorLabel.textContent = activeOpt.textContent;
      endpointDropdown.querySelectorAll('.model-option').forEach(o => {
        o.classList.toggle('active', o.dataset.endpoint === selectedEndpointType);
      });
    }
  }
}

populateModels(selectedEndpointType);

// Toggle endpoint dropdown
if (endpointSelectorBtn) {
  endpointSelectorBtn.onclick = (e) => {
    e.stopPropagation();
    endpointDropdown.classList.toggle('hidden');
  };
}

// Toggle model dropdown
if (modelSelectorBtn) {
  modelSelectorBtn.onclick = (e) => {
    e.stopPropagation();
    modelDropdown.classList.toggle('hidden');
  };
}

// Handle endpoint selection
if (endpointDropdown) {
  endpointDropdown.querySelectorAll('.model-option').forEach(opt => {
    opt.onclick = () => {
      const ep = opt.dataset.endpoint;
      selectedEndpointType = ep;
      localStorage.setItem('lite_endpoint_type', ep);
      if (endpointSelectorLabel) {
        endpointSelectorLabel.textContent = opt.textContent;
      }
      endpointDropdown.querySelectorAll('.model-option').forEach(o => o.classList.remove('active'));
      opt.classList.add('active');
      endpointDropdown.classList.add('hidden');

      // Refresh model dropdown
      populateModels(ep);
      // Select the first model as default
      const defaultModel = ENDPOINT_MODELS[ep][0].model;
      selectModelVisual(defaultModel);
    };
  });
}

// Close dropdowns on outside click
document.addEventListener('click', (e) => {
  if (modelDropdown && !modelDropdown.contains(e.target) && e.target !== modelSelectorBtn) {
    modelDropdown.classList.add('hidden');
  }
  if (endpointDropdown && !endpointDropdown.contains(e.target) && e.target !== endpointSelectorBtn) {
    endpointDropdown.classList.add('hidden');
  }
});

// Custom model modal handlers
if (customModelCancel) {
  customModelCancel.onclick = () => customModelModal.classList.add('hidden');
}
if (customModelApply) {
  customModelApply.onclick = () => {
    const customName = customModelInput.value.trim();
    if (!customName) return;
    selectedModel = customName;
    localStorage.setItem('lite_model', selectedModel);
    modelSelectorLabel.textContent = selectedModel;
    if (modelDropdown) {
      modelDropdown.querySelectorAll('.model-option')
        .forEach(o => o.classList.remove('active'));
    }
    // Custom modality from select
    if (customModalitySelect) {
      customBaseModality = customModalitySelect.value;
      localStorage.setItem('lite_custom_modality', customBaseModality);
    }
    customModelModal.classList.add('hidden');
    updateAudioUI();
    syncThinkingLevelForModel(selectedModel);
  };
}

// Handle Audio Toggle button clicks
if (audioToggleBtn) {
  audioToggleBtn.onclick = () => {
    if (audioToggleBtn.disabled) return;
    selectedAudioEnabled = !selectedAudioEnabled;
    localStorage.setItem('lite_audio_enabled', selectedAudioEnabled);
    updateAudioUI();
  };
}

function updateAudioUI() {
  if (!audioToggleBtn) return;

  const ICON_AUDIO_ON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 010 14.14"/><path d="M15.54 8.46a5 5 0 010 7.07"/></svg>';
  const ICON_AUDIO_OFF = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>';

  // Resolve base modality of selected model
  let baseModality = 'TEXT';
  if (selectedModel === 'custom') {
    baseModality = customBaseModality;
  } else if (modelDropdown) {
    const selectedOpt =
      Array.from(modelDropdown.querySelectorAll('.model-option'))
        .find(opt => opt.dataset.model === selectedModel);
    if (selectedOpt) {
      baseModality = selectedOpt.dataset.modality || 'TEXT';
    }
  }

  if (baseModality === 'AUDIO') {
    // Natively AUDIO model -> force Speaker to ON and disable it
    audioToggleBtn.disabled = true;
    audioToggleBtn.classList.add('active');
    audioToggleBtn.innerHTML = ICON_AUDIO_ON;
    audioToggleBtn.title = 'Audio is required for this model';
  } else {
    // Natively TEXT model -> restore user's toggle state and enable it
    audioToggleBtn.disabled = false;
    audioToggleBtn.classList.toggle('active', selectedAudioEnabled);
    audioToggleBtn.innerHTML = selectedAudioEnabled ? ICON_AUDIO_ON : ICON_AUDIO_OFF;
    audioToggleBtn.title =
      selectedAudioEnabled ? 'Click to mute audio' : 'Click to unmute audio';
  }
}

// Run initial update
updateAudioUI();
// (updateMicUI() runs after geminiClient is constructed — see below.)

function selectModelVisual(model) {
  selectedModel = model;
  localStorage.setItem('lite_model', selectedModel);

  if (modelDropdown) {
    let activeText = selectedModel;
    modelDropdown.querySelectorAll('.model-option')
      .forEach(o => {
        const isActive = o.dataset.model === selectedModel;
        o.classList.toggle('active', isActive);
        if (isActive) activeText = o.textContent;
      });
    if (modelSelectorLabel) modelSelectorLabel.textContent = activeText;
  } else if (modelSelectorLabel) {
    modelSelectorLabel.textContent = selectedModel;
  }
  updateAudioUI();
  syncThinkingLevelForModel(selectedModel);
}



// Helper to dynamically resolve session configs — always Live API
function getResolvedSessionConfig() {
  let baseModality = 'TEXT';
  if (selectedModel === 'custom') {
    baseModality = customBaseModality;
  } else if (modelDropdown) {
    const selectedOpt =
      Array.from(modelDropdown.querySelectorAll('.model-option'))
        .find(opt => opt.dataset.model === selectedModel);
    if (selectedOpt) {
      baseModality = selectedOpt.dataset.modality || 'TEXT';
    }
  }

  let resolvedModality = 'TEXT';
  let resolvedUseTts = false;

  if (baseModality === 'AUDIO') {
    resolvedModality = 'AUDIO';
    resolvedUseTts = selectedAudioEnabled;
  } else {
    resolvedModality = 'TEXT';
    resolvedUseTts = selectedAudioEnabled;
  }

  return {
    model: selectedModel,
    response_modality: resolvedModality,
    use_tts: resolvedUseTts,
    endpoint_type: selectedEndpointType
  };
}

// Fetch server defaults if not in localStorage

// Browser key for the Maps Embed API (restricted to Embed + our domains).
var mapsEmbedKey = '';

async function initServerDefaults() {
  try {
    const resp = await fetch('/api/server_defaults');
    if (!resp.ok) throw new Error();
    const defaults = await resp.json();
    if (defaults.thinking) modelThinkingInfo = defaults.thinking;
    mapsEmbedKey = defaults.maps_embed_key || '';
    recordingAvailable = !!defaults.recording_available;
    updateRecordBtn();
    if (defaults.video_sources) {
      agentVideoSources = Object.assign({}, agentVideoSources, defaults.video_sources);
      if (defaults.spot_simulated) {
        const spotOpt = agentDropdown &&
          Array.from(agentDropdown.querySelectorAll('.agent-option'))
            .find(o => o.dataset.agent === 'spot');
        if (spotOpt && !/sim/i.test(spotOpt.textContent)) {
          spotOpt.textContent = spotOpt.textContent.trim() + ' (webcam sim)';
        }
      }
    }
    // Server-chosen default agent (DEFAULT_AGENT env, "spot" by default).
    // Only applied before a session starts so a live run is never switched.
    if (defaults.default_agent && typeof selectedAgentName !== 'undefined' &&
        !(typeof geminiClient !== 'undefined' && geminiClient.isConnected())) {
      selectAgent(defaults.default_agent);
    }
    // Dynamically rebuild the model dropdown based on available server models
    if (defaults.available_models && modelDropdown) {
      modelDropdown.innerHTML = '';
      const allKnown = Object.values(ENDPOINT_MODELS).flat();
      defaults.available_models.forEach(modelName => {
        const found = allKnown.find(m => m.model === modelName);
        const label = found ? found.label : modelName;
        const modality = found ?
          found.modality :
          (modelName.includes('audio') || modelName.includes('live') ? 'AUDIO' : 'TEXT');
        const textOnly = found ? found.textOnly : false;

        const opt = document.createElement('div');
        opt.className = 'model-option';
        opt.dataset.model = modelName;
        opt.dataset.modality = modality;
        opt.dataset.textOnly = textOnly ? 'true' : 'false';
        opt.textContent = label;
        bindModelOption(opt);
        modelDropdown.appendChild(opt);
      });
      // Add custom option back
      const customOpt = document.createElement('div');
      customOpt.className = 'model-option model-option-custom';
      customOpt.dataset.model = 'custom';
      customOpt.innerHTML = 'Custom&hellip;';
      bindModelOption(customOpt);
      modelDropdown.appendChild(customOpt);
    }

    // Always fetch server default model if nothing explicitly preferred by user
    // or if the preferred model is not available
    const localModel = localStorage.getItem('lite_model');
    if (!localModel ||
      (defaults.available_models &&
        !defaults.available_models.includes(localModel))) {
      selectedModel = defaults.model;
      localStorage.setItem('lite_model', selectedModel);
    } else {
      selectedModel = localModel;
    }

    // Speaker defaults to ON on first visit (server use_tts only turns it
    // off when TTS is disabled server-side, since audio can't play anyway).
    if (!localStorage.getItem('lite_audio_enabled_set')) {
      selectedAudioEnabled = defaults.use_tts !== false;
      localStorage.setItem('lite_audio_enabled', selectedAudioEnabled);
      localStorage.setItem('lite_audio_enabled_set', 'true');
    }

    if (modelDropdown) {
      let activeText = selectedModel;
      modelDropdown.querySelectorAll('.model-option').forEach(opt => {
        const isActive = opt.dataset.model === selectedModel;
        opt.classList.toggle('active', isActive);
        if (isActive) activeText = opt.textContent;
      });
      if (modelSelectorLabel) modelSelectorLabel.textContent = activeText;
    }
    updateAudioUI();
    // Thinking info is now known: paint dropdown + footer for the selected model.
    syncThinkingLevelForModel(selectedModel);
  } catch (e) {
    console.warn('Failed to fetch server defaults, using local defaults');
  }
}
initServerDefaults();

let currentGeminiMessageDiv = null;
let currentUserMessageDiv = null;
let currentThinkingContentDiv = null;
let videoSource = 'computer';  // "computer" or "robot"
let toolCallCounter = 0;
let lastAckRow = null;  // Reference to the current aggregated ack heartbeat row
let ackCount = 0;       // Number of consecutive ack calls in the current row

// --- Latency Tracking ---
let MAX_HISTORY = 50;  // Sliding window for CI computation
let querySendTime = null;
let querySource = null;    // 'user' (text/voice) | 'heartbeat'
let ttftRecorded = false;
let ttfcRecorded = false;

// Client-measured latency histories (sliding window). Each sample is
// { v: milliseconds, src: 'user' | 'heartbeat' } so the bar can show either
// user-initiated turns only or everything (user + heartbeat).
let ttftHistory = [];  // Time to First Token (any model response)
let ttfcHistory = [];  // Time to First Function Call (tool invocation)
let ttltHistory = [];  // Time to Last Token (turn_complete)

// Latency scope shown in the footer bar. 'user' = only turns started by a
// user utterance / typed text; 'all' = user + heartbeat turns.
const LATENCY_SCOPES = ['user', 'all'];
let latencyScope = localStorage.getItem('lite_latency_scope') || 'user';
if (!LATENCY_SCOPES.includes(latencyScope)) latencyScope = 'user';

function scopedValues(history) {
  return history
    .filter(x => latencyScope === 'all' || x.src === 'user')
    .map(x => x.v);
}

function renderLatencyMetric(prefix, history) {
  const ci = computeCI(scopedValues(history));
  const el = document.getElementById(prefix + '-val');
  const countEl = document.getElementById(prefix + '-count');
  if (el) el.textContent = formatCI(ci);
  if (countEl) countEl.textContent = ci ? 'n=' + ci.n : '';
}

function renderLatencyBar() {
  renderLatencyMetric('ttft', ttftHistory);
  renderLatencyMetric('ttfc', ttfcHistory);
  renderLatencyMetric('ttlt', ttltHistory);
  const btn = document.getElementById('latencyScopeBtn');
  if (btn) {
    const isUser = latencyScope === 'user';
    btn.textContent = isUser ? '\u{1F464} USER' : '\u2211 ALL';
    btn.classList.toggle('scope-user', isUser);
    btn.classList.toggle('scope-all', !isUser);
    btn.title = isUser ?
      'Latency of user-initiated turns only (auto-rotates every 5 s; click to switch now)' :
      'Latency of all turns incl. heartbeats (auto-rotates every 5 s; click to switch now)';
    // Colour the TTFT/TTFC/TTLT values to match the active scope.
    const bar = btn.closest('.latency-bar');
    if (bar) {
      bar.classList.toggle('scope-user', isUser);
      bar.classList.toggle('scope-all', !isUser);
    }
  }
}

function cycleLatencyScope() {
  const i = LATENCY_SCOPES.indexOf(latencyScope);
  latencyScope = LATENCY_SCOPES[(i + 1) % LATENCY_SCOPES.length];
  localStorage.setItem('lite_latency_scope', latencyScope);
  renderLatencyBar();
}

// Auto-rotate the latency scope (USER <-> ALL) so both views are visible
// without clicking. A manual click switches immediately and restarts the
// rotation timer so the chosen view stays up for a full period.
const LATENCY_SCOPE_ROTATE_MS = 5000;
let latencyScopeTimer = null;
function startLatencyScopeRotation() {
  if (latencyScopeTimer) clearInterval(latencyScopeTimer);
  latencyScopeTimer = setInterval(cycleLatencyScope, LATENCY_SCOPE_ROTATE_MS);
}
function onLatencyScopeClick() {
  cycleLatencyScope();
  startLatencyScopeRotation();
}
// Server-reported latency breakdown histories (milliseconds).
let srvClientHistory = [];
let srvServerHistory = [];
let srvPrefillHistory = [];
let srvDecodeHistory = [];
let srvTtftHistory = [];
let srvClientTtftHistory = [];

function pushCapped(arr, val) {
  arr.push(val);
  if (arr.length > MAX_HISTORY) arr.shift();
}

function resetLatencyTimer(source = 'user') {
  querySendTime = performance.now();
  querySource = source;
  ttftRecorded = false;
  ttfcRecorded = false;
}

// Called on turn_complete: closes the current measurement window so the
// next turn (voice or heartbeat) starts a fresh T0 instead of inheriting
// a stale one.
function closeLatencyTurn() {
  querySendTime = null;
  querySource = null;
  ttftRecorded = false;
  ttfcRecorded = false;
}

function computeCI(values) {
  const n = values.length;
  if (n === 0) return null;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  if (n === 1) return { mean, lo: mean, hi: mean, n };
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1);
  const stderr = Math.sqrt(variance / n);
  const z = 1.96;  // 95% CI
  return { mean, lo: mean - z * stderr, hi: mean + z * stderr, n };
}

function formatCI(ci) {
  if (!ci) return '--';
  const m = (ci.mean / 1000).toFixed(2);
  if (ci.n === 1) return m + 's';
  const lo = (Math.max(0, ci.lo) / 1000).toFixed(2);
  const hi = (ci.hi / 1000).toFixed(2);
  return m + 's [' + lo + ', ' + hi + ']';
}

function formatCIms(ci) {
  if (!ci) return '--';
  const m = Math.round(ci.mean);
  if (ci.n === 1) return m + 'ms';
  const lo = Math.round(Math.max(0, ci.lo));
  const hi = Math.round(ci.hi);
  return m + 'ms [' + lo + ', ' + hi + ']';
}

function recordTTFT() {
  if (!ttftRecorded && querySendTime !== null) {
    pushCapped(ttftHistory, { v: performance.now() - querySendTime, src: querySource || 'user' });
    renderLatencyMetric('ttft', ttftHistory);
    ttftRecorded = true;
  }
}

function recordTTFC() {
  if (!ttfcRecorded && querySendTime !== null) {
    pushCapped(ttfcHistory, { v: performance.now() - querySendTime, src: querySource || 'user' });
    renderLatencyMetric('ttfc', ttfcHistory);
    ttfcRecorded = true;
  }
}

function recordTTLT() {
  // Only turns that produced a first token count; empty heartbeat turns
  // (model replies with nothing) would otherwise flood TTLT with tiny
  // samples and make it read lower than TTFT.
  if (querySendTime !== null && ttftRecorded) {
    pushCapped(ttltHistory, { v: performance.now() - querySendTime, src: querySource || 'user' });
    renderLatencyMetric('ttlt', ttltHistory);
  }
}

// Helper to update a metric's DOM elements.
function updateMetric(prefix, history) {
  const ci = computeCI(history);
  const el = document.getElementById(prefix + '-val');
  const countEl = document.getElementById(prefix + '-count');
  if (el) el.textContent = ci ? formatCIms(ci) : '--';
  if (countEl) countEl.textContent = ci ? 'n=' + ci.n : '';
}

function updateServerLatency(sl, serverTtftMs) {
  // Only use client TTFT if it was freshly recorded for the current turn.
  const clientTtftMs = ttftRecorded && ttftHistory.length > 0 ?
    ttftHistory[ttftHistory.length - 1] :
    null;

  // ① Client Overhead = Client TTFT - Server TTFT
  if (clientTtftMs !== null && serverTtftMs > 0) {
    const val = Math.max(0, clientTtftMs - serverTtftMs);
    pushCapped(srvClientHistory, val);
    updateMetric('srv-client', srvClientHistory);
  }

  // ② Server Overhead = Server TTFT - known inference
  if (serverTtftMs > 0 && sl.request_ttft_ms > 0) {
    const known = (sl.prefill_queue_ms || 0) + (sl.prefill_ms || 0) +
      (sl.decode_queue_ms || 0) + (sl.decode_ttft_ms || 0);
    const val = Math.max(0, serverTtftMs - known);
    pushCapped(srvServerHistory, val);
    updateMetric('srv-server', srvServerHistory);
  }

  // ③ Prefill
  if (sl.prefill_ms > 0) {
    pushCapped(srvPrefillHistory, sl.prefill_ms);
    updateMetric('srv-prefill', srvPrefillHistory);
  }

  // ④ Decode
  if (sl.decode_ttft_ms > 0) {
    pushCapped(srvDecodeHistory, sl.decode_ttft_ms);
    updateMetric('srv-decode', srvDecodeHistory);
  }

  // Reference: Srv TTFT
  if (sl.request_ttft_ms > 0) {
    pushCapped(srvTtftHistory, sl.request_ttft_ms);
    updateMetric('srv-ttft', srvTtftHistory);
  }

  // Reference: Client TTFT
  if (clientTtftMs !== null) {
    pushCapped(srvClientTtftHistory, clientTtftMs);
    updateMetric('srv-client-ttft', srvClientTtftHistory);
  }
}

function formatTokenCount(n) {
  if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
  return String(n);
}

function updateTokenUsage(tu) {
  if (!tu) return;

  // Footer bar: CTX utilization %, input, and cumulative output tokens
  const ctxEl = document.getElementById('tok-ctx-pct');
  const inputBarEl = document.getElementById('tok-input-bar');
  const outputBarEl = document.getElementById('tok-output-bar');
  if (ctxEl) {
    const pct = tu.context_window_utilization_pct;
    ctxEl.textContent = pct != null ? pct.toFixed(1) + '%' : '--';
    // Color-code: green < 50%, yellow 50-80%, red > 80%
    if (pct != null) {
      ctxEl.style.color = pct >= 80 ? '#e74c3c' : pct >= 50 ? '#f39c12' : '';
    }
  }
  if (inputBarEl) {
    inputBarEl.textContent = formatTokenCount(tu.prompt_token_count);
  }
  if (outputBarEl) {
    outputBarEl.textContent = formatTokenCount(tu.cumulative_output_tokens);
  }

  // Detail panel: per-field breakdown
  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };
  setEl('tok-prompt-val', formatTokenCount(tu.prompt_token_count));
  setEl('tok-response-val', formatTokenCount(tu.response_token_count));
  setEl('tok-thoughts-val', formatTokenCount(tu.thoughts_token_count));
  setEl(
    'tok-total-val',
    formatTokenCount(tu.total_token_count) + ' / ' +
    formatTokenCount(tu.context_window_limit));
  setEl('tok-cumulative-val', formatTokenCount(tu.cumulative_output_tokens));

  // Per-modality breakdown (compact format)
  const modality = tu.prompt_tokens_by_modality || {};
  const parts = Object.entries(modality)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => k + ':' + formatTokenCount(v));
  setEl('tok-modality-val', parts.length > 0 ? parts.join(' ') : '--');
}

let mediaHandler = new MediaHandler();
let audioFlushPending = false;
mediaHandler.onBargeIn = () => {
  if (geminiClient && geminiClient.isConnected()) {
    geminiClient.send(JSON.stringify({ type: 'barge_in' }));
  }
};
let _geoWatchId = null;
function _sendGeolocationFix(pos) {
  if (!geminiClient || !geminiClient.isConnected()) return;
  const c = pos.coords;
  geminiClient.send(JSON.stringify({
    type: 'geolocation',
    latitude: c.latitude,
    longitude: c.longitude,
    accuracy: c.accuracy,
    timestamp: Math.floor((pos.timestamp || Date.now()) / 1000),
  }));
}
function startGeolocationReporting() {
  if (!('geolocation' in navigator)) {
    if (geminiClient && geminiClient.isConnected()) {
      geminiClient.send(JSON.stringify(
          {type: 'geolocation', error: 'unsupported'}));
    }
    return;
  }
  const onErr = (err) => {
    console.warn('Geolocation unavailable:', err && err.message);
    if (geminiClient && geminiClient.isConnected()) {
      geminiClient.send(JSON.stringify({
        type: 'geolocation',
        error: (err && err.message) || 'unavailable',
      }));
    }
  };
  // Fast coarse fix first (cached ok), then a high-accuracy watch.
  navigator.geolocation.getCurrentPosition(_sendGeolocationFix, onErr, {
    enableHighAccuracy: false, timeout: 8000, maximumAge: 300000,
  });
  if (_geoWatchId !== null) navigator.geolocation.clearWatch(_geoWatchId);
  _geoWatchId = navigator.geolocation.watchPosition(
      _sendGeolocationFix, onErr,
      {enableHighAccuracy: true, timeout: 20000, maximumAge: 60000});
}
function stopGeolocationReporting() {
  if (_geoWatchId !== null && 'geolocation' in navigator) {
    navigator.geolocation.clearWatch(_geoWatchId);
  }
  _geoWatchId = null;
}
// ---- Opt-in session recording (test dataset) ----
let recordingAvailable = false;
let recordingOn = false;
const recordBtn = document.getElementById('recordBtn');
function updateRecordBtn(status) {
  if (!recordBtn) return;
  recordBtn.classList.toggle('hidden', !recordingAvailable);
  const connected = !!(typeof geminiClient !== 'undefined' && geminiClient && geminiClient.isConnected());
  recordBtn.disabled = !connected;
  recordBtn.classList.toggle('recording', recordingOn);
  recordBtn.title = recordingOn
    ? `Recording to ${status && status.path ? status.path : 'the private dataset bucket'} — click to stop`
    : 'Record this session (camera frames sent to the model, mic audio, events) to a private bucket for the test dataset';
}
if (recordBtn) {
  recordBtn.addEventListener('click', () => {
    if (!geminiClient || !geminiClient.isConnected()) return;
    geminiClient.send(JSON.stringify({ type: 'recording', enabled: !recordingOn }));
    recordBtn.disabled = true;
  });
}

let geminiClient = new GeminiClient({
  onOpen: () => {
    // utterance_id numbering restarts with every session.
    window._userTranscriptDivsByUtterance = new Map();
    window._lastUserTranscriptDiv = null;
    statusLabel.className = 'status-label connected';
    // Keep the power icon; the state is shown by colour only.
    connectBtn.classList.remove('connecting');
    connectBtn.classList.add('connected');
    connectBtn.title = 'Connected — click to disconnect';
    connectBtn.disabled = false;
    recordingOn = false;
    setTimeout(() => updateRecordBtn(), 0);
    if (modelSelectorBtn) modelSelectorBtn.disabled = true;
    if (audioToggleBtn) audioToggleBtn.disabled = true;
    if (agentSelectorBtn) agentSelectorBtn.disabled = true;

    // Apply the current video source mode
    applyVideoSource(videoSource);

    // Start session timer
    startSessionTimer();

    // Mic defaults to ON: start capture as soon as the session is up so the
    // user can just talk. Skipped if they explicitly turned the mic off.
    if (micAutoStart && !mediaHandler.isRecording) {
      micBtn.click();
    }

    // Device location for the get_current_location / google_maps_search
    // tools. Uses the browser Geolocation API (GPS on phones, Wi-Fi/IP on
    // laptops) and keeps a fresh fix for the whole session.
    startGeolocationReporting();
  },
  onMessage: (event) => {
    if (typeof event.data === 'string') {
      try {
        const msg = JSON.parse(event.data);
        handleJsonMessage(msg);
      } catch (e) {
        console.error('Parse error:', e);
      }
    } else {
      recordTTFT();
      if (audioFlushPending) {
        mediaHandler.stopAudioPlayback();
        audioFlushPending = false;
      }
      mediaHandler.playAudio(event.data);
    }
  },
  onClose: (e) => {
    recordingOn = false;
    setTimeout(() => updateRecordBtn(), 0);
    console.log('WS Closed:', e);
    activeThinkingLevel = null;
    renderThinkingStatus();
    statusLabel.className = 'status-label disconnected';
    statusLabel.textContent = 'Disconnected';
    stopSessionTimer();
    stopGeolocationReporting();
    resetUI();
  },
  onError: (e) => {
    console.error('WS Error:', e);
    statusLabel.className = 'status-label error';
    statusLabel.textContent = 'Error';
  },
});

// Initial mic button paint (needs geminiClient + mediaHandler to exist).
updateMicUI();

// --- Plan panel (driven by the update_plan tool) ---

let agentVideoSources = { human: 'computer', spot: 'spot', tinybot: 'tinybot' };

function videoSourceForAgent(agent) {
  return agentVideoSources[agent] || (agent === 'human' ? 'computer' : agent);
}

// Composite "Live + ER2" mode: make it obvious that the ER2 planner (not the
// Live voice model) plans, executes and tracks the robot task.
function renderEr2Status(msg) {
  const badge = document.getElementById('plan-agent');
  const model = (msg.model || 'ER2').replace(/^models\//, '');
  let text = '', line = '';
  if (msg.state === 'started') {
    text = '🤖 ER2 planning…';
    line = `🤖 ER2 planner (${model}) took over the task: “${msg.instruction || ''}”`;
  } else if (msg.state === 'executing') {
    text = '🤖 ER2 executing';
    line = `🤖 ER2 planned ${msg.steps || ''} steps and is executing / tracking them`;
  } else if (msg.state === 'finished') {
    const r = msg.result || {};
    const ok = r.status === 'done';
    text = ok ? '🤖 ER2 ✓ done' : `🤖 ER2 ✗ ${r.status || 'stopped'}`;
    line = `🤖 ER2 finished: ${r.status} (${r.steps_done}/${r.steps_total} steps, ${r.duration_s}s) → result handed to the Live model`;
  }
  if (badge && text) {
    badge.textContent = text;
    badge.className = 'plan-agent ' + (msg.state || '');
    badge.title = `Planned and tracked by ${model}; Gemini Live only talks to you`;
    const panel = document.getElementById('plan-panel');
    if (panel && msg.state === 'started') panel.classList.remove('hidden');
  }
  if (line) {
    const row = document.createElement('div');
    row.className = 'message er2-status ' + (msg.state || '');
    row.textContent = line;
    const isAtBottom = chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 50;
    chatLog.appendChild(row);
    if (isAtBottom) chatLog.scrollTop = chatLog.scrollHeight;
  }
}

function renderPlan(args) {
  const panel = document.getElementById('plan-panel');
  if (!panel || !args || !Array.isArray(args.steps) || !args.steps.length) return;
  const steps = args.steps.map(String);
  const status = String(args.status || 'in_progress');
  let current = parseInt(args.current_step, 10);
  if (Number.isNaN(current)) current = 0;
  current = Math.max(0, Math.min(current, steps.length - 1));
  const isDone = status === 'done';
  const isFailed = status === 'failed';
  const doneCount = isDone ? steps.length : current;

  panel.classList.remove('hidden');
  const statusEl = document.getElementById('plan-status');
  statusEl.textContent = status.replace('_', ' ');
  statusEl.className = 'plan-status ' + status;
  document.getElementById('plan-progress').textContent =
    isDone ? `${steps.length}/${steps.length} ✓` : `step ${current + 1}/${steps.length}`;
  document.getElementById('plan-goal').textContent = args.goal || '';
  const stopBtn = document.getElementById('plan-stop-btn');
  if (stopBtn) {
    stopBtn.style.display = (isDone || isFailed) ? 'none' : '';
    if (!stopBtn._wired) {
      stopBtn._wired = true;
      stopBtn.addEventListener('click', () => {
        if (geminiClient && geminiClient.isConnected()) {
          geminiClient.send(JSON.stringify({type: 'abort_task'}));
          stopBtn.disabled = true;
          setTimeout(() => { stopBtn.disabled = false; }, 1500);
        }
      });
    }
  }

  // Progress bar (done steps / total).
  const bar = document.getElementById('plan-progressbar-fill');
  if (bar) {
    bar.style.width = `${Math.round((doneCount / steps.length) * 100)}%`;
    bar.className = 'plan-progressbar-fill ' + status;
  }

  // Compact step list: one line per step, full text on hover.
  const ol = document.getElementById('plan-steps');
  ol.innerHTML = '';
  steps.forEach((text, i) => {
    const li = document.createElement('li');
    let cls, mark;
    if (isDone || i < current)      { cls = 'done';    mark = '✓'; }
    else if (i === current)         { cls = 'current'; mark = isFailed ? '✗' : '▶'; }
    else                            { cls = 'pending'; mark = String(i + 1); }
    if (isFailed && i === current) cls += ' failed';
    li.className = cls;
    li.title = `${i + 1}. ${text}`;
    li.innerHTML = `<span class="step-idx">${mark}</span><span class="step-text"></span>`;
    li.querySelector('.step-text').textContent = text;
    ol.appendChild(li);
  });
  const li = ol.children[current];
  if (li && li.scrollIntoView) li.scrollIntoView({ block: 'nearest' });

  const noteEl = document.getElementById('plan-note');
  if (args.note) { noteEl.textContent = '📝 ' + args.note; noteEl.classList.remove('hidden'); }
  else { noteEl.classList.add('hidden'); }

  renderPlanHud(steps, current, status);

  // Task over: the pick/place target marker and trajectory are no longer meaningful.
  if (isDone || isFailed) {
    clearOverlay('target');
    clearOverlay('trajectory');
  }
  // A finished plan dismisses a pending/confirmed watchdog row; an abort row
  // ("got no response after 3 attempts") stays next to the failed plan.
  if (isDone || (isFailed && actionStatus && actionStatus.state !== 'failed')) {
    renderActionStatus({ state: 'cleared' });
  }
}

// Mirror of the plan on top of the video: step dots + the current step text,
// so the user can follow progress even with the chat sidebar collapsed.
function renderPlanHud(steps, current, status) {
  const hud = document.getElementById('plan-hud');
  if (!hud) return;
  const isDone = status === 'done';
  const isFailed = status === 'failed';
  hud.classList.remove('hidden');
  hud.className = 'plan-hud ' + status;

  const dots = hud.querySelector('.hud-dots');
  dots.innerHTML = '';
  steps.forEach((text, i) => {
    const d = document.createElement('span');
    d.className = 'hud-dot ' +
      (isDone || i < current ? 'done' : i === current ? 'current' : 'pending');
    d.title = `${i + 1}. ${text}`;
    dots.appendChild(d);
  });

  const idx = hud.querySelector('.hud-idx');
  const txt = hud.querySelector('.hud-text');
  if (isDone) {
    idx.textContent = '✓';
    txt.textContent = `Done · ${steps.length} steps`;
  } else {
    idx.textContent = (isFailed ? '✗ ' : '▶ ') + `${current + 1}/${steps.length}`;
    txt.textContent = steps[current];
  }
  txt.title = steps[current] || '';
}

function resetPlan() {
  const panel = document.getElementById('plan-panel');
  if (panel) panel.classList.add('hidden');
  const hud = document.getElementById('plan-hud');
  if (hud) hud.classList.add('hidden');
  renderActionStatus({ state: 'cleared' });
  clearOverlay();
}

// --- Action watchdog status (command → visual confirmation → retry → abort) ---
//
// The server publishes `action_status` events whenever a supervised command
// (pick/place) is sent, times out, is re-sent, is confirmed from the camera,
// or exhausts its retries. We keep the last payload and tick the elapsed
// time locally so the countdown bar stays smooth between server updates.
let actionStatus = null;
let actionStatusAt = 0;
let actionStatusTicker = null;

function stopActionTicker() {
  if (actionStatusTicker) { clearInterval(actionStatusTicker); actionStatusTicker = null; }
}

function renderActionStatus(msg) {
  const row = document.getElementById('plan-watchdog');
  const hudRow = document.querySelector('#plan-hud .hud-watchdog');
  if (!row) return;
  const state = (msg && msg.state) || 'idle';
  window.dispatchEvent(new CustomEvent('robot-twin:status', { detail: msg || { state } }));
  if (state === 'cleared' || state === 'idle') {
    actionStatus = null;
    stopActionTicker();
    row.classList.add('hidden');
    if (hudRow) hudRow.classList.add('hidden');
    return;
  }
  actionStatus = msg;
  actionStatusAt = Date.now();
  // A command can be supervised before any plan exists; make the row visible.
  const panel = document.getElementById('plan-panel');
  if (panel) panel.classList.remove('hidden');
  row.classList.remove('hidden');
  paintActionStatus();
  if (state === 'pending' || state === 'retry' || state === 'timed_out') {
    if (!actionStatusTicker) actionStatusTicker = setInterval(paintActionStatus, 250);
  } else {
    stopActionTicker();
  }
  if (state === 'confirmed') {
    // Auto-dismiss the confirmation after a few seconds.
    setTimeout(() => { if (actionStatus === msg) renderActionStatus({ state: 'cleared' }); }, 6000);
  }
}

function paintActionStatus() {
  const s = actionStatus;
  if (!s) return;
  const row = document.getElementById('plan-watchdog');
  const since = (Date.now() - actionStatusAt) / 1000;
  const timeout = Number(s.timeout_s) || 30;
  const elapsed = Math.min(timeout, Math.max(0, (Number(s.elapsed_s) || 0) + since));
  const attempt = `${s.attempt || 1}/${s.max_attempts || 3}`;
  let icon, text, frac = elapsed / timeout;
  switch (s.state) {
    case 'pending':
    case 'retry':
      icon = s.state === 'retry' ? '🔁' : '⏳';
      text = `${s.action} sent · attempt ${attempt} · awaiting visual confirmation ` +
        `${elapsed.toFixed(1)}s / ${timeout}s` +
        (s.source === 'watchdog' ? ' · re-sent by watchdog' : '');
      break;
    case 'timed_out':
      icon = '⚠️'; frac = 1;
      text = `${s.action} attempt ${attempt} timed out · no visible change in ${timeout}s · ` +
        'waiting for the command to be re-issued';
      break;
    case 'confirmed':
      icon = '✅'; frac = 1;
      text = `${s.action} confirmed from camera (attempt ${attempt})` +
        (s.observation ? ` · ${s.observation}` : '');
      break;
    case 'failed':
      icon = '✗'; frac = 1;
      text = `${s.action} got no response after ${s.attempt || s.max_attempts || 3} attempts · task aborted`;
      break;
    default:
      icon = '•'; text = JSON.stringify(s);
  }
  row.className = `plan-watchdog ${s.state}`;
  row.querySelector('.wd-icon').textContent = icon;
  const textEl = row.querySelector('.wd-text');
  textEl.textContent = text;
  textEl.title = s.reason || text;
  row.querySelector('.wd-bar-fill').style.width = `${Math.round(frac * 100)}%`;

  const hud = document.getElementById('plan-hud');
  const hudRow = hud && hud.querySelector('.hud-watchdog');
  if (hudRow) {
    hudRow.textContent = `${icon} ${text}`;
    hudRow.className = `hud-watchdog ${s.state}`;
    hud.classList.remove('hidden');
  }
}

// --- Video Source Toggle ---

function applyVideoSource(source) {
  videoSource = source;

  // Tell the server which mode we're in
  if (geminiClient.isConnected()) {
    geminiClient.send(JSON.stringify({ type: 'video_source', source: source }));
  }

  const videoContainer = robotCamImg.parentElement;

  if (source === 'computer') {
    // Computer mode: webcam → Gemini directly
    robotCamImg.classList.add('hidden');
    robotCamImg.src = '';
    videoContainer.classList.remove('atari-aspect');
    videoContainer.classList.remove('spot-aspect');
    // Show the full, uncropped camera frame (what the model actually sees).
    syncContainerAspectToWebcam();

    // Start webcam
    videoPlaceholder.classList.add('hidden');
    videoPreview.classList.remove('hidden');
    if (!mediaHandler.videoStream) {
      mediaHandler
        .startVideo(
          videoPreview,
          (base64Data) => {
            if (geminiClient.isConnected()) {
              geminiClient.sendImage(base64Data);
            }
          })
        .catch(e => console.warn('Camera start failed:', e));
    }
  } else {
    // Robot / Atari mode: camera poller → Gemini
    mediaHandler.stopVideo(videoPreview);
    videoPreview.classList.add('hidden');
    videoPlaceholder.classList.add('hidden');
    videoContainer.style.aspectRatio = '';

    // Atari uses 8/5 aspect ratio
    if (source === 'atari') {
      videoContainer.classList.add('atari-aspect');
      videoContainer.classList.remove('spot-aspect');
    } else {
      videoContainer.classList.remove('atari-aspect');
      videoContainer.classList.toggle('spot-aspect', source === 'spot');
    }

    // Show robot cam (stitched MJPEG from camera poller)
    robotCamImg.classList.remove('hidden');
    robotCamImg.src = '/api/camera?' + Date.now();

    // Retry after a short delay
    if (geminiClient.isConnected()) {
      setTimeout(() => {
        robotCamImg.src = '/api/camera?' + Date.now();
      }, 2000);
    }
  }
}

// --- Agent Selector Click Bindings ---
if (agentSelectorBtn) {
  agentSelectorBtn.onclick = (e) => {
    e.stopPropagation();
    agentDropdown.classList.toggle('hidden');
  };
}

document.addEventListener('click', (e) => {
  if (agentDropdown && !agentDropdown.contains(e.target) &&
    e.target !== agentSelectorBtn) {
    agentDropdown.classList.add('hidden');
  }
});

if (agentDropdown) {
  agentDropdown.querySelectorAll('.agent-option').forEach(opt => {
    opt.onclick = () => {
      const agent = opt.dataset.agent;
      selectAgent(agent);

      applyVideoSource(videoSourceForAgent(agent));

      agentDropdown.classList.add('hidden');
    };
  });
}

function selectAgent(type) {
  if (typeof selectedAgentName !== 'undefined') {
    selectedAgentName = type;
  }
  // The 3D twin models a quadruped with an arm — only meaningful for Spot.
  const twin = document.getElementById('robot-twin');
  if (twin) twin.classList.toggle('hidden', type !== 'spot');

  // Clear active state from all agent dropdown options
  if (agentDropdown) {
    agentDropdown.querySelectorAll('.agent-option')
      .forEach(o => o.classList.remove('active'));
    const activeOpt =
      Array.from(agentDropdown.querySelectorAll('.agent-option'))
        .find(o => o.dataset.agent === type);
    if (activeOpt) {
      activeOpt.classList.add('active');
      if (agentSelectorLabel) {
        agentSelectorLabel.textContent = activeOpt.textContent;
      }
    }
  }
}

// --- Chat Sidebar Toggle ---
chatToggle.onclick = () => {
  chatSidebar.classList.toggle('collapsed');
};

// --- Message Handling ---

function handleJsonMessage(msg) {
  if (msg.type === 'heartbeat_sent') {
    resetLatencyTimer('heartbeat');
    return;
  }
  if (msg.type === 'stream_status') {
    handleStreamStatus(msg);
    return;
  }
  if (msg.type === 'error') {
    const text = typeof msg.error === 'string' ?
      msg.error : JSON.stringify(msg.error);
    appendMessage('system', `⚠️ ${text}`);
    return;
  }
  if (msg.type === 'interrupted') {
    mediaHandler.stopAudioPlayback();
    audioFlushPending = false;
    currentGeminiMessageDiv = null;
    currentUserMessageDiv = null;
    currentThinkingContentDiv = null;
  } else if (msg.type === 'turn_complete') {
    recordTTLT();
    closeLatencyTurn();
    currentGeminiMessageDiv = null;
    currentUserMessageDiv = null;
    currentThinkingContentDiv = null;
  } else if (msg.type === 'telemetry') {
    if (msg.server_latency || msg.server_ttft_ms) {
      updateServerLatency(
        msg.server_latency || {}, msg.server_ttft_ms || 0);
    }
    if (msg.token_usage) {
      updateTokenUsage(msg.token_usage);
    }

  } else if (msg.type === 'draw_points') {
    recordTTFT();
    drawPointsOverlay(msg.points, msg.layer || 'annotation', msg);
  } else if (msg.type === 'clear_overlay') {
    clearOverlay(msg.layer);
  } else if (msg.type === 'recording_status') {
    recordingOn = msg.state === 'on';
    updateRecordBtn(msg);
    const row = document.createElement('div');
    row.className = 'message er2-status';
    row.textContent = msg.state === 'on'
      ? `⏺ Recording this session → ${msg.path}`
      : msg.state === 'off'
        ? `⏹ Recording stopped${msg.frames !== undefined ? ` (${msg.frames} frames, ${msg.events} events)` : ''}`
        : `⏺ Recording unavailable: ${msg.reason || ''}`;
    chatLog.appendChild(row);
    chatLog.scrollTop = chatLog.scrollHeight;
  } else if (msg.type === 'er2_status') {
    renderEr2Status(msg);
  } else if (msg.type === 'action_status') {
    renderActionStatus(msg);
  } else if (msg.type === 'user' || msg.type === 'user_transcript') {
    // Only stop audio on typed 'user' messages. For voice input, barge-in is
    // already stopped at speech start via local VAD + 'interrupted' (ACTIVITY_START),
    // whereas 'user_transcript' arrives at ACTIVITY_END concurrently with the
    // model's first native audio chunks.
    if (msg.type === 'user') {
      mediaHandler.stopAudioPlayback();
      audioFlushPending = false;
    }
    if (!window._userTranscriptDivsByUtterance) {
      window._userTranscriptDivsByUtterance = new Map();
    }
    const uid = msg.utterance_id;
    // In-place update of an existing utterance bubble (e.g. ASR refinement or
    // multi-chunk CJK transcript after the model has already started replying).
    // utterance_id restarts at 1 for every session, so only reuse a bubble
    // that is still attached to the chat log (resetUI() clears old ones).
    const existingDiv =
      msg.type === 'user_transcript' && uid !== undefined ?
        window._userTranscriptDivsByUtterance.get(uid) : null;
    if (existingDiv && !existingDiv.isConnected) {
      window._userTranscriptDivsByUtterance.delete(uid);
    }
    if (existingDiv && existingDiv.isConnected) {
      const targetDiv = existingDiv;
      if (msg.replace) {
        targetDiv.textContent = msg.text;
      } else {
        const combined = (targetDiv.textContent || '') + (msg.text || '');
        targetDiv.textContent = combined.replace(
          /([\u4e00-\u9fff])\s+([\u4e00-\u9fff，。！？、；：])/g,
          '$1$2'
        );
      }
      const isAtBottomUser =
        chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 50;
      if (isAtBottomUser) chatLog.scrollTop = chatLog.scrollHeight;
      return;
    }
    if (
      msg.type === 'user_transcript' &&
      msg.replace &&
      window._lastUserTranscriptDiv &&
      window._lastUserTranscriptDiv.isConnected &&
      uid === undefined
    ) {
      window._lastUserTranscriptDiv.textContent = msg.text;
      return;
    }
    // User message breaks ack streak
    lastAckRow = null;
    ackCount = 0;
    currentThinkingContentDiv = null;
    currentGeminiMessageDiv = null;
    // Voice input: first transcript chunk of an utterance starts a new
    // user turn. Subsequent partial chunks of the same utterance don't.
    if (msg.type === 'user_transcript' && querySource !== 'user') {
      resetLatencyTimer('user');
    }
    const sameUtterance =
      uid === undefined || window._currentUserUtteranceId === uid;
    if (currentUserMessageDiv && !msg.replace && sameUtterance) {
      const combined = (currentUserMessageDiv.textContent || '') + (msg.text || '');
      currentUserMessageDiv.textContent = combined.replace(
        /([\u4e00-\u9fff])\s+([\u4e00-\u9fff，。！？、；：])/g,
        '$1$2'
      );
      const isAtBottomUser =
        chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 50;
      if (isAtBottomUser) chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentUserMessageDiv = appendMessage('user', msg.text);
      window._lastUserTranscriptDiv = currentUserMessageDiv;
      window._currentUserUtteranceId = uid;
      if (uid !== undefined) {
        window._userTranscriptDivsByUtterance.set(uid, currentUserMessageDiv);
      }
      chatLog.scrollTop = chatLog.scrollHeight;
    }
  } else if (msg.type === 'text_input') {
    if (typeof msg.text === 'string' &&
      msg.text.trimStart().startsWith('[HEARTBEAT]')) {
      return;
    }
    currentThinkingContentDiv = null;
    appendMessage('system', `📨 ${msg.text}`);
  } else if (msg.type === 'gemini' || msg.type === 'gemini_transcript') {
    // Gemini message breaks ack streak
    lastAckRow = null;
    ackCount = 0;
    currentThinkingContentDiv = null;
    currentUserMessageDiv = null;
    recordTTFT();
    if (currentGeminiMessageDiv) {
      currentGeminiMessageDiv.textContent += msg.text;
      const isAtBottomGemini =
        chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 50;
      if (isAtBottomGemini) chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentGeminiMessageDiv = appendMessage('gemini', msg.text);
    }
  } else if (msg.type === 'gemini_thought') {
    // Gemini thought message breaks ack streak
    lastAckRow = null;
    ackCount = 0;
    recordTTFT();
    currentGeminiMessageDiv = null;
    currentUserMessageDiv = null;
    if (currentThinkingContentDiv) {
      currentThinkingContentDiv.textContent += msg.text;
      const isAtBottom =
        chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 50;
      if (isAtBottom) chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentThinkingContentDiv = appendThoughtMessage(msg.text);
    }
  } else if (msg.type === 'tool_call') {
    currentThinkingContentDiv = null;
    recordTTFT();
    recordTTFC();
    toolCallCounter++;
    if (msg.name === 'ack') {
      // Aggregate consecutive ack calls into a single heartbeat row
      ackCount++;
      if (lastAckRow) {
        // Update existing heartbeat row
        const countSpan = lastAckRow.querySelector('.hb-count');
        if (countSpan) countSpan.textContent = `x${ackCount}`;
        // Scroll if at bottom
        const isAtBottom =
          chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight <
          50;
        if (isAtBottom) chatLog.scrollTop = chatLog.scrollHeight;
      } else {
        // Create new heartbeat row
        const row = document.createElement('div');
        row.className = 'message tool_call heartbeat-row';
        row.innerHTML = `<span class="hb-icon">❤️</span>` +
          `<span class="hb-count">x${ackCount}</span>`;
        const isAtBottom =
          chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight <
          50;
        chatLog.appendChild(row);
        if (isAtBottom) chatLog.scrollTop = chatLog.scrollHeight;
        lastAckRow = row;
      }
    } else {
      // Non-ack tool call — reset ack aggregation and active text bubbles
      lastAckRow = null;
      ackCount = 0;
      currentGeminiMessageDiv = null;
      currentUserMessageDiv = null;
      const desc = describeToolCall(msg);
      const result =
        msg.result !== undefined ? formatFunctionResult(msg.result) : '';

      if (result) {
        // Render as collapsible details
        const details = document.createElement('details');
        details.className = 'message tool_call_details';
        details.open = desc.open;

        const summary = document.createElement('summary');
        summary.className = 'tool_call_summary';
        summary.textContent = `${msg.agent === 'er2' ? '🤖 ER2 · ' : '🔧 '}[${toolCallCounter}] ${desc.summary}`;
        summary.title = `${msg.name}(${msg.args ? JSON.stringify(msg.args) : ''})`;
        details.appendChild(summary);

        const contentDiv = document.createElement('div');
        contentDiv.className = 'tool_response_content';
        contentDiv.textContent = (desc.argsText ? desc.argsText + '\n' : '') + result;
        details.appendChild(contentDiv);

        const isAtBottom =
          chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 50;
        chatLog.appendChild(details);
        if (isAtBottom) chatLog.scrollTop = chatLog.scrollHeight;
      } else {
        // Fallback if no result yet
        appendMessage('tool_call', `${msg.agent === 'er2' ? '🤖 ER2 · ' : '🔧 '}[${toolCallCounter}] ${desc.summary}`);
      }

      if (msg.name === 'update_plan') {
        // A rejected update (e.g. advancing past an unconfirmed pick), an
        // ignored self-invented task, or a replan the server folded back into
        // the original plan must not move the panel; the server echoes the
        // accepted plan otherwise.
        const r = msg.result && typeof msg.result === 'object' ? msg.result : {};
        const rejected = r.error || r.status === 'ignored' || r.status === 'kept_original_plan';
        if (!rejected) {
          // Render what the server ACCEPTED (it may keep progress instead of
          // the step the model proposed), so panel, camera and robot agree.
          const accepted = { ...(msg.args || {}) };
          if (Number.isInteger(r.current_step)) {
            if (r.current_step !== accepted.current_step) delete accepted.note;
            accepted.current_step = r.current_step;
          }
          if (typeof r.plan_status === 'string') accepted.status = r.plan_status;
          renderPlan(accepted);
        }
      } else {
        recordToolVisual(msg);
        renderRichToolCard(msg);
        // 3D robot twin + animated camera FX (pick/place/detect).
        window.dispatchEvent(new CustomEvent('robot-twin:tool', {
          detail: { name: msg.name, args: msg.args || {}, result: msg.result },
        }));
        startToolFx(msg);
      }

      if (msg.name === 'run_instruction') {
        updateRobotStatus(
          'executing', 'Executing: ' + (msg.args?.instruction || '...'));
      } else if (msg.name === 'stop') {
        updateRobotStatus('idle', 'Stopped');
      } else if (msg.name === 'reset') {
        updateRobotStatus('idle', 'Resetting...');
      }
    }
  } else if (msg.type === 'tool_response') {
    currentThinkingContentDiv = null;
    // Tool response breaks ack streak
    lastAckRow = null;
    ackCount = 0;
    appendMessage(
      'tool_response', `✅ ${msg.name}: ${formatFunctionResult(msg.result)}`);
  }
}

// --- Overlay Drawing ---
//
// Overlay items live in three layers that are redrawn together whenever the
// view changes (resize, new video source), so markers stay put:
//   * target     — the one-time detect target consumed by pick/place
//                  (point + 2D bounding box box_2d)
//   * trajectory — recommended 2D gripper path for pick/place (with arrows
//                  and optional pick-origin box/point)
//   * annotation — points/boxes/paths the model draws with draw_points
// In addition, `lastToolVisual` renders a live HUD badge + vector graphics
// for every non-ack tool call (look, drive, navigate, stow, stand, sit, etc.)
// so all ER2 tool parameters are directly visible on the camera canvas.
const overlayLayers = { target: [], trajectory: [], annotation: [] };
const overlayMeta = {
  trajectory: { connect: true, action: 'pick', origin: null, target: null },
  annotation: { connect: false },
};
let lastToolVisual = null;

const OVERLAY_STYLE = {
  target:     { fill: 'rgba(255, 122, 0, 0.96)',  stroke: '#ff7a00', boxFill: 'rgba(255, 122, 0, 0.14)', pillBg: 'rgba(120, 48, 0, 0.88)' },
  trajectory: { fill: 'rgba(0, 230, 118, 0.95)',  stroke: '#00e676', boxFill: 'rgba(0, 230, 118, 0.12)', pillBg: 'rgba(0, 72, 40, 0.86)' },
  pick_traj:  { fill: 'rgba(0, 229, 255, 0.95)',  stroke: '#00e5ff', boxFill: 'rgba(0, 229, 255, 0.12)', pillBg: 'rgba(0, 62, 84, 0.86)' },
  annotation: { fill: 'rgba(47, 128, 237, 0.94)', stroke: '#2f80ed', boxFill: 'rgba(47, 128, 237, 0.12)', pillBg: 'rgba(12, 36, 78, 0.84)' },
  origin:     { fill: 'rgba(0, 229, 255, 0.90)',  stroke: '#00e5ff', boxFill: 'rgba(0, 229, 255, 0.10)', pillBg: 'rgba(0, 56, 76, 0.84)' },
};

function normalizeOverlayPoint(p) {
  if (!p || typeof p !== 'object') return null;
  const x = Number(p.x ?? p.normalized_x), y = Number(p.y ?? p.normalized_y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  const item = {
    x: Math.max(0, Math.min(1000, x)),
    y: Math.max(0, Math.min(1000, y)),
    label: p.label ? String(p.label) : '',
    grasp_part: p.grasp_part ? String(p.grasp_part) : '',
    kind: p.kind || '',
  };
  const box = p.box || p.box_2d;
  if (Array.isArray(box) && box.length === 4 && box.every(v => Number.isFinite(Number(v)))) {
    item.box = box.map(v => Math.max(0, Math.min(1000, Number(v))));  // [ymin, xmin, ymax, xmax]
  }
  return item;
}

function drawPointsOverlay(points, layer = 'annotation', meta = {}) {
  const key = overlayLayers.hasOwnProperty(layer) ? layer : 'annotation';
  overlayLayers[key] = (points || []).map(normalizeOverlayPoint).filter(Boolean);
  if (key === 'target' && overlayLayers[key].length > 0) {
    // A fresh detect starts a new target step: clear any stale trajectory until
    // pick/place is called for this new target.
    overlayLayers.trajectory = [];
    const tp = overlayLayers[key][0];
    window.dispatchEvent(new CustomEvent('robot-twin:target', {
      detail: { x: tp.x, y: tp.y, label: tp.label },
    }));
    startDetectFx(tp);
  } else if (key === 'trajectory') {
    const refinedTarget = normalizeOverlayPoint(meta.target);
    overlayMeta.trajectory = {
      connect: meta.connect !== false,
      action: meta.action || 'pick',
      origin: normalizeOverlayPoint(meta.origin),
      target: refinedTarget,
    };
    if (refinedTarget) {
      overlayLayers.target = [refinedTarget];
    }
  } else if (key === 'annotation') {
    overlayMeta.annotation = {
      connect: Boolean(meta.connect),
    };
  }
  redrawOverlay();
}

function clearOverlay(layer) {
  if (layer && overlayLayers.hasOwnProperty(layer)) {
    overlayLayers[layer] = [];
  } else {
    Object.keys(overlayLayers).forEach(k => { overlayLayers[k] = []; });
    lastToolVisual = null;
    clearToolFx();
  }
  redrawOverlay();
}

// --- Animated camera FX (separate canvas above the static overlay) ---------
//
//   detect → a reticle that spins and locks onto the target point
//   pick   → a gripper claw travels the recommended trajectory, closes at the
//            grasp waypoint (ripple), then lifts along the remaining path
//   place  → the claw (holding) travels the transport path and releases at
//            the end with a green drop ripple + down arrow
// Purely cosmetic: it reads the same 0-1000 points the static overlay uses.
// `var` (not let/const): clearOverlay() may run before this point executes.
var fxCanvas = document.getElementById('fx-canvas');
var fxItems = [];
var fxRaf = null;

function clearToolFx() {
  fxItems = [];
  if (fxCanvas) {
    const ctx = fxCanvas.getContext('2d');
    ctx.clearRect(0, 0, fxCanvas.width, fxCanvas.height);
  }
}

function startDetectFx(p) {
  if (!p) return;
  fxItems = fxItems.filter(f => f.kind !== 'detect');
  fxItems.push({ kind: 'detect', p: { x: p.x, y: p.y }, t0: performance.now(), dur: 4500 });
  ensureFxLoop();
}

function startToolFx(msg) {
  if (!msg || (msg.name !== 'pick' && msg.name !== 'place')) {
    if (msg && msg.name === 'stop') clearToolFx();
    return;
  }
  const r = (msg.result && typeof msg.result === 'object') ? msg.result : {};
  if (r.error) return;
  const a = msg.args || {};
  let pts = overlayLayers.trajectory.length >= 2 ? overlayLayers.trajectory : null;
  if (!pts) {
    const raw = (Array.isArray(r.trajectory) && r.trajectory) || (Array.isArray(a.trajectory) && a.trajectory) || [];
    pts = raw.map(normalizeOverlayPoint).filter(Boolean);
  }
  if (pts.length < 2) {
    const tgt = overlayLayers.target[0];
    if (!tgt) return;
    pts = [{ x: tgt.x, y: Math.max(0, tgt.y - 120) }, { x: tgt.x, y: tgt.y }];
  }
  pts = pts.map(p => ({ x: p.x, y: p.y, label: (p.label || '').toLowerCase() }));
  // Where the gripper acts: the 'grasp' waypoint for pick, the last for place.
  let actIdx = pts.length - 1;
  if (msg.name === 'pick') {
    const gi = pts.findIndex(p => p.label.includes('grasp'));
    if (gi >= 0) actIdx = gi;
    else {
      const tgt = overlayLayers.target[0];
      if (tgt) {
        let best = Infinity;
        pts.forEach((p, i) => { const d = Math.hypot(p.x - tgt.x, p.y - tgt.y); if (d < best) { best = d; actIdx = i; } });
      }
    }
  }
  fxItems = [];  // a new pick/place replaces any running detect/pick FX
  fxItems.push({ kind: msg.name, pts, actIdx, t0: performance.now(), dur: 5200 });
  ensureFxLoop();
}

function ensureFxLoop() {
  if (!fxCanvas || fxRaf) return;
  const loop = () => {
    fxRaf = null;
    drawToolFx();
    if (fxItems.length) fxRaf = requestAnimationFrame(loop);
  };
  fxRaf = requestAnimationFrame(loop);
}

function fxPathLength(pts) {
  let L = 0;
  for (let i = 1; i < pts.length; i++) L += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
  return L;
}

function fxPointAt(pts, s) {  // s in [0, total length]
  for (let i = 1; i < pts.length; i++) {
    const seg = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
    if (s <= seg || i === pts.length - 1) {
      const u = seg > 0 ? Math.min(1, s / seg) : 1;
      return {
        x: pts[i - 1].x + (pts[i].x - pts[i - 1].x) * u,
        y: pts[i - 1].y + (pts[i].y - pts[i - 1].y) * u,
        ang: Math.atan2(pts[i].y - pts[i - 1].y, pts[i].x - pts[i - 1].x),
      };
    }
    s -= seg;
  }
  return { x: pts[0].x, y: pts[0].y, ang: 0 };
}

function drawClaw(ctx, x, y, open, color, scale = 1) {
  // A top-down two-jaw gripper icon pointing "down" into the image.
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(scale, scale);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3.2;
  // wrist
  ctx.beginPath(); ctx.moveTo(0, -30); ctx.lineTo(0, -14); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(-14, -14); ctx.lineTo(14, -14); ctx.stroke();
  // jaws
  const spread = 4 + open * 10;
  for (const s of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(s * 14, -14);
    ctx.quadraticCurveTo(s * (spread + 8), -4, s * spread, 6);
    ctx.stroke();
  }
  ctx.restore();
}

function drawRipple(ctx, x, y, u, color) {
  for (let k = 0; k < 3; k++) {
    const v = (u + k / 3) % 1;
    ctx.beginPath();
    ctx.arc(x, y, 8 + v * 38, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.globalAlpha = (1 - v) * 0.85;
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawToolFx() {
  if (!fxCanvas) return;
  const rect = fxCanvas.parentElement.getBoundingClientRect();
  if (fxCanvas.width !== Math.round(rect.width) || fxCanvas.height !== Math.round(rect.height)) {
    fxCanvas.width = Math.round(rect.width);
    fxCanvas.height = Math.round(rect.height);
  }
  const ctx = fxCanvas.getContext('2d');
  ctx.clearRect(0, 0, fxCanvas.width, fxCanvas.height);
  const m = mediaDisplayRect();
  const toPx = (nx, ny) => [m.x + (nx / 1000) * m.w, m.y + (ny / 1000) * m.h];
  const now = performance.now();
  fxItems = fxItems.filter(f => now - f.t0 < f.dur);
  const ease = u => u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;

  for (const f of fxItems) {
    const el = now - f.t0;
    const fade = Math.min(1, (f.dur - el) / 600);
    ctx.globalAlpha = fade;
    if (f.kind === 'detect') {
      const [x, y] = toPx(f.p.x, f.p.y);
      const lock = Math.min(1, el / 900);
      const R = 70 - 46 * ease(lock);
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(el / 380);
      ctx.strokeStyle = '#ffb300';
      ctx.shadowColor = '#ffb300';
      ctx.shadowBlur = 10;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([10, 8]);
      ctx.beginPath(); ctx.arc(0, 0, R, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
      for (let q = 0; q < 4; q++) {
        ctx.rotate(Math.PI / 2);
        ctx.beginPath(); ctx.moveTo(R + 4, 0); ctx.lineTo(R + 14, 0); ctx.stroke();
      }
      ctx.restore();
      if (lock >= 1) drawRipple(ctx, x, y, ((el - 900) / 1400) % 1, '#ffb300');
      continue;
    }
    // pick / place
    const isPick = f.kind === 'pick';
    const color = isPick ? '#00e5ff' : '#00e676';
    const pts = f.pts;
    const total = fxPathLength(pts);
    const actLen = fxPathLength(pts.slice(0, f.actIdx + 1));
    const tApproach = 1700, tAct = 700, tRest = 1300;
    let s, open, phaseAct = 0;
    if (el < tApproach) {
      s = actLen * ease(el / tApproach);
      open = isPick ? 1 : 0;
    } else if (el < tApproach + tAct) {
      s = actLen;
      phaseAct = (el - tApproach) / tAct;
      open = isPick ? 1 - phaseAct : phaseAct;
    } else {
      const u = Math.min(1, (el - tApproach - tAct) / tRest);
      s = actLen + (total - actLen) * ease(u);
      open = isPick ? 0 : 1;
    }
    // Travelled path glow.
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 5;
    ctx.globalAlpha = 0.35 * fade;
    ctx.lineCap = 'round';
    ctx.beginPath();
    let acc = 0;
    let [px0, py0] = toPx(pts[0].x, pts[0].y);
    ctx.moveTo(px0, py0);
    for (let i = 1; i < pts.length; i++) {
      const seg = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
      if (acc + seg >= s) {
        const p = fxPointAt(pts, s);
        const [qx, qy] = toPx(p.x, p.y);
        ctx.lineTo(qx, qy);
        break;
      }
      acc += seg;
      const [qx, qy] = toPx(pts[i].x, pts[i].y);
      ctx.lineTo(qx, qy);
    }
    ctx.stroke();
    ctx.restore();
    ctx.globalAlpha = fade;
    // Action point ripple (grasp / release).
    const ap = pts[f.actIdx];
    const [ax, ay] = toPx(ap.x, ap.y);
    if (el >= tApproach) drawRipple(ctx, ax, ay, ((el - tApproach) / 900) % 1, color);
    if (!isPick && el >= tApproach) {
      // Down arrow above the drop point.
      const bob = Math.sin(el / 160) * 4;
      ctx.save();
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.moveTo(ax, ay - 16 + bob);
      ctx.lineTo(ax - 10, ay - 30 + bob);
      ctx.lineTo(ax - 4, ay - 30 + bob);
      ctx.lineTo(ax - 4, ay - 44 + bob);
      ctx.lineTo(ax + 4, ay - 44 + bob);
      ctx.lineTo(ax + 4, ay - 30 + bob);
      ctx.lineTo(ax + 10, ay - 30 + bob);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
    // The claw itself (+ a held "item" dot while carrying).
    const cp = fxPointAt(pts, s);
    const [cx, cy] = toPx(cp.x, cp.y);
    const holding = isPick ? (el >= tApproach + tAct * 0.6) : (el < tApproach + tAct * 0.4);
    if (holding) {
      ctx.beginPath();
      ctx.arc(cx, cy + 2, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#ff7043';
      ctx.fill();
    }
    drawClaw(ctx, cx, cy, open, color, 1);
    // Caption.
    const cap = isPick
      ? (el < tApproach ? '🤏 approach' : el < tApproach + tAct ? '✊ grasp' : '⬆ lift')
      : (el < tApproach ? '📦 carry' : el < tApproach + tAct ? '🖐 release' : '✅ placed');
    ctx.font = '600 12px ui-sans-serif, system-ui, sans-serif';
    const tw = ctx.measureText(cap).width;
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(cx + 18, cy - 40, tw + 12, 20);
    ctx.fillStyle = '#fff';
    ctx.fillText(cap, cx + 24, cy - 26);
  }
  ctx.globalAlpha = 1;
}

function recordToolVisual(msg) {
  const a = msg.args || {};
  const r = (msg.result && typeof msg.result === 'object') ? msg.result : {};
  lastToolVisual = {
    name: msg.name || 'tool',
    args: a,
    result: r,
    source: msg.source || 'model',
    at: Date.now(),
  };
  // If pick/place carries a trajectory in args or result (e.g. if ui_overlay was
  // already stripped or re-sent by watchdog), ensure the trajectory layer is populated.
  if (msg.name === 'pick' || msg.name === 'place') {
    const rawTraj = (Array.isArray(r.trajectory) && r.trajectory.length ? r.trajectory : null) ||
      (Array.isArray(a.trajectory) && a.trajectory.length ? a.trajectory : null);
    if (rawTraj && !overlayLayers.trajectory.length) {
      drawPointsOverlay(rawTraj, 'trajectory', {
        connect: true,
        action: msg.name,
        origin: null,
        target: r.target || null,
      });
      return;
    }
  }
  redrawOverlay();
}

// Rectangle (in canvas px) covered by the full source frame, honouring the
// element's object-fit: the robot image uses `contain` (letterboxed) and the
// webcam <video> uses `cover` (cropped, so the rect may exceed the canvas).
function mediaDisplayRect() {
  const rect = overlayCanvas.parentElement.getBoundingClientRect();
  let srcW = 0, srcH = 0, fit = 'contain';
  if (!robotCamImg.classList.contains('hidden') &&
      robotCamImg.naturalWidth > 0 && robotCamImg.naturalHeight > 0) {
    srcW = robotCamImg.naturalWidth; srcH = robotCamImg.naturalHeight; fit = 'contain';
  } else if (videoPreview && !videoPreview.classList.contains('hidden') &&
             videoPreview.videoWidth > 0 && videoPreview.videoHeight > 0) {
    srcW = videoPreview.videoWidth; srcH = videoPreview.videoHeight;
    fit = (getComputedStyle(videoPreview).objectFit || 'cover');
  }
  if (!srcW || !srcH) return { x: 0, y: 0, w: rect.width, h: rect.height };
  const srcAspect = srcW / srcH;
  const boxAspect = rect.width / rect.height;
  let w, h;
  const widthLimited = fit === 'cover' ? boxAspect > srcAspect : boxAspect <= srcAspect;
  if (widthLimited) { w = rect.width; h = w / srcAspect; }
  else              { h = rect.height; w = h * srcAspect; }
  return { x: (rect.width - w) / 2, y: (rect.height - h) / 2, w, h };
}

function drawArrowhead(ctx, x0, y0, x1, y1, color, size = 10) {
  const dx = x1 - x0, dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  if (len < 6) return;
  const angle = Math.atan2(dy, dx);
  ctx.save();
  ctx.translate(x1, y1);
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(-size, -size * 0.55);
  ctx.lineTo(-size * 0.72, 0);
  ctx.lineTo(-size, size * 0.55);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.65)';
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.restore();
}

function drawBoxWithBrackets(ctx, toPx, box, st, isPrimary = true) {
  const [ymin, xmin, ymax, xmax] = box;
  const [bx0, by0] = toPx(xmin, ymin);
  const [bx1, by1] = toPx(xmax, ymax);
  const bw = bx1 - bx0, bh = by1 - by0;
  ctx.save();
  ctx.fillStyle = st.boxFill;
  ctx.fillRect(bx0, by0, bw, bh);
  ctx.strokeStyle = st.stroke;
  ctx.lineWidth = isPrimary ? 2 : 1.75;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(bx0, by0, bw, bh);

  // Crisp L-shaped HUD corner brackets so box_2d is unmistakable.
  ctx.setLineDash([]);
  ctx.lineWidth = isPrimary ? 3.2 : 2.5;
  const cLen = Math.max(8, Math.min(18, Math.min(Math.abs(bw), Math.abs(bh)) * 0.28));
  const corners = [
    [bx0, by0, 1, 1],
    [bx1, by0, -1, 1],
    [bx0, by1, 1, -1],
    [bx1, by1, -1, -1],
  ];
  ctx.beginPath();
  for (const [cx, cy, sx, sy] of corners) {
    ctx.moveTo(cx, cy + sy * cLen);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx + sx * cLen, cy);
  }
  ctx.stroke();
  ctx.restore();
  return [bx0, by0, bx1, by1];
}

function drawPill(ctx, rect, labelBoxes, text, anchorX, anchorY, st, bold = false) {
  ctx.save();
  ctx.font = `${bold ? 'bold ' : ''}12.5px sans-serif`;
  const padX = 6, padY = 4, lineH = 15;
  const tw = ctx.measureText(text).width;
  const pillW = tw + padX * 2, pillH = lineH + padY * 2 - 2;
  let px = Math.max(4, Math.min(anchorX, rect.width - pillW - 4));
  let py = Math.max(4, Math.min(anchorY, rect.height - pillH - 4));
  for (let guard = 0; guard < 10; guard++) {
    const hit = labelBoxes.find(
      b => px < b.x + b.w && px + pillW > b.x && py < b.y + b.h && py + pillH > b.y);
    if (!hit) break;
    py = hit.y + hit.h + 3;
  }
  py = Math.max(4, Math.min(py, rect.height - pillH - 4));
  labelBoxes.push({ x: px, y: py, w: pillW, h: pillH });

  ctx.fillStyle = st.pillBg || 'rgba(0, 0, 0, 0.78)';
  ctx.beginPath();
  ctx.roundRect(px, py, pillW, pillH, 5);
  ctx.fill();
  ctx.strokeStyle = st.stroke;
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.97)';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, px + padX, py + pillH / 2 + 0.5);
  ctx.restore();
}

function drawConnectedPath(ctx, rect, toPx, labelBoxes, pts, st, actionLabel = '') {
  if (!pts || pts.length < 2) return;
  const pxPts = pts.map(p => toPx(p.x, p.y));

  // Outer glow + inner dashed path
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  pxPts.forEach(([x, y], i) => { if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)';
  ctx.lineWidth = 6.5;
  ctx.stroke();

  ctx.beginPath();
  pxPts.forEach(([x, y], i) => { if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
  ctx.strokeStyle = st.stroke;
  ctx.lineWidth = 3.2;
  ctx.setLineDash([8, 5]);
  ctx.stroke();
  ctx.restore();

  // Directional arrowheads on each segment (midpoint + endpoint)
  for (let i = 0; i < pxPts.length - 1; i++) {
    const [x0, y0] = pxPts[i];
    const [x1, y1] = pxPts[i + 1];
    const segLen = Math.hypot(x1 - x0, y1 - y0);
    if (segLen > 55) {
      const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      drawArrowhead(ctx, x0, y0, mx, my, st.stroke, 9);
    }
    // Pull arrowhead slightly back from the node center so it doesn't cover the number badge
    if (segLen > 18) {
      const ratio = Math.max(0.5, (segLen - 11) / segLen);
      const ex = x0 + (x1 - x0) * ratio;
      const ey = y0 + (y1 - y0) * ratio;
      drawArrowhead(ctx, x0, y0, ex, ey, st.stroke, 11);
    }
  }

  // Waypoint numbered nodes + coordinate labels
  pts.forEach((pt, idx) => {
    const [cx, cy] = pxPts[idx];
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, 9, 0, 2 * Math.PI);
    ctx.fillStyle = st.fill;
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.96)';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#061a14';
    ctx.font = 'bold 10.5px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(idx + 1), cx, cy + 0.5);
    ctx.restore();

    const stepTag = pt.label ? `${idx + 1}·${pt.label}` : `wp${idx + 1}`;
    const prefix = idx === 0 && actionLabel ? `${actionLabel} ` : '';
    const txt = `${prefix}${stepTag} (${Math.round(pt.x)}, ${Math.round(pt.y)})`;
    drawPill(ctx, rect, labelBoxes, txt, cx + 13, cy - 11, st, idx === 0 || idx === pts.length - 1);
  });
}

function drawToolTelemetryOverlay(ctx, rect, labelBoxes) {
  if (!lastToolVisual) return;
  const tv = lastToolVisual;
  const name = tv.name;
  const a = tv.args || {};
  const r = tv.result || {};
  const hudStyle = { stroke: '#60a5fa', pillBg: 'rgba(15, 23, 42, 0.86)' };

  // Build concise summary of all parameters for the top-left HUD badge
  let badgeText = '';
  if (name === 'detect') {
    const t = r.target || {};
    const x = a.x ?? t.normalized_x;
    const y = a.y ?? t.normalized_y;
    const gp = a.grasp_part || t.grasp_part;
    const gpStr = gp ? ` · grasp="${gp}"` : '';
    const box = a.box_2d || t.box_2d;
    const boxStr = Array.isArray(box) ? ` · box_2d=[${box.map(Math.round).join(', ')}]` : '';
    badgeText = `🎯 detect("${a.instruction || t.label || ''}") · grasp_xy=(${Math.round(x || 0)}, ${Math.round(y || 0)})${gpStr}${boxStr}`;
  } else if (name === 'pick' || name === 'place') {
    const t = r.target || {};
    const pt = r.grasp_point || r.place_point || [t.normalized_x, t.normalized_y];
    const gp = r.grasp_part || a.grasp_part || t.grasp_part;
    const gpStr = (name === 'pick' && gp) ? ` · grasp="${gp}"` : '';
    const traj = Array.isArray(r.trajectory) && r.trajectory.length ? r.trajectory : (Array.isArray(a.trajectory) ? a.trajectory : []);
    const trajStr = traj.length ? ` · path=[${traj.map(p => `(${Math.round(p.x)},${Math.round(p.y)})`).join('→')}]` : '';
    const boxStr = Array.isArray(t.box_2d) ? ` · box=[${t.box_2d.map(Math.round).join(',')}]` : '';
    const icon = name === 'pick' ? '🤏' : '📦';
    const lbl = t.label || r.target_label || r.placed || '';
    const xyStr = Array.isArray(pt) && pt[0] != null ? ` @ (${Math.round(pt[0])}, ${Math.round(pt[1])})` : '';
    badgeText = `${icon} ${name}(${lbl ? `"${lbl}"` : ''})${xyStr}${gpStr}${boxStr}${trajStr}`;
  } else if (name === 'look') {
    const dir = a.direction || r.direction || 'forward';
    const rad = Number(a.angle_rad ?? r.angle_rad ?? 0.15);
    const deg = Math.round((rad * 180) / Math.PI);
    badgeText = `👁️ look(direction="${dir}", angle_rad=${rad.toFixed(2)} rad / ${deg}°)`;

    // Draw camera-gaze vector from frame center
    const cx = rect.width / 2, cy = rect.height / 2;
    const len = Math.max(45, Math.min(130, (rad / 0.35) * 120));
    const dirs = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
    const [dx, dy] = dirs[dir] || [0, 0];
    if (dx || dy) {
      const ex = cx + dx * len, ey = cy + dy * len;
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
      ctx.fillStyle = '#38bdf8';
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 3.5;
      ctx.stroke();
      drawArrowhead(ctx, cx, cy, ex, ey, '#38bdf8', 12);
      ctx.restore();
      drawPill(ctx, rect, labelBoxes, `👁️ look ${dir} (${deg}°)`, ex + 10, ey - 12, { stroke: '#38bdf8', pillBg: 'rgba(7, 46, 74, 0.88)' }, true);
    }
  } else if (name === 'drive') {
    const vx = Number(a.v_x ?? r.v_x ?? 0);
    const vy = Number(a.v_y ?? r.v_y ?? 0);
    const vrot = Number(a.v_rot ?? r.v_rot ?? 0);
    const dur = Number(a.duration ?? r.duration ?? 0);
    badgeText = `🚗 drive(v_x=${vx.toFixed(2)} m/s, v_y=${vy.toFixed(2)} m/s, v_rot=${vrot.toFixed(2)} rad/s, duration=${dur.toFixed(1)}s)`;

    // Draw body-frame velocity compass in bottom-right of canvas
    const cx = rect.width - 78, cy = rect.height - 86, rad = 42;
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(15, 23, 42, 0.80)';
    ctx.fill();
    ctx.strokeStyle = '#a78bfa';
    ctx.lineWidth = 2;
    ctx.stroke();
    // Forward/strafe vector (vx > 0 is up, vy > 0 is left)
    const tx = cx - (vy / 0.5) * (rad - 8);
    const ty = cy - (vx / 0.8) * (rad - 8);
    if (Math.hypot(tx - cx, ty - cy) > 4) {
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(tx, ty);
      ctx.strokeStyle = '#34d399';
      ctx.lineWidth = 3;
      ctx.stroke();
      drawArrowhead(ctx, cx, cy, tx, ty, '#34d399', 9);
    }
    // Yaw rotation arc
    if (Math.abs(vrot) > 0.02) {
      ctx.beginPath();
      const startA = -Math.PI / 2;
      const endA = startA - Math.max(-1, Math.min(1, vrot)) * 1.3;
      ctx.arc(cx, cy, rad - 6, startA, endA, vrot > 0);
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }
    ctx.restore();
  } else if (name === 'navigate') {
    const dest = a.waypoint || r.destination || '';
    const from = r.from_waypoint ? `${r.from_waypoint} ➔ ` : '';
    badgeText = `🧭 navigate(waypoint="${dest}") · ${from}${dest}`;
  } else if (name === 'stow' || name === 'stand' || name === 'sit') {
    badgeText = `🤖 ${name}() → status: ${r.status || name}`;
  } else if (name === 'wait_for_pick_up') {
    badgeText = `🤝 wait_for_pick_up(monitor=${a.monitor_sec ?? 30}s, threshold=${a.upward_threshold_m ?? 0.02}m)`;
  } else if (name === 'confirm_action') {
    badgeText = `✅ confirm_action("${a.observation || ''}")`;
  } else if (name === 'draw_points') {
    const pts = Array.isArray(a.points) ? a.points : [];
    badgeText = `📍 draw_points(${pts.length} pts${a.connect ? ', connect=true' : ''})`;
  } else {
    badgeText = `🔧 ${name}(${JSON.stringify(a)})`;
  }

  if (badgeText) {
    drawPill(ctx, rect, labelBoxes, badgeText, 12, 12, hudStyle, true);
  }
}

function redrawOverlay() {
  if (!overlayCanvas) return;
  const rect = overlayCanvas.parentElement.getBoundingClientRect();
  overlayCanvas.width = rect.width;   // also clears the canvas
  overlayCanvas.height = rect.height;

  const hasPoints =
    overlayLayers.annotation.length > 0 ||
    overlayLayers.trajectory.length > 0 ||
    overlayLayers.target.length > 0 ||
    Boolean(lastToolVisual);
  if (!hasPoints) return;

  const m = mediaDisplayRect();
  const toPx = (nx, ny) => [m.x + (nx / 1000) * m.w, m.y + (ny / 1000) * m.h];
  const ctx = overlayCanvas.getContext('2d');
  const labelBoxes = [];

  // 1) Top-left live tool telemetry HUD & vector graphics (look / drive / navigate / etc.)
  drawToolTelemetryOverlay(ctx, rect, labelBoxes);

  // 2) During `place`, mark where the carried object is now (trajectory start), if it has a box
  const trajMeta = overlayMeta.trajectory || {};
  if (overlayLayers.trajectory.length > 0 && trajMeta.origin && trajMeta.action === 'place') {
    const orig = trajMeta.origin;
    const origSt = OVERLAY_STYLE.origin;
    if (orig.box) {
      const [bx0, by0] = drawBoxWithBrackets(ctx, toPx, orig.box, origSt, false);
      const bStr = orig.box.map(Math.round).join(',');
      drawPill(ctx, rect, labelBoxes, `📤 from: ${orig.label || 'item'} (${Math.round(orig.x)}, ${Math.round(orig.y)}) [box: ${bStr}]`, bx0, by0 - 26, origSt, false);
    }
  }

  // 3) Connected trajectory layer (`pick` or `place` recommended 2D path)
  if (overlayLayers.trajectory.length >= 2) {
    const isPick = (trajMeta.action || 'pick') === 'pick';
    const trajSt = isPick ? OVERLAY_STYLE.pick_traj : OVERLAY_STYLE.trajectory;
    const tag = isPick ? '🤏 pick path:' : '📦 place path:';
    drawConnectedPath(ctx, rect, toPx, labelBoxes, overlayLayers.trajectory, trajSt, tag);
  }

  // 4) Annotation layer (`draw_points`) — optionally connected as a path
  if (overlayLayers.annotation.length >= 2 && overlayMeta.annotation.connect) {
    drawConnectedPath(ctx, rect, toPx, labelBoxes, overlayLayers.annotation, OVERLAY_STYLE.annotation, '🛤️ path:');
  }

  // 5) Individual annotation points/boxes & target crosshair/box_2d on top
  const items = [
    ...overlayLayers.annotation.map(p => ({ ...p, layer: 'annotation' })),
    ...overlayLayers.target.map(p => ({ ...p, layer: 'target' })),
  ];

  for (const it of items) {
    const st = OVERLAY_STYLE[it.layer];
    const isTarget = it.layer === 'target';
    const [cx, cy] = toPx(it.x, it.y);

    // Bounding box (`box_2d`) with HUD corner brackets.
    let boxRect = null;
    if (it.box) {
      boxRect = drawBoxWithBrackets(ctx, toPx, it.box, st, isTarget);
    }

    // Point marker (target gets a crosshair ring).
    const r = isTarget ? 7 : 6;
    if (isTarget) {
      ctx.save();
      ctx.strokeStyle = st.stroke;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, 15, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - 22, cy); ctx.lineTo(cx - 10, cy);
      ctx.moveTo(cx + 10, cy); ctx.lineTo(cx + 22, cy);
      ctx.moveTo(cx, cy - 22); ctx.lineTo(cx, cy - 10);
      ctx.moveTo(cx, cy + 10); ctx.lineTo(cx, cy + 22);
      ctx.stroke();
      ctx.restore();
    }
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.fillStyle = st.fill;
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Label pill: "label (x, y) [grasp: part] [box: ymin,xmin,ymax,xmax]"
    const coords = `(${Math.round(it.x)}, ${Math.round(it.y)})`;
    const graspTag = (isTarget && it.grasp_part) ? ` 🤏[${it.grasp_part}]` : '';
    const boxTag = it.box ? ` [box: ${it.box.map(Math.round).join(', ')}]` : '';
    const text = (it.label ? `${it.label} ${coords}` : coords) + graspTag + boxTag;
    let px = cx + r + 8, py = cy - 12;
    if (boxRect) {
      px = boxRect[0];
      py = boxRect[1] - 26;
    }
    drawPill(ctx, rect, labelBoxes, text, px, py, st, isTarget);
  }
}

// Keep markers aligned when the layout or the video geometry changes.
window.addEventListener('resize', redrawOverlay);
if (videoPreview) videoPreview.addEventListener('loadedmetadata', redrawOverlay);
if (robotCamImg) robotCamImg.addEventListener('load', redrawOverlay);

// In webcam mode size the container to the camera's intrinsic aspect ratio
// so the preview is neither cropped (object-fit: cover) nor letterboxed and
// shows exactly the frame the model receives.
function syncContainerAspectToWebcam() {
  const container = videoPreview && videoPreview.parentElement;
  if (!container) return;
  const apply = () => {
    if (videoSource !== 'computer') return;
    if (videoPreview.videoWidth > 0 && videoPreview.videoHeight > 0) {
      container.style.aspectRatio =
        `${videoPreview.videoWidth} / ${videoPreview.videoHeight}`;
      redrawOverlay();
    }
  };
  apply();
  if (!videoPreview._aspectSyncBound) {
    videoPreview.addEventListener('loadedmetadata', apply);
    videoPreview._aspectSyncBound = true;
  }
}

// Human-readable one-line summary for a tool call row in the chat, surfacing
// coordinates (📍), box_2d (📦), and trajectory waypoints (🛤️) for grounding tools.
function describeToolCall(msg) {
  const name = msg.name || 'tool';
  const a = msg.args || {};
  const r = (msg.result && typeof msg.result === 'object') ? msg.result : {};
  const rawArgs = msg.args ? JSON.stringify(msg.args) : '';
  const fmtPt = (x, y) => `📍 (${Math.round(x)}, ${Math.round(y)})`;
  const q = (s) => JSON.stringify(String(s));

  const byWatchdog = msg.source === 'watchdog';
  const wd = (r.watchdog && typeof r.watchdog === 'object') ? r.watchdog : null;

  if (name === 'get_current_location') {
    if (r.available === false || r.error) {
      return { summary: `📍 get_current_location ✗ ${r.error || 'unavailable'}`, open: true, argsText: r.next_action || '' };
    }
    const acc = Number.isFinite(Number(r.accuracy_m)) ? ` ±${Math.round(r.accuracy_m)}m` : '';
    const desc = r.place_description ? `\n${r.place_description}` : '';
    return {
      summary: `📍 get_current_location → ${Number(r.latitude).toFixed(5)}, ${Number(r.longitude).toFixed(5)}${acc}`,
      open: true,
      argsText: `${r.google_maps_url || ''}${desc}`.trim(),
    };
  }
  if (name === 'google_search' || name === 'google_maps_search') {
    const icon = name === 'google_search' ? '🔎' : '🗺️';
    if (r.error) {
      return { summary: `${icon} ${name}(${q(a.query || '')}) ✗ ${r.error}`, open: true, argsText: '' };
    }
    const srcs = Array.isArray(r.sources) ? r.sources : [];
    const srcText = srcs.map((s) => `• ${s.title || s.uri || ''}${s.uri ? ' — ' + s.uri : ''}`).join('\n');
    const lat = Number.isFinite(Number(r.latency_ms)) ? ` · ${r.latency_ms}ms` : '';
    return {
      summary: `${icon} ${name}(${q(a.query || '')}) → ${srcs.length} source${srcs.length === 1 ? '' : 's'}${lat}`,
      open: true,
      argsText: `${r.answer || ''}${srcText ? '\n\nSources:\n' + srcText : ''}`.trim(),
    };
  }

  if (name === 'update_plan') {
    const steps = Array.isArray(a.steps) ? a.steps : [];
    const cur = Math.max(0, Math.min(parseInt(a.current_step, 10) || 0, steps.length - 1));
    const status = a.status || 'in_progress';
    if (r.error) {
      return { summary: `📋 update_plan ✗ ${r.error}`, open: false, argsText: r.next_action || '' };
    }
    const head = status === 'done' ? `done ✓ ${steps.length}/${steps.length}`
      : status === 'failed' ? `FAILED at step ${cur + 1}/${steps.length} · ${steps[cur] || ''}`
      : `step ${cur + 1}/${steps.length} · ${steps[cur] || ''}`;
    const argsText = steps.map((s, i) => `${i === cur && status !== 'done' ? '▶' : i < cur || status === 'done' ? '✓' : ' '} ${i + 1}. ${s}`).join('\n');
    const prefix = byWatchdog ? '🛡️ watchdog → ' : '📋 ';
    return { summary: `${prefix}update_plan → ${head}${a.note ? ' — ' + a.note : ''}`, open: false, argsText };
  }
  if (name === 'detect') {
    const t = r.target || {};
    const x = Number.isFinite(Number(a.x)) ? Number(a.x) : Number(t.normalized_x);
    const y = Number.isFinite(Number(a.y)) ? Number(a.y) : Number(t.normalized_y);
    const where = Number.isFinite(x) && Number.isFinite(y) ? ' ' + fmtPt(x, y) : '';
    const gp = a.grasp_part || t.grasp_part;
    const gpText = gp ? ` 🤏[${gp}]` : '';
    const box = a.box_2d || t.box_2d;
    const boxText = Array.isArray(box) ? ` 📦 box[${box.map(Math.round).join(',')}]` : '';
    const fail = r.detected === false || r.error ? ' ✗' : '';
    return { summary: `detect(${q(a.instruction || '')})${where}${gpText}${boxText}${fail}`, open: true, argsText: '' };
  }
  if (name === 'pick' || name === 'place') {
    const pt = r.grasp_point || r.place_point ||
      (r.target && [r.target.normalized_x, r.target.normalized_y]);
    const where = Array.isArray(pt) && pt.length === 2 ? ' ' + fmtPt(pt[0], pt[1]) : '';
    const gp = r.grasp_part || a.grasp_part || (r.target && r.target.grasp_part);
    const gpText = (name === 'pick' && gp) ? ` 🤏[${gp}]` : '';
    const box = r.target && r.target.box_2d;
    const boxText = Array.isArray(box) ? ` 📦[${box.map(Math.round).join(',')}]` : '';
    const traj = (Array.isArray(r.trajectory) && r.trajectory.length ? r.trajectory : null) ||
      (Array.isArray(a.trajectory) ? a.trajectory : []);
    const trajText = traj.length
      ? ` 🛤️ ${traj.map(p => `${p.label ? p.label + ':' : ''}(${Math.round(p.x)},${Math.round(p.y)})`).join('→')}`
      : '';
    const label = (r.target && r.target.label) || r.target_label || r.placed || '';
    let tail = '';
    if (r.error) {
      tail = ` ✗ ${String(r.error).slice(0, 90)}`;
    } else if (wd) {
      if (wd.merged) tail = ` · attempt ${wd.attempt}/${wd.max_attempts} already in flight`;
      else if (wd.attempt > 1) tail = ` · 🔁 retry ${wd.attempt}/${wd.max_attempts}`;
      else tail = ` · attempt ${wd.attempt}/${wd.max_attempts} · awaiting confirmation`;
    }
    const prefix = byWatchdog ? '🛡️ watchdog re-sent ' : '';
    return { summary: `${prefix}${name}()${label ? ' ' + q(label) : ''}${where}${gpText}${boxText}${trajText}${tail}`, open: !byWatchdog, argsText: '' };
  }
  if (name === 'confirm_action') {
    const ok = r.status === 'confirmed';
    const head = ok ? `✅ confirm_action → ${r.action} executed (attempt ${r.attempts_used || 1})`
      : r.status === 'nothing_pending' ? '✅ confirm_action → nothing pending'
      : `confirm_action → ${r.status || ''}`;
    return { summary: `${head}${a.observation ? ' · ' + a.observation : ''}`, open: false, argsText: '' };
  }
  if (name === 'draw_points') {
    const pts = Array.isArray(a.points) ? a.points : [];
    const sep = a.connect ? ' → ' : ' · ';
    const parts = pts.slice(0, 5).map(p => {
      const b = Array.isArray(p.box_2d) ? `[${p.box_2d.map(Math.round).join(',')}]` : '';
      return `${p.label ? p.label + ' ' : ''}(${Math.round(p.x)}, ${Math.round(p.y)})${b}`;
    });
    const more = pts.length > 5 ? ` +${pts.length - 5}` : '';
    return { summary: `draw_points ${a.connect ? '🛤️' : '📍'} ${parts.join(sep)}${more}`, open: false, argsText: '' };
  }
  if (name === 'send_message') {
    return { summary: `send_message(${a.target || 'user'}): ${a.message || ''}`, open: false, argsText: '' };
  }
  return { summary: `${name}(${rawArgs})`, open: true, argsText: '' };
}

let robotState = 'idle';
let robotStateText = '';
// Backend ⇄ Gemini socket health ('ok' | 'reconnecting'); the browser's
// own WebSocket stays up while the backend swaps the model socket.
let streamState = 'ok';

function updateRobotStatus(state, text) {
  robotState = state;
  robotStateText = text;
  refreshStatusLabel();
}

function setStreamState(state) {
  streamState = state;
  if (!statusLabel) return;
  if (state === 'reconnecting' && geminiClient.isConnected()) {
    statusLabel.className = 'status-label reconnecting';
    statusLabel.textContent = '🔄 Reconnecting…';
    return;
  }
  refreshStatusLabel();
  updateSessionTimer();
}

function describeStreamStatus(msg) {
  const detail = [msg.code, msg.reason].filter(Boolean).join(' ');
  switch (msg.state) {
    case 'stalled':
      return '⚠️ 模型连接无响应（语音通道卡住），正在重连…';
    case 'reconnecting':
      return `🔄 正在重连模型 (${msg.attempt || 1}/${msg.max_attempts || 3})` +
        (detail ? `，上次断开：${detail}` : '') + '…';
    case 'reconnected':
      return msg.resumed ?
        '✅ 模型已重连，对话上下文已恢复。' :
        '✅ 模型已重连（新会话，之前的对话上下文已重置）。';
    case 'failed':
      return '❌ 模型重连失败' + (msg.reason ? `：${msg.reason}` : '') +
        '，请点击 ⏻ 重新连接。';
    default:
      return `ℹ️ 模型连接状态：${msg.state || '?'}`;
  }
}

function handleStreamStatus(msg) {
  appendMessage('system', describeStreamStatus(msg));
  if (msg.state === 'stalled' || msg.state === 'reconnecting') {
    setStreamState('reconnecting');
  } else {
    setStreamState('ok');
  }
}

function refreshStatusLabel() {
  if (!statusLabel) return;
  if (streamState === 'reconnecting' && geminiClient.isConnected()) {
    statusLabel.className = 'status-label reconnecting';
    statusLabel.textContent = '🔄 Reconnecting…';
    return;
  }
  // Robot executing takes priority over connection state display
  if (robotState === 'executing' && geminiClient.isConnected()) {
    statusLabel.className = 'status-label executing';
    statusLabel.textContent = robotStateText;
    return;
  }
  // Otherwise show connection state (timer is updated separately)
  if (geminiClient.isConnected()) {
    statusLabel.className = 'status-label connected';
    if (!sessionStartTime) {
      statusLabel.textContent = 'Connected';
    }
  }
}

function appendMessage(type, text) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${type}`;
  msgDiv.textContent = text;
  const isAtBottom =
    chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 50;
  chatLog.appendChild(msgDiv);
  if (isAtBottom) {
    chatLog.scrollTop = chatLog.scrollHeight;
  }
  return msgDiv;
}

function appendThoughtMessage(text) {
  const details = document.createElement('details');
  details.className = 'message thought';
  details.open = false; // Start collapsed

  const summary = document.createElement('summary');
  summary.textContent = 'Thinking Process';
  details.appendChild(summary);

  const contentDiv = document.createElement('div');
  contentDiv.className = 'thought-content';
  contentDiv.textContent = text;
  details.appendChild(contentDiv);

  const isAtBottom =
    chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 50;
  chatLog.appendChild(details);
  if (isAtBottom) {
    chatLog.scrollTop = chatLog.scrollHeight;
  }
  return contentDiv;
}

// --- Connect ---

connectBtn.onclick = async () => {
  resetPlan();
  if (geminiClient.isConnected()) {
    geminiClient.disconnect();
  } else {
    // Clear previous conversation history when starting a new connection
    chatLog.innerHTML = '';
    const memoryLog = document.getElementById('memory-log');
    if (memoryLog) memoryLog.innerHTML = '';

    statusLabel.className = 'status-label connecting';
    statusLabel.textContent = 'Connecting…';
    connectBtn.disabled = true;
    connectBtn.classList.add('connecting');

    try {
      await mediaHandler.initializeAudio();

      const sessionCfg = getResolvedSessionConfig();
      const connParams = {
        agent_name: selectedAgentName,
        model: sessionCfg.model,
        response_modality: sessionCfg.response_modality,
        use_tts: sessionCfg.use_tts,
        endpoint_type: sessionCfg.endpoint_type,
        thinking_level: pendingThinkingLevel(sessionCfg.model),
      };
      activeThinkingLevel = connParams.thinking_level;
      renderThinkingStatus();
      geminiClient.connect(connParams);
    } catch (error) {
      console.error('Connection error:', error);
      statusLabel.className = 'status-label error';
      statusLabel.textContent = 'Error';
      connectBtn.disabled = false;
      connectBtn.classList.remove('connecting');
    }
  }
};

// Mic button visual state. Before a session exists the button shows whether
// the mic is *armed* (will auto-start on Connect); once connected it shows
// whether capture is actually running.
function updateMicUI() {
  if (!micBtn) return;
  if (geminiClient.isConnected()) {
    micBtn.classList.toggle('active', mediaHandler.isRecording);
    micBtn.title = mediaHandler.isRecording ? 'Stop Microphone' : 'Start Microphone';
  } else {
    micBtn.classList.toggle('active', micAutoStart);
    micBtn.title = micAutoStart ?
      'Microphone ON — starts automatically on Connect (click to disable)' :
      'Microphone OFF — click to enable auto-start on Connect';
  }
}

micBtn.onclick = async (ev) => {
  // A real user click (not the programmatic auto-start) records the
  // preference so an explicit "mic off" survives reconnects/reloads.
  const userInitiated = !!(ev && ev.isTrusted);
  if (!geminiClient.isConnected()) {
    // Not connected: just toggle the armed preference, no capture yet.
    micAutoStart = !micAutoStart;
    localStorage.setItem('lite_mic_auto', String(micAutoStart));
    updateMicUI();
    return;
  }
  if (mediaHandler.isRecording) {
    mediaHandler.stopAudio();
    micBtn.classList.remove('active');
    micBtn.title = 'Start Microphone';
    if (userInitiated) {
      micAutoStart = false;
      localStorage.setItem('lite_mic_auto', 'false');
    }
  } else {
    try {
      await mediaHandler.startAudio((data) => {
        if (geminiClient.isConnected()) {
          geminiClient.send(data);
        }
      });
      micBtn.classList.add('active');
      micBtn.title = 'Stop Microphone';
      if (userInitiated) {
        micAutoStart = true;
        localStorage.setItem('lite_mic_auto', 'true');
      }
    } catch (e) {
      console.error('Mic start failed:', e);
      alert('Could not start audio capture');
    }
  }
};

sendBtn.onclick = sendText;
textInput.onkeypress = (e) => {
  if (e.key === 'Enter') sendText();
};

function sendText() {
  const text = textInput.value;
  if (text && geminiClient.isConnected()) {
    mediaHandler.stopAudioPlayback();
    resetLatencyTimer();
    geminiClient.sendText(text);
    appendMessage('user', text);
    textInput.value = '';
  }
}

function resetUI() {
  connectBtn.classList.remove('danger', 'connected', 'connecting');
  connectBtn.title = 'Connect to Agent';
  if (modelSelectorBtn) modelSelectorBtn.disabled = false;
  if (audioToggleBtn) audioToggleBtn.disabled = false;
  if (agentSelectorBtn) agentSelectorBtn.disabled = false;
  updateAudioUI();

  mediaHandler.stopAudio();
  mediaHandler.stopVideo(videoPreview);
  robotCamImg.classList.add('hidden');
  robotCamImg.src = '';
  videoPlaceholder.classList.remove('hidden');
  clearOverlay();
  clearToolFx();
  window.dispatchEvent(new CustomEvent('robot-twin:reset'));

  updateMicUI();
  connectBtn.disabled = false;
  updateRobotStatus('idle', 'Stopped');
  toolCallCounter = 0;
  lastAckRow = null;
  ackCount = 0;
}

// --- Settings Modal and Theme Toggle ---

let settingsBtn = document.getElementById('settings-btn');
let settingsModal = document.getElementById('settings-modal');
let closeModalBtn = document.getElementById('close-modal-btn');
let themeRadios = document.querySelectorAll('input[name="theme"]');
let modalTabs = document.querySelectorAll('.modal-sidebar li[data-tab]');

let selectedAgentName = 'spot';  // Track the currently selected agent

// Tab switching
modalTabs.forEach(tab => {
  tab.onclick = () => {
    modalTabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const tabName = tab.dataset.tab;
    document.querySelectorAll('.modal-panel')
      .forEach(p => p.classList.add('hidden'));
    document.getElementById(`modal-panel-${tabName}`)
      .classList.remove('hidden');
  };
});

// Fetch and display agent config
async function loadAgentConfig(name, model) {
  try {
    let url = `/api/agent_config/${encodeURIComponent(name)}?endpoint_type=${encodeURIComponent(selectedEndpointType)}`;
    if (model) {
      url += `&model=${encodeURIComponent(model)}`;
    }
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const cfg = await resp.json();

    document.getElementById('config-title').textContent =
      `${cfg.name} Agent Config`;
    const sessionCfg = getResolvedSessionConfig();
    document.getElementById('config-model').textContent = sessionCfg.model;
    document.getElementById('config-modality').textContent =
      sessionCfg.response_modality;
    document.getElementById('config-tts').textContent =
      sessionCfg.use_tts ? 'Enabled' : 'Disabled';
    document.getElementById('config-tts-voice').textContent = cfg.tts_voice;
    document.getElementById('config-endpoint').textContent = cfg.endpoint_type;
    document.getElementById('config-service').textContent = cfg.service_address;

    // Tools
    const toolsDiv = document.getElementById('config-tools');
    toolsDiv.innerHTML = '';
    if (cfg.tools && cfg.tools.length > 0) {
      cfg.tools.forEach(tool => {
        const chip = document.createElement('span');
        chip.className = 'tool-chip';
        chip.textContent = tool.name;
        chip.title = tool.description || '';
        toolsDiv.appendChild(chip);
      });
    } else {
      toolsDiv.textContent = 'None';
    }

    // Instructions
    const devPromptEl = document.getElementById('config-developer-prompt');
    if (devPromptEl) devPromptEl.value = cfg.developer_instruction || '';

    const hbTextEl = document.getElementById('config-heartbeat-text');
    if (hbTextEl) hbTextEl.value = cfg.heartbeat_text || '';

    // Thinking level: saved user choice wins over the server's model default.
    syncThinkingLevelForModel(selectedModel);

    document.getElementById('config-heartbeat-interval').value =
      cfg.heartbeat_interval_seconds !== undefined ? cfg.heartbeat_interval_seconds : '';
    document.getElementById('config-heartbeat-min-delay').value =
      cfg.heartbeat_min_delay_seconds !== undefined ? cfg.heartbeat_min_delay_seconds : '';
    document.getElementById('config-use-event-driven-heartbeat').checked =
      !!cfg.use_event_driven_heartbeat;

  } catch (e) {
    console.error('Failed to load agent config:', e);
    const devPromptEl = document.getElementById('config-developer-prompt');
    if (devPromptEl) devPromptEl.value = 'Error loading config';
  }
}

if (settingsBtn) {
  settingsBtn.onclick = () => {
    settingsModal.classList.remove('hidden');
    // Reset to Agent Config tab
    modalTabs.forEach(t => t.classList.remove('active'));
    document.getElementById('modal-tab-config').classList.add('active');
    document.querySelectorAll('.modal-panel')
      .forEach(p => p.classList.add('hidden'));
    document.getElementById('modal-panel-config').classList.remove('hidden');
    loadAgentConfig(selectedAgentName, selectedModel);
  };
}

let updatePromptBtn = document.getElementById('update-prompt-btn');
if (updatePromptBtn) {
  updatePromptBtn.onclick = async () => {
    const newDeveloperPrompt =
      document.getElementById('config-developer-prompt') ? document.getElementById('config-developer-prompt').value : '';
    const newHeartbeatText =
      document.getElementById('config-heartbeat-text') ? document.getElementById('config-heartbeat-text').value : '';
    const heartbeatInterval =
      document.getElementById('config-heartbeat-interval').value;
    const heartbeatMinDelay =
      document.getElementById('config-heartbeat-min-delay').value;
    const useEventDrivenHeartbeat =
      document.getElementById('config-use-event-driven-heartbeat').checked;
    updatePromptBtn.disabled = true;
    updatePromptBtn.textContent = 'Updating...';
    try {
      const resp = await fetch('/api/config/instructions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          agent_name: selectedAgentName,
          developer_instruction: newDeveloperPrompt,
          heartbeat_text: newHeartbeatText,
          heartbeat_interval_seconds: heartbeatInterval !== '' ? parseFloat(heartbeatInterval) : null,
          heartbeat_min_delay_seconds: heartbeatMinDelay !== '' ? parseFloat(heartbeatMinDelay) : null,
          use_event_driven_heartbeat: useEventDrivenHeartbeat,
        }),
      });
      const data = await resp.json();
      if (data.success) {
        updatePromptBtn.textContent = 'Updated ✅';
        setTimeout(() => {
          loadAgentConfig(selectedAgentName, selectedModel);
        }, 1000);
      } else {
        updatePromptBtn.textContent = 'Failed ❌';
      }
    } catch (e) {
      console.error('Failed to update instructions:', e);
      updatePromptBtn.textContent = 'Error ❌';
    }
    setTimeout(() => {
      updatePromptBtn.disabled = false;
      updatePromptBtn.textContent = 'Update Instructions';
    }, 2000);
  };
}

let resetPromptBtn = document.getElementById('reset-prompt-btn');
if (resetPromptBtn) {
  resetPromptBtn.onclick = async () => {
    resetPromptBtn.disabled = true;
    resetPromptBtn.textContent = 'Resetting...';
    try {
      const resp = await fetch(`/api/config/instructions?agent_name=${encodeURIComponent(selectedAgentName)}`, {
        method: 'DELETE',
      });
      const data = await resp.json();
      if (data.success) {
        resetPromptBtn.textContent = 'Reset ✅';
        setTimeout(() => {
          loadAgentConfig(selectedAgentName, selectedModel);
        }, 1000);
      } else {
        resetPromptBtn.textContent = 'Failed ❌';
      }
    } catch (e) {
      console.error('Failed to reset instructions:', e);
      resetPromptBtn.textContent = 'Error ❌';
    }
    setTimeout(() => {
      resetPromptBtn.disabled = false;
      resetPromptBtn.textContent = 'Reset to Default';
    }, 2000);
  };
}

if (closeModalBtn) {
  closeModalBtn.onclick = () => {
    settingsModal.classList.add('hidden');
  };
}

// Close modal on click outside
window.onclick = (event) => {
  if (event.target === settingsModal) {
    settingsModal.classList.add('hidden');
  }
};

themeRadios.forEach(radio => {
  radio.onchange = (e) => {
    const theme = e.target.value;
    applyTheme(theme);
    localStorage.setItem('theme', theme);
  };
});

function applyTheme(theme) {
  if (theme === 'dark') {
    document.body.classList.add('dark-theme');
  } else {
    document.body.classList.remove('dark-theme');
  }
  const tbtn = document.getElementById('themeToggleBtn');
  if (tbtn) tbtn.title = theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme';
  themeRadios.forEach(r => { r.checked = r.value === theme; });
}

// Header light/dark toggle (same storage as the Settings radios).
{
  const tbtn = document.getElementById('themeToggleBtn');
  if (tbtn) tbtn.onclick = () => {
    const next = document.body.classList.contains('dark-theme') ? 'light' : 'dark';
    applyTheme(next);
    localStorage.setItem('theme', next);
  };
}

// Load theme on startup
let savedTheme = localStorage.getItem('theme') || 'light';
applyTheme(savedTheme);

// Update radio button state
themeRadios.forEach(radio => {
  if (radio.value === savedTheme) {
    radio.checked = true;
  }
});

// --- Thinking level dropdown: persist per model on change ---
{
  const sel = document.getElementById('config-thinking-level');
  if (sel) {
    sel.onchange = () => {
      if (!sel.disabled) saveThinkingLevel(selectedModel, sel.value);
      renderThinkingStatus();
    };
  }
}

// --- Latency scope toggle (USER ⇄ ALL) ---
{
  const scopeBtn = document.getElementById('latencyScopeBtn');
  if (scopeBtn) scopeBtn.onclick = onLatencyScopeClick;
  renderLatencyBar();
  startLatencyScopeRotation();
}

// --- Robot skills map (header button → modal) ---
{
  const modal = document.getElementById('skills-modal');
  const openBtn = document.getElementById('skillsMapBtn');
  const closeBtn = document.getElementById('skillsCloseBtn');
  const close = () => modal && modal.classList.add('hidden');
  if (openBtn && modal) openBtn.onclick = (e) => { e.stopPropagation(); modal.classList.remove('hidden'); };
  if (closeBtn) closeBtn.onclick = close;
  if (modal) modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal && !modal.classList.contains('hidden')) {
      close();
      e.stopImmediatePropagation();
    }
  }, true);
}

// --- Keyboard Shortcuts ---
document.addEventListener('keydown', (e) => {
  // Don't intercept when typing in input fields
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
    if (e.key === 'Escape') {
      e.target.blur();
    }
    return;
  }

  switch (e.key) {
    case ' ':
      e.preventDefault();
      micBtn.click();
      break;
    case 'Escape':
      if (!settingsModal.classList.contains('hidden')) {
        settingsModal.classList.add('hidden');
      } else if (geminiClient.isConnected()) {
        // Send stop command
        geminiClient.sendText('stop');
      }
      break;
    case '1':
      selectAgent('human');
      applyVideoSource('computer');
      break;
    case '2':
      selectAgent('spot');
      applyVideoSource(videoSourceForAgent('spot'));
      break;
    case 't':
    case 'T':
      const currentTheme =
        document.body.classList.contains('dark-theme') ? 'light' : 'dark';
      applyTheme(currentTheme);
      localStorage.setItem('theme', currentTheme);
      themeRadios.forEach(r => {
        r.checked = r.value === currentTheme;
      });
      break;
    case '?':
      const helpModal = document.getElementById('shortcuts-modal');
      if (helpModal) helpModal.classList.toggle('hidden');
      break;
  }
});

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// Internal prompt-coaching fields injected into functionResponse solely to steer
// the ER2 state machine; hidden from the user-facing tool telemetry card.
const INTERNAL_LLM_PROMPT_KEYS = new Set(['next_action', 'note', 'instruction']);

function formatFunctionResult(value, depth = 0) {
  if (value === null || value === undefined) return '';
  if (typeof value !== 'object') return String(value);

  const indent = '  '.repeat(depth);
  if (Array.isArray(value)) {
    return value.map(item => {
      const text = formatFunctionResult(item, depth + 1);
      const lines = text.split('\n');
      return `${indent}- ${lines[0]}${lines.slice(1).map(line => `\n${indent}  ${line}`).join('')}`;
    }).join('\n');
  }

  return Object.entries(value)
    .filter(([key]) => !(depth === 0 && INTERNAL_LLM_PROMPT_KEYS.has(key)))
    .map(([key, item]) => {
      const label = key.replaceAll('_', ' ');
      if (item !== null && typeof item === 'object') {
        return `${indent}${label}:\n${formatFunctionResult(item, depth + 1)}`;
      }
      return `${indent}${label}: ${formatFunctionResult(item)}`;
    }).join('\n');
}

// --- Session Timer (integrated into status label) ---
let sessionStartTime = null;
let sessionTimerInterval = null;

function startSessionTimer() {
  sessionStartTime = Date.now();
  updateSessionTimer();
  sessionTimerInterval = setInterval(updateSessionTimer, 1000);
}

function stopSessionTimer() {
  if (sessionTimerInterval) {
    clearInterval(sessionTimerInterval);
    sessionTimerInterval = null;
  }
  sessionStartTime = null;
}

function updateSessionTimer() {
  if (!sessionStartTime || !statusLabel) return;
  const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
  const h = Math.floor(elapsed / 3600);
  const m = Math.floor((elapsed % 3600) / 60);
  const s = elapsed % 60;
  const pad = n => String(n).padStart(2, '0');
  const time = h > 0 ? `${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
  if (streamState === 'reconnecting') {
    statusLabel.textContent = `🔄 Reconnecting… ${time}`;
    return;
  }
  statusLabel.textContent = `Connected ${time}`;
}

// --- Transparent History Viewer ---

function prettyPrintPartJs(part) {
  if (!part) return '';
  const isThought = part.thought || part.raw_thought || part.rawThought;
  const prefix = isThought ? 'Thought: ' : '';
  if (part.text) {
    return `${prefix}${part.text}`;
  } else if (part.inlineData) {
    const mime = part.inlineData.mimeType || 'unknown';
    const len = (part.inlineData.data || '').length;
    if (mime.startsWith('image/') || mime.includes('jpeg') ||
      mime.includes('png')) {
      return `<div class="memory-part-media"><span class="chip">🖼️ Frame</span><img src="data:image/jpeg;base64,${part.inlineData.data}" class="memory-inline-img" /></div>`;
    }
    return `<span class="chip">📦 Data</span> <code>[InlineData: ${mime} (${len} bytes)]</code>`;
  } else if (part.audioTranscription) {
    return `<span class="chip">🗣️ Transcript</span> <em>"${part.audioTranscription.text}"</em>`;
  } else if (part.audio_transcription) {
    return `<span class="chip">🗣️ Transcript</span> <em>"${part.audio_transcription.text}"</em>`;
  } else if (part.functionCall) {
    const fc = part.functionCall;
    const args = JSON.stringify(fc.args || {});
    return `<span class="chip code-chip">🔧 Call</span> <code>${fc.name}(${args})</code>`;
  } else if (part.functionResponse) {
    const fr = part.functionResponse;
    const resp = escapeHtml(formatFunctionResult(fr.response || {}))
      .replaceAll('\n', '<br>');
    return `<span class="chip code-chip">✅ Result</span> <code>${fr.name} -> ${resp}</code>`;
  } else if (part.fileData) {
    return `<span class="chip">📁 File</span> <code>[FileData: ${part.fileData.mimeType}]</code>`;
  }
  return `<code>[Unknown Part]</code>`;
}



// --- Latency Detail Toggle ---
let latencyDetailToggle = document.getElementById('latencyDetailToggle');
let latencyDetail = document.getElementById('latency-detail');
if (latencyDetailToggle) {
  latencyDetailToggle.onclick = () => {
    latencyDetail.classList.toggle('hidden');
  };
}

// --- Help Guide Floating Popover ---
const helpGuideBtn = document.getElementById('helpGuideBtn');
const helpGuidePopover = document.getElementById('helpGuidePopover');
const helpGuideCloseBtn = document.getElementById('helpGuideCloseBtn');

if (helpGuideBtn && helpGuidePopover) {
  helpGuideBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    helpGuidePopover.classList.toggle('hidden');
  });

  if (helpGuideCloseBtn) {
    helpGuideCloseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      helpGuidePopover.classList.add('hidden');
    });
  }

  helpGuidePopover.addEventListener('click', (e) => {
    e.stopPropagation();
  });

  document.querySelectorAll('.help-prompt-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const promptText = chip.getAttribute('data-prompt') || '';
      if (textInput && promptText) {
        if (chatSidebar && chatSidebar.classList.contains('collapsed')) {
          chatSidebar.classList.remove('collapsed');
        }
        textInput.value = promptText;
        textInput.focus();
      }
      helpGuidePopover.classList.add('hidden');
    });
  });

  document.addEventListener('click', (e) => {
    if (!helpGuidePopover.classList.contains('hidden') && !helpGuidePopover.contains(e.target) && e.target !== helpGuideBtn) {
      helpGuidePopover.classList.add('hidden');
    }
  });
}

// --- Rich cards in the chat: maps + place photos ---------------------------
//
// Rendered from the results of the information tools, so what the user sees
// is exactly what the tool returned (no model-generated images):
//   google_maps_search   -> map card with a pin per place (chips to switch)
//   get_current_location -> map card pinned at the device location
//   get_place_photos     -> photo strip with author attributions + map link
function _mapsEmbedUrl(q, zoom) {
  if (!mapsEmbedKey) return '';
  const lang = (navigator.language || 'en').slice(0, 5);
  let u = 'https://www.google.com/maps/embed/v1/place?key=' + encodeURIComponent(mapsEmbedKey) +
    '&q=' + encodeURIComponent(q) + '&language=' + encodeURIComponent(lang);
  if (zoom) u += '&zoom=' + zoom;
  return u;
}

function _appendCard(el) {
  const isAtBottom = chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 80;
  chatLog.appendChild(el);
  if (isAtBottom) chatLog.scrollTop = chatLog.scrollHeight;
  // Images / iframes load after insertion; keep the log pinned to the bottom.
  el.querySelectorAll('img, iframe').forEach(m => m.addEventListener('load', () => {
    if (chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 400) chatLog.scrollTop = chatLog.scrollHeight;
  }, { once: true }));
}

function _el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function appendMapCard(title, places, centerQuery) {
  const card = _el('div', 'rich-card map-card');
  const head = _el('div', 'rich-head');
  head.appendChild(_el('span', 'rich-icon', '🗺️'));
  head.appendChild(_el('span', 'rich-title', title));
  card.appendChild(head);
  const first = places[0];
  const firstQ = first ? (first.place_id ? 'place_id:' + first.place_id : first.q) : centerQuery;
  const url = _mapsEmbedUrl(firstQ, first && first.zoom);
  let frame = null;
  if (url) {
    frame = document.createElement('iframe');
    frame.className = 'map-frame';
    frame.loading = 'lazy';
    frame.referrerPolicy = 'no-referrer-when-downgrade';
    frame.allowFullscreen = true;
    frame.src = url;
    card.appendChild(frame);
  }
  if (places.length) {
    const chips = _el('div', 'place-chips');
    places.forEach((p, i) => {
      const chip = _el('button', 'place-chip' + (i === 0 ? ' active' : ''), '📍 ' + (p.name || p.q));
      chip.title = 'Show on map';
      chip.onclick = () => {
        chips.querySelectorAll('.place-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        if (frame) frame.src = _mapsEmbedUrl(p.place_id ? 'place_id:' + p.place_id : p.q, p.zoom);
        else if (p.uri) window.open(p.uri, '_blank', 'noopener');
      };
      chips.appendChild(chip);
    });
    card.appendChild(chips);
  }
  const link = first && (first.uri || (first.place_id ? 'https://www.google.com/maps/place/?q=place_id:' + first.place_id : ''));
  if (link) {
    const a = _el('a', 'rich-link', 'Open in Google Maps ↗');
    a.href = link; a.target = '_blank'; a.rel = 'noopener';
    card.appendChild(a);
  }
  _appendCard(card);
}

function appendPhotoCard(r) {
  const card = _el('div', 'rich-card photo-card');
  const head = _el('div', 'rich-head');
  head.appendChild(_el('span', 'rich-icon', '📷'));
  head.appendChild(_el('span', 'rich-title', r.name || 'Photos'));
  if (r.rating) head.appendChild(_el('span', 'rich-meta', `★ ${r.rating}${r.rating_count ? ' (' + r.rating_count + ')' : ''}`));
  card.appendChild(head);
  if (r.address) card.appendChild(_el('div', 'rich-sub', r.address));
  const strip = _el('div', 'photo-strip');
  (r.photos || []).forEach(ph => {
    const fig = _el('figure', 'photo');
    const a = document.createElement('a');
    a.href = ph.uri; a.target = '_blank'; a.rel = 'noopener';
    const img = document.createElement('img');
    img.src = ph.uri; img.loading = 'lazy'; img.alt = r.name || 'place photo';
    a.appendChild(img); fig.appendChild(a);
    const att = (ph.attributions || [])[0];
    if (att && att.name) {
      const cap = _el('figcaption', 'photo-attr');
      const al = _el('a', '', '© ' + att.name);
      if (att.uri) { al.href = att.uri; al.target = '_blank'; al.rel = 'noopener'; }
      cap.appendChild(al); fig.appendChild(cap);
    }
    strip.appendChild(fig);
  });
  card.appendChild(strip);
  if (r.google_maps_url) {
    const a = _el('a', 'rich-link', 'Open in Google Maps ↗');
    a.href = r.google_maps_url; a.target = '_blank'; a.rel = 'noopener';
    card.appendChild(a);
  }
  _appendCard(card);
}

function renderRichToolCard(msg) {
  const r = (msg && msg.result && typeof msg.result === 'object') ? msg.result : null;
  if (!r || r.error) return;
  try {
    if (msg.name === 'google_maps_search' && Array.isArray(r.places) && r.places.length) {
      appendMapCard(r.query || 'Map', r.places, '');
    } else if (msg.name === 'get_current_location' && r.available && r.latitude != null) {
      const q = `${r.latitude},${r.longitude}`;
      appendMapCard('📍 You are here', [{ q, name: r.place_description ? r.place_description.split(/[。.,，]/)[0] : q, zoom: 16, uri: r.google_maps_url }], q);
    } else if (msg.name === 'get_place_photos' && Array.isArray(r.photos) && r.photos.length) {
      appendPhotoCard(r);
    }
  } catch (e) {
    console.warn('rich card failed', e);
  }
}

// --- Tools map: hover tips for each tool name ------------------------------
// Data = <script id="tool-tips"> generated from the real function
// declarations by agent/scripts/gen_tools_map.py.
{
  let tips = {};
  try { tips = JSON.parse((document.getElementById('tool-tips') || {}).textContent || '{}'); } catch (e) { tips = {}; }
  const tipEl = document.createElement('div');
  tipEl.className = 'tool-tip hidden';
  document.body.appendChild(tipEl);
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  function show(name, x, y) {
    const t = tips[name];
    if (!t) return;
    const params = (t.params || []).map(p =>
      `<li><code>${esc(p.name)}</code><span class="tt-type">${esc(p.type)}${p.required ? ' · required' : ''}</span>` +
      ((p.enum && p.enum.length) ? `<div class="tt-enum">${p.enum.map(v => `<code>${esc(v)}</code>`).join(' ')}</div>` : '') +
      (p.desc ? `<div class="tt-pdesc">${esc(p.desc)}</div>` : '') + '</li>').join('');
    tipEl.innerHTML = `<div class="tt-name">${esc(name)}()</div><div class="tt-desc">${esc(t.desc)}</div>` +
      (params ? `<div class="tt-h">Parameters</div><ul class="tt-params">${params}</ul>` : '<div class="tt-h">No parameters</div>');
    tipEl.classList.remove('hidden');
    const r = tipEl.getBoundingClientRect();
    let left = x + 14, top = y + 16;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = Math.max(8, y - r.height - 12);
    tipEl.style.left = left + 'px'; tipEl.style.top = top + 'px';
  }
  document.addEventListener('mouseover', (e) => {
    const el = e.target.closest && e.target.closest('.fc-tool');
    if (el) show(el.dataset.tool, e.clientX, e.clientY);
  });
  document.addEventListener('mousemove', (e) => {
    if (!tipEl.classList.contains('hidden') && e.target.closest && e.target.closest('.fc-tool')) show(e.target.closest('.fc-tool').dataset.tool, e.clientX, e.clientY);
  });
  document.addEventListener('mouseout', (e) => {
    if (e.target.closest && e.target.closest('.fc-tool')) tipEl.classList.add('hidden');
  });
}
