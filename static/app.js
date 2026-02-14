const form = document.getElementById('generatorForm');
const controlsPanel = document.querySelector('.controls');
const providerSelect = document.getElementById('provider');
const sizeSelect = document.getElementById('size');
const qualitySelect = document.getElementById('quality');
const ratioSelect = document.getElementById('ratio');
const descriptionInput = document.getElementById('description');
const referenceInput = document.getElementById('referenceImages');
const previewImage = document.getElementById('previewImage');
const emptyPreview = document.getElementById('emptyPreview');
const progressOverlay = document.getElementById('progressOverlay');
const status = document.getElementById('status');
const referenceMeta = document.getElementById('referenceMeta');
const referenceGallery = document.getElementById('referenceGallery');
const generateButton = document.getElementById('generateButton');
const aiNarrationToggle = document.getElementById('aiNarrationToggle');
const downloadButton = document.getElementById('downloadButton');
const cancelButton = document.getElementById('cancelButton');
const fullSizeButton = document.getElementById('fullSizeButton');
const previewBar = document.getElementById('previewBar');
const imageHint = document.getElementById('imageHint');
const aiReply = document.getElementById('aiReply');
const aiReplyRow = document.getElementById('aiReplyRow');
const aiReplySpeakButton = document.getElementById('aiReplySpeakButton');
const clearRefsButton = document.getElementById('clearRefsButton');
const voiceButton = document.getElementById('voiceButton');
const openAiVoiceButton = document.getElementById('openAiVoiceButton');
const cameraButton = document.getElementById('cameraButton');
const cameraInput = document.getElementById('cameraInput');
const voicePopup = document.getElementById('voicePopup');
const voiceCancelButton = document.getElementById('voiceCancelButton');
const voicePopupNote = document.getElementById('voicePopupNote');
const voiceWaveCanvas = document.getElementById('voiceWaveCanvas');
const voiceBandsCanvas = document.getElementById('voiceBandsCanvas');
const voiceLoudnessCanvas = document.getElementById('voiceLoudnessCanvas');

const lightbox = document.getElementById('lightbox');
const lightboxTitle = document.getElementById('lightboxTitle');
const lightboxCanvas = document.getElementById('lightboxCanvas');
const lightboxImage = document.getElementById('lightboxImage');
const zoomOutButton = document.getElementById('zoomOutButton');
const zoomResetButton = document.getElementById('zoomResetButton');
const zoomInButton = document.getElementById('zoomInButton');
const lightboxPrevButton = document.getElementById('lightboxPrevButton');
const lightboxNextButton = document.getElementById('lightboxNextButton');
const lightboxCloseButton = document.getElementById('lightboxCloseButton');
const lightboxDownloadButton = document.getElementById('lightboxDownloadButton');

const dragHandle = document.getElementById('dragHandle');
const windowShell = document.getElementById('windowShell');

const GENERATION_TIMEOUT_MS = 120000;

let providers = {};

const fallbackFilenameFromDescription = (description) => {
  const fallback = description
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .trim()
    .replace(/[-\s]+/g, '-')
    .slice(0, 80);
  return `${fallback || 'generated-image'}.png`;
};
let isGenerating = false;
let generatedImageDataUrl = '';
let generatedFilename = 'generated-image.png';
let aiNarrationEnabled = true;
let lastAiReplyText = '';
let lastAiReplyLanguage = 'en-US';
let aiReplyAudioUrl = '';
let aiReplyAudio = null;
let referenceItems = [];
let selectedReferenceIndex = -1;
let nextReferenceId = 1;
let lightboxMode = null;
let lightboxIndex = -1;
let lightboxScale = 1;
let generationAbortController = null;
let lightboxPanState = { active: false, startX: 0, startY: 0, scrollLeft: 0, scrollTop: 0 };
let lightboxItems = [];
let speechRecognition = null;
let isRecording = false;
let openAiRecorder = null;
let openAiStream = null;
let openAiChunks = [];
let openAiBlob = null;
let openAiElapsedTimerId = null;
let openAiSilenceTimerId = null;
let openAiAudioContext = null;
let openAiAnalyser = null;
let openAiSourceNode = null;
let openAiRecordingStartedAt = 0;
let openAiLastVoiceAt = 0;
let openAiVisualRafId = null;
let openAiLoudnessHistory = [];
let openAiTranscribeAbortController = null;

const OPENAI_VOICE_SILENCE_MS = 2500;
const OPENAI_VOICE_SILENCE_THRESHOLD = 0.012;
const OPENAI_VOICE_MAX_SECONDS = 90;
const OPENAI_LOUDNESS_HISTORY_SIZE = 160;

const setStatus = (text) => {
  status.textContent = text;
};

const setAiNarrationEnabled = (enabled) => {
  aiNarrationEnabled = Boolean(enabled);
  aiNarrationToggle.setAttribute('aria-pressed', aiNarrationEnabled ? 'true' : 'false');
  aiNarrationToggle.title = aiNarrationEnabled ? 'AI voice on' : 'AI voice off';
  if (!aiNarrationEnabled && aiReplyAudio) {
    aiReplyAudio.pause();
  }
  if (!aiNarrationEnabled && 'speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
};

const inferSpeechLanguageFromText = (text) => {
  const normalized = (text || '').toLowerCase();
  if (/[äöüß]/.test(normalized) || /\b(und|oder|der|die|das|eine|einer|mit|auf|im|ist)\b/.test(normalized)) {
    return 'de-DE';
  }
  return 'en-US';
};

const clearCachedAiReplyAudio = () => {
  if (aiReplyAudio) {
    aiReplyAudio.pause();
    aiReplyAudio.src = '';
    aiReplyAudio = null;
  }
  if (aiReplyAudioUrl) {
    URL.revokeObjectURL(aiReplyAudioUrl);
    aiReplyAudioUrl = '';
  }
};

const requestOpenAiNarrationAudio = async (text, language) => {
  const response = await fetch('/api/tts-openai', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, language }),
  });

  if (!response.ok) {
    let payload = {};
    try {
      payload = await response.json();
    } catch {
      payload = {};
    }
    if (response.status === 404) {
      throw new Error('TTS endpoint not found (404). Restart FastAPI.');
    }
    throw new Error(payload.detail || 'OpenAI TTS failed');
  }

  return response.blob();
};

const playAndCacheAiReplyAudio = async (audioBlob) => {
  if (!audioBlob || !audioBlob.size) {
    throw new Error('No narration audio returned');
  }
  clearCachedAiReplyAudio();
  aiReplyAudioUrl = URL.createObjectURL(audioBlob);
  aiReplyAudio = new Audio(aiReplyAudioUrl);
  aiReplyAudio.preload = 'auto';
  await aiReplyAudio.play();
};

const replayCachedAiReplyAudio = async () => {
  if (!aiReplyAudio) {
    return false;
  }
  aiReplyAudio.currentTime = 0;
  await aiReplyAudio.play();
  return true;
};

const speakTextInLanguage = async (text, language, force = false) => {
  if ((!aiNarrationEnabled && !force) || !text) {
    return;
  }

  const normalizedLang = (language || inferSpeechLanguageFromText(text) || 'en-US').toLowerCase();

  try {
    const audioBlob = await requestOpenAiNarrationAudio(text, normalizedLang);
    await playAndCacheAiReplyAudio(audioBlob);
    return;
  } catch (error) {
    console.warn('OpenAI TTS failed, falling back to browser speech:', error);
  }

  if (!('speechSynthesis' in window)) {
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = normalizedLang;

  const applyBestVoice = () => {
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return;
    const exact = voices.find((v) => v.lang.toLowerCase() === normalizedLang);
    const family = voices.find((v) => v.lang.toLowerCase().startsWith(normalizedLang.split('-')[0]));
    utterance.voice = exact || family || null;
  };

  applyBestVoice();
  if (!utterance.voice) {
    await new Promise((resolve) => {
      const onVoices = () => {
        applyBestVoice();
        window.speechSynthesis.removeEventListener('voiceschanged', onVoices);
        resolve();
      };
      window.speechSynthesis.addEventListener('voiceschanged', onVoices, { once: true });
      setTimeout(resolve, 250);
    });
  }

  window.speechSynthesis.resume();
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
};

const describeGeneratedImage = async (imageDataUrl, sourceText, language = '') => {
  const response = await fetch('/api/describe-image', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_data_url: imageDataUrl, source_text: sourceText, language }),
  });
  let payload = {};
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok) {
    if (response.status === 404) {
      throw new Error('AI narration endpoint not found (404). Restart the FastAPI server.');
    }
    throw new Error(payload.detail || 'Description failed');
  }
  return (payload.description || '').trim();
};

const toOptionMarkup = (value) => {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = value;
  return option;
};

const setGenerating = (value) => {
  isGenerating = value;
  controlsPanel.classList.toggle('locked', value);
  generateButton.disabled = value;
  progressOverlay.hidden = !value;
  if (!value) {
    generationAbortController = null;
  }
};

const fetchWithTimeout = async (url, options, timeoutMs) => {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error(`Generation timed out after ${Math.round(timeoutMs / 1000)}s.`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
};

const updateProviderOptions = () => {
  const provider = providers[providerSelect.value];
  if (!provider) {
    return;
  }

  [sizeSelect, qualitySelect, ratioSelect].forEach((el) => {
    el.innerHTML = '';
  });

  provider.sizes.forEach((item) => sizeSelect.appendChild(toOptionMarkup(item)));
  provider.qualities.forEach((item) => qualitySelect.appendChild(toOptionMarkup(item)));
  provider.ratios.forEach((item) => ratioSelect.appendChild(toOptionMarkup(item)));
};

const updateReferenceActions = () => {
  const hasItems = referenceItems.length > 0;
  clearRefsButton.hidden = !hasItems;

  if (!hasItems) {
    referenceMeta.textContent = '';
    return;
  }

  referenceMeta.textContent = `(${referenceItems.length})`;
};

const selectReference = (index) => {
  if (!referenceItems.length) {
    selectedReferenceIndex = -1;
    renderReferenceGallery();
    updateReferenceActions();
    return;
  }

  const bounded = ((index % referenceItems.length) + referenceItems.length) % referenceItems.length;
  selectedReferenceIndex = bounded;
  renderReferenceGallery();
  updateReferenceActions();
};

const renderReferenceGallery = () => {
  referenceGallery.innerHTML = '';

  if (!referenceItems.length) {
    updateReferenceActions();
    return;
  }

  referenceItems.forEach((item, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `ref-thumb ${index === selectedReferenceIndex ? 'is-selected' : ''}`;
    button.title = item.file.name;

    const img = document.createElement('img');
    img.src = item.url;
    img.alt = item.file.name;

    const closeBtn = document.createElement('span');
    closeBtn.className = 'ref-thumb-close';
    closeBtn.textContent = '\u00d7';
    closeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      removeReferenceAt(index);
    });

    button.appendChild(img);
    button.appendChild(closeBtn);
    button.addEventListener('click', () => selectReference(index));
    button.addEventListener('dblclick', () => openLightbox('references', index));
    referenceGallery.appendChild(button);
  });

  updateReferenceActions();
};

const addReferenceFiles = (files, selectLast = true) => {
  files.forEach((file) => {
    referenceItems.push({
      id: nextReferenceId,
      file,
      url: URL.createObjectURL(file),
    });
    nextReferenceId += 1;
  });

  if (selectLast && referenceItems.length) {
    selectedReferenceIndex = referenceItems.length - 1;
  }

  renderReferenceGallery();
};

const removeReferenceAt = (index) => {
  if (index < 0 || index >= referenceItems.length) {
    return;
  }

  const [removed] = referenceItems.splice(index, 1);
  URL.revokeObjectURL(removed.url);

  if (!referenceItems.length) {
    selectedReferenceIndex = -1;
  } else {
    selectedReferenceIndex = Math.min(index, referenceItems.length - 1);
  }

  renderReferenceGallery();
};

const clearReferences = () => {
  referenceItems.forEach((item) => URL.revokeObjectURL(item.url));
  referenceItems = [];
  selectedReferenceIndex = -1;
  renderReferenceGallery();
};

const dataUrlToFile = async (dataUrl, fileName) => {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  return new File([blob], fileName, { type: blob.type || 'image/png' });
};

const suggestFilename = async (description) => {
  try {
    const response = await fetch('/api/suggest-filename', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description }),
    });

    if (!response.ok) {
      throw new Error('Filename API failed');
    }

    const payload = await response.json();
    const stem = (payload.filename || 'generated-image').trim() || 'generated-image';
    return `${stem}.png`;
  } catch {
    const fallback = description
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .trim()
      .replace(/[-\s]+/g, '-')
      .slice(0, 80);
    return `${fallback || 'generated-image'}.png`;
  }
};

const setGeneratedPreview = (imageDataUrl) => {
  generatedImageDataUrl = imageDataUrl;
  previewImage.src = imageDataUrl;
  previewImage.hidden = false;
  emptyPreview.hidden = true;
  fullSizeButton.hidden = false;
  previewBar.hidden = false;
};

const setImageHint = (text) => {
  imageHint.textContent = text;
  imageHint.title = text;
};

const buildLightboxItems = () => {
  const items = [];
  if (generatedImageDataUrl) {
    items.push({ url: generatedImageDataUrl, label: `Generated • ${generatedFilename}`, filename: generatedFilename });
  }
  const refs = generatedImageDataUrl
    ? referenceItems.filter((item) => item.file.name !== generatedFilename)
    : referenceItems;
  refs.forEach((item, i) => {
    items.push({ url: item.url, label: `Reference ${i + 1}/${refs.length} • ${item.file.name}`, filename: item.file.name });
  });
  return items;
};

const showLightboxItem = (index) => {
  const item = lightboxItems[index];
  if (!item) return;
  lightboxIndex = index;
  lightboxTitle.textContent = item.label;
  lightboxImage.src = item.url;
  const multi = lightboxItems.length > 1;
  lightboxPrevButton.disabled = !multi;
  lightboxNextButton.disabled = !multi;
};

const openLightbox = (mode, index = -1) => {
  lightboxItems = buildLightboxItems();
  if (!lightboxItems.length) return;

  let startIndex = 0;
  if (mode === 'generated') {
    startIndex = 0;
  } else {
    const refIdx = index >= 0 ? index : selectedReferenceIndex;
    const offset = generatedImageDataUrl ? 1 : 0;
    startIndex = offset + Math.max(0, refIdx);
  }

  lightboxScale = 1;
  lightboxImage.style.transform = 'scale(1)';
  zoomResetButton.textContent = '100%';
  lightbox.hidden = false;
  showLightboxItem(startIndex);
};

const closeLightbox = () => {
  lightbox.hidden = true;
};

const zoomLightbox = (delta) => {
  lightboxScale = Math.min(4, Math.max(0.25, lightboxScale + delta));
  lightboxImage.style.transform = `scale(${lightboxScale})`;
  zoomResetButton.textContent = `${Math.round(lightboxScale * 100)}%`;
};

const cycleLightbox = (step) => {
  if (lightboxItems.length <= 1) return;
  const next = ((lightboxIndex + step) % lightboxItems.length + lightboxItems.length) % lightboxItems.length;
  showLightboxItem(next);
};

const generateImage = async () => {
  if (isGenerating) {
    return;
  }

  const description = descriptionInput.value.trim();
  if (!description) {
    setStatus('Description is required.');
    return;
  }

  clearCachedAiReplyAudio();
  lastAiReplyText = '';
  lastAiReplyLanguage = 'en-US';
  aiReply.textContent = '';
  aiReplyRow.hidden = true;

  generationAbortController = new AbortController();
  setGenerating(true);
  setStatus('Generating...');

  const filenamePromise = suggestFilename(description);

  try {
    const fd = new FormData();
    fd.append('provider', providerSelect.value);
    fd.append('description', description);
    fd.append('size', sizeSelect.value);
    fd.append('quality', qualitySelect.value);
    fd.append('ratio', ratioSelect.value);
    referenceItems.forEach((item) => fd.append('reference_images', item.file));

    const controller = generationAbortController;
    const timeoutId = window.setTimeout(() => controller.abort(), GENERATION_TIMEOUT_MS);
    let response;
    try {
      response = await fetch('/api/generate', {
        method: 'POST',
        body: fd,
        signal: controller.signal,
      });
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error('Generation cancelled.');
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Generation failed');
    }

    generatedFilename = fallbackFilenameFromDescription(description);
    const suggestedFilename = await Promise.race([
      filenamePromise,
      new Promise((resolve) => window.setTimeout(() => resolve(generatedFilename), 2500)),
    ]);
    generatedFilename = suggestedFilename || generatedFilename;
    setGeneratedPreview(data.image_data_url);
    setImageHint(`${description.slice(0, 120)}${description.length > 120 ? '...' : ''} — ${data.provider}, ${data.size}`);

    const generatedFile = await dataUrlToFile(data.image_data_url, generatedFilename);
    addReferenceFiles([generatedFile], true);

    let narrationErrorMessage = '';
    const language = inferSpeechLanguageFromText(description);
    try {
      const reply = await describeGeneratedImage(data.image_data_url, description);
      if (reply) {
        aiReply.textContent = reply;
        aiReplyRow.hidden = false;
        lastAiReplyText = reply;
        lastAiReplyLanguage = language;
        await speakTextInLanguage(reply, language);
      }
    } catch (describeError) {
      narrationErrorMessage = describeError.message || 'AI narration unavailable.';
      console.warn('Describe image failed:', describeError);
    }

    const baseDoneStatus = `Done — ${data.provider}, ${data.used_reference_images} ref(s) used.`;
    setStatus(narrationErrorMessage ? `${baseDoneStatus} ${narrationErrorMessage}` : baseDoneStatus);
  } catch (error) {
    setStatus(error.message || 'Generation failed');
  } finally {
    setGenerating(false);
  }
};

referenceInput.addEventListener('change', () => {
  const files = Array.from(referenceInput.files || []);
  if (files.length) {
    addReferenceFiles(files, true);
  }
  referenceInput.value = '';
});

providerSelect.addEventListener('change', updateProviderOptions);

form.addEventListener('submit', (event) => {
  event.preventDefault();
  generateImage();
});

aiNarrationToggle.addEventListener('click', () => {
  setAiNarrationEnabled(!aiNarrationEnabled);
});

aiReplySpeakButton.addEventListener('click', () => {
  const replayText = (lastAiReplyText || aiReply.textContent || '').trim();
  if (!replayText) {
    setStatus('No AI summary available to replay yet.');
    return;
  }
  const replayLanguage = lastAiReplyLanguage || inferSpeechLanguageFromText(replayText);
  replayCachedAiReplyAudio()
    .catch(() => false)
    .then((playedFromCache) => {
      if (!playedFromCache) {
        return speakTextInLanguage(replayText, replayLanguage, true);
      }
      return undefined;
    })
    .catch((error) => {
      setStatus(error.message || 'Replay failed.');
    });
});

previewImage.addEventListener('click', () => openLightbox('generated'));

downloadButton.addEventListener('click', () => {
  if (!generatedImageDataUrl) {
    return;
  }
  const link = document.createElement('a');
  link.href = generatedImageDataUrl;
  link.download = generatedFilename;
  link.click();
});

clearRefsButton.addEventListener('click', () => clearReferences());

const isTouchDevice = () => 'ontouchstart' in window || navigator.maxTouchPoints > 0;

const captureFromWebcam = async () => {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
  } catch (err) {
    setStatus(`Camera access denied: ${err.message}`);
    return;
  }

  const video = document.createElement('video');
  video.srcObject = stream;
  video.setAttribute('playsinline', '');
  video.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;object-fit:cover;z-index:50;background:#000;';

  const shutterBtn = document.createElement('button');
  shutterBtn.textContent = 'Capture';
  shutterBtn.style.cssText = 'position:fixed;bottom:32px;left:50%;transform:translateX(-50%);z-index:51;padding:14px 36px;border-radius:50px;background:#fff;color:#222;font-size:1.1rem;font-weight:700;border:none;cursor:pointer;box-shadow:0 4px 20px rgba(0,0,0,0.4);';

  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = '✕';
  cancelBtn.style.cssText = 'position:fixed;top:16px;right:16px;z-index:51;width:40px;height:40px;border-radius:50%;background:rgba(0,0,0,0.5);color:#fff;font-size:1.2rem;border:none;cursor:pointer;';

  document.body.appendChild(video);
  document.body.appendChild(shutterBtn);
  document.body.appendChild(cancelBtn);
  await video.play();

  const cleanup = () => {
    stream.getTracks().forEach((t) => t.stop());
    video.remove();
    shutterBtn.remove();
    cancelBtn.remove();
  };

  cancelBtn.addEventListener('click', cleanup);

  shutterBtn.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    cleanup();
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `camera-${Date.now()}.png`, { type: 'image/png' });
        addReferenceFiles([file], true);
        setStatus('Photo captured and added to references.');
      }
    }, 'image/png');
  });
};

cameraButton.addEventListener('click', () => {
  if (isTouchDevice()) {
    cameraInput.click();
  } else {
    captureFromWebcam();
  }
});

cameraInput.addEventListener('change', () => {
  const files = Array.from(cameraInput.files || []);
  if (files.length) {
    addReferenceFiles(files, true);
  }
  cameraInput.value = '';
});

const startVoice = () => {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    setStatus('Speech recognition not supported in this browser.');
    return;
  }

  if (isRecording && speechRecognition) {
    speechRecognition.stop();
    return;
  }

  speechRecognition = new SpeechRecognition();
  speechRecognition.continuous = true;
  speechRecognition.interimResults = true;
  speechRecognition.lang = navigator.language || 'en-US';

  const existingText = descriptionInput.value;
  let finalTranscript = '';

  speechRecognition.onstart = () => {
    isRecording = true;
    voiceButton.classList.add('recording');
    voiceButton.title = 'Stop dictation';
    setStatus('Listening... tap mic again to stop.');
  };

  speechRecognition.onresult = (event) => {
    let interim = '';
    finalTranscript = '';
    for (let i = 0; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += transcript;
      } else {
        interim += transcript;
      }
    }
    const separator = existingText && !existingText.endsWith(' ') ? ' ' : '';
    descriptionInput.value = existingText + separator + finalTranscript + interim;
  };

  speechRecognition.onerror = (event) => {
    if (event.error !== 'aborted') {
      setStatus(`Voice error: ${event.error}`);
    }
    stopVoice();
  };

  speechRecognition.onend = () => {
    const separator = existingText && !existingText.endsWith(' ') ? ' ' : '';
    if (finalTranscript) {
      descriptionInput.value = existingText + separator + finalTranscript;
    }
    stopVoice();
  };

  speechRecognition.start();
};

const stopVoice = () => {
  isRecording = false;
  voiceButton.classList.remove('recording');
  voiceButton.title = 'Dictate with microphone';
  speechRecognition = null;
  if (status.textContent.startsWith('Listening')) {
    setStatus('Ready.');
  }
};

voiceButton.addEventListener('click', startVoice);

const formatVoiceTime = (seconds) => {
  const clamped = Math.max(0, seconds);
  const mm = String(Math.floor(clamped / 60)).padStart(2, '0');
  const ss = String(clamped % 60).padStart(2, '0');
  return `${mm}:${ss}`;
};

const ensureCanvasSize = (canvas) => {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { dpr, width, height };
};

const drawVoiceVisuals = (pcm, freq, rms) => {
  const waveCtx = voiceWaveCanvas.getContext('2d');
  const bandsCtx = voiceBandsCanvas.getContext('2d');
  const loudCtx = voiceLoudnessCanvas.getContext('2d');
  if (!waveCtx || !bandsCtx || !loudCtx) return;

  const { width: ww, height: wh } = ensureCanvasSize(voiceWaveCanvas);
  const { width: bw, height: bh } = ensureCanvasSize(voiceBandsCanvas);
  const { width: lw, height: lh } = ensureCanvasSize(voiceLoudnessCanvas);

  waveCtx.clearRect(0, 0, ww, wh);
  waveCtx.strokeStyle = 'rgba(255,255,255,0.20)';
  waveCtx.lineWidth = 1;
  waveCtx.beginPath();
  waveCtx.moveTo(0, wh * 0.5);
  waveCtx.lineTo(ww, wh * 0.5);
  waveCtx.stroke();

  waveCtx.strokeStyle = '#f0f2f8';
  waveCtx.lineWidth = 2;
  waveCtx.beginPath();
  for (let i = 0; i < pcm.length; i += 1) {
    const x = (i / (pcm.length - 1)) * ww;
    const y = (pcm[i] / 255) * wh;
    if (i === 0) waveCtx.moveTo(x, y);
    else waveCtx.lineTo(x, y);
  }
  waveCtx.stroke();

  const barCount = 28;
  const barGap = 3;
  const barWidth = Math.max(2, (bw - barGap * (barCount - 1)) / barCount);
  bandsCtx.clearRect(0, 0, bw, bh);
  for (let i = 0; i < barCount; i += 1) {
    const idx = Math.floor((i / barCount) * freq.length);
    const v = (freq[idx] || 0) / 255;
    const h = Math.max(2, v * (bh - 4));
    const x = i * (barWidth + barGap);
    const y = bh - h;
    bandsCtx.fillStyle = `rgba(255,255,255,${0.2 + v * 0.75})`;
    bandsCtx.fillRect(x, y, barWidth, h);
  }

  openAiLoudnessHistory.push(Math.min(1, rms * 4.8));
  if (openAiLoudnessHistory.length > OPENAI_LOUDNESS_HISTORY_SIZE) {
    openAiLoudnessHistory.shift();
  }
  loudCtx.clearRect(0, 0, lw, lh);
  loudCtx.strokeStyle = 'rgba(255,255,255,0.28)';
  loudCtx.lineWidth = 1;
  loudCtx.beginPath();
  loudCtx.moveTo(0, lh - 1);
  loudCtx.lineTo(lw, lh - 1);
  loudCtx.stroke();

  loudCtx.strokeStyle = '#ffd878';
  loudCtx.lineWidth = 2;
  loudCtx.beginPath();
  openAiLoudnessHistory.forEach((value, i) => {
    const x = (i / Math.max(1, openAiLoudnessHistory.length - 1)) * lw;
    const y = lh - value * (lh - 4) - 2;
    if (i === 0) loudCtx.moveTo(x, y);
    else loudCtx.lineTo(x, y);
  });
  loudCtx.stroke();
};

const stopOpenAiStream = () => {
  if (openAiStream) {
    openAiStream.getTracks().forEach((track) => track.stop());
    openAiStream = null;
  }
  if (openAiSourceNode) {
    openAiSourceNode.disconnect();
    openAiSourceNode = null;
  }
  if (openAiAnalyser) {
    openAiAnalyser.disconnect();
    openAiAnalyser = null;
  }
  if (openAiAudioContext) {
    openAiAudioContext.close().catch(() => undefined);
    openAiAudioContext = null;
  }
  if (openAiVisualRafId) {
    cancelAnimationFrame(openAiVisualRafId);
    openAiVisualRafId = null;
  }
  openAiLoudnessHistory = Array(OPENAI_LOUDNESS_HISTORY_SIZE).fill(0);
};

const clearOpenAiTimers = () => {
  if (openAiElapsedTimerId) {
    clearInterval(openAiElapsedTimerId);
    openAiElapsedTimerId = null;
  }
  if (openAiSilenceTimerId) {
    clearInterval(openAiSilenceTimerId);
    openAiSilenceTimerId = null;
  }
};

const closeOpenAiVoicePopup = () => {
  if (openAiRecorder && openAiRecorder.state !== 'inactive') {
    openAiRecorder.stop();
  }
  if (openAiTranscribeAbortController) {
    openAiTranscribeAbortController.abort();
    openAiTranscribeAbortController = null;
  }
  openAiRecorder = null;
  clearOpenAiTimers();
  stopOpenAiStream();
  openAiChunks = [];
  openAiBlob = null;
  openAiRecordingStartedAt = 0;
  openAiLastVoiceAt = 0;
  voicePopupNote.textContent = 'Listening...';
  voicePopup.hidden = true;
};

const stopOpenAiRecording = () => {
  if (openAiRecorder && openAiRecorder.state !== 'inactive') {
    openAiRecorder.stop();
  }
  clearOpenAiTimers();
};

const startOpenAiRecording = async () => {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    voicePopupNote.textContent = 'This browser does not support microphone recording.';
    return;
  }

  openAiChunks = [];
  openAiBlob = null;
  openAiLoudnessHistory = Array(OPENAI_LOUDNESS_HISTORY_SIZE).fill(0);

  try {
    openAiStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    voicePopupNote.textContent = `Microphone access failed: ${error.message}`;
    return;
  }

  const preferredMimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus'
    : MediaRecorder.isTypeSupported('audio/webm')
      ? 'audio/webm'
      : '';

  openAiRecorder = preferredMimeType
    ? new MediaRecorder(openAiStream, { mimeType: preferredMimeType })
    : new MediaRecorder(openAiStream);

  openAiAudioContext = new (window.AudioContext || window.webkitAudioContext)();
  openAiSourceNode = openAiAudioContext.createMediaStreamSource(openAiStream);
  openAiAnalyser = openAiAudioContext.createAnalyser();
  openAiAnalyser.fftSize = 2048;
  openAiSourceNode.connect(openAiAnalyser);
  const pcm = new Uint8Array(openAiAnalyser.fftSize);
  const freq = new Uint8Array(openAiAnalyser.frequencyBinCount);

  openAiRecorder.addEventListener('dataavailable', (event) => {
    if (event.data && event.data.size > 0) {
      openAiChunks.push(event.data);
    }
  });

  openAiRecorder.addEventListener('stop', () => {
    clearOpenAiTimers();
    stopOpenAiStream();
    const recordedSeconds = Math.max(1, Math.round((Date.now() - openAiRecordingStartedAt) / 1000));
    if (openAiChunks.length) {
      openAiBlob = new Blob(openAiChunks, { type: openAiRecorder?.mimeType || 'audio/webm' });
      sendOpenAiRecording(recordedSeconds);
    } else {
      voicePopupNote.textContent = 'No audio captured. Try again.';
    }
    openAiRecorder = null;
  });

  openAiRecorder.start(250);
  openAiRecordingStartedAt = Date.now();
  openAiLastVoiceAt = openAiRecordingStartedAt;
  voicePopupNote.textContent = 'Listening...';

  const renderFrame = () => {
    if (!openAiAnalyser) {
      return;
    }
    openAiAnalyser.getByteTimeDomainData(pcm);
    openAiAnalyser.getByteFrequencyData(freq);
    let sum = 0;
    for (let i = 0; i < pcm.length; i += 1) {
      const centered = (pcm[i] - 128) / 128;
      sum += centered * centered;
    }
    const rms = Math.sqrt(sum / pcm.length);
    if (rms > OPENAI_VOICE_SILENCE_THRESHOLD) {
      openAiLastVoiceAt = Date.now();
    }
    drawVoiceVisuals(pcm, freq, rms);
    openAiVisualRafId = requestAnimationFrame(renderFrame);
  };
  openAiVisualRafId = requestAnimationFrame(renderFrame);

  openAiElapsedTimerId = setInterval(() => {
    const elapsed = Math.floor((Date.now() - openAiRecordingStartedAt) / 1000);
    const silenceSecs = Math.max(0, OPENAI_VOICE_SILENCE_MS - (Date.now() - openAiLastVoiceAt)) / 1000;
    voicePopupNote.textContent = `Listening ${formatVoiceTime(elapsed)} • auto-send in ${silenceSecs.toFixed(1)}s silence`;
    if (elapsed >= OPENAI_VOICE_MAX_SECONDS) {
      voicePopupNote.textContent = 'Reached safety max length, stopping.';
      stopOpenAiRecording();
    }
  }, 1000);

  openAiSilenceTimerId = setInterval(() => {
    if (!openAiAnalyser) {
      return;
    }
    openAiAnalyser.getByteTimeDomainData(pcm);
    let sum = 0;
    for (let i = 0; i < pcm.length; i += 1) {
      const centered = (pcm[i] - 128) / 128;
      sum += centered * centered;
    }
    const rms = Math.sqrt(sum / pcm.length);
    const now = Date.now();
    if (rms > OPENAI_VOICE_SILENCE_THRESHOLD) {
      openAiLastVoiceAt = now;
    }

    const silenceDuration = now - openAiLastVoiceAt;
    const elapsed = now - openAiRecordingStartedAt;
    if (elapsed > 1200 && silenceDuration >= OPENAI_VOICE_SILENCE_MS) {
      voicePopupNote.textContent = 'Silence detected. Sending...';
      stopOpenAiRecording();
    }
  }, 180);
};

const sendOpenAiRecording = async (recordedSeconds = 0) => {
  if (!openAiBlob) {
    return;
  }

  voicePopupNote.textContent = `Transcribing ${recordedSeconds || '?'}s audio...`;

  try {
    const file = new File([openAiBlob], 'voice.webm', { type: openAiBlob.type || 'audio/webm' });
    const formData = new FormData();
    formData.append('audio', file);

    openAiTranscribeAbortController = new AbortController();

    const response = await fetch('/api/transcribe-openai', {
      method: 'POST',
      body: formData,
      signal: openAiTranscribeAbortController.signal,
    });
    openAiTranscribeAbortController = null;

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || 'Transcription failed');
    }

    const text = (payload.text || '').trim();
    if (!text) {
      throw new Error('No transcript returned');
    }

    const base = descriptionInput.value.trim();
    descriptionInput.value = base ? `${base} ${text}` : text;
    setStatus('OpenAI voice transcript added.');
    closeOpenAiVoicePopup();
  } catch (error) {
    if (error.name === 'AbortError') {
      return;
    }
    openAiTranscribeAbortController = null;
    voicePopupNote.textContent = error.message || 'Transcription failed.';
  }
};

openAiVoiceButton.addEventListener('click', () => {
  voicePopup.hidden = false;
  voicePopupNote.textContent = 'Listening...';
  startOpenAiRecording();
});
voiceCancelButton.addEventListener('click', closeOpenAiVoicePopup);

cancelButton.addEventListener('click', () => {
  if (generationAbortController) {
    generationAbortController.abort();
  }
});

fullSizeButton.addEventListener('click', () => openLightbox('generated'));

lightboxCloseButton.addEventListener('click', closeLightbox);
zoomInButton.addEventListener('click', () => zoomLightbox(0.2));
zoomOutButton.addEventListener('click', () => zoomLightbox(-0.2));
zoomResetButton.addEventListener('click', () => {
  lightboxScale = 1;
  lightboxImage.style.transform = 'scale(1)';
  zoomResetButton.textContent = '100%';
});
lightboxPrevButton.addEventListener('click', () => cycleLightbox(-1));
lightboxNextButton.addEventListener('click', () => cycleLightbox(1));
lightboxDownloadButton.addEventListener('click', () => {
  const item = lightboxItems[lightboxIndex];
  if (!item) return;
  const link = document.createElement('a');
  link.href = item.url;
  link.download = item.filename;
  link.click();
});
lightboxCanvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  zoomLightbox(event.deltaY > 0 ? -0.08 : 0.08);
});

lightboxCanvas.addEventListener('pointerdown', (event) => {
  if (event.button !== 0) return;
  lightboxPanState.active = true;
  lightboxPanState.startX = event.clientX;
  lightboxPanState.startY = event.clientY;
  lightboxPanState.scrollLeft = lightboxCanvas.scrollLeft;
  lightboxPanState.scrollTop = lightboxCanvas.scrollTop;
  lightboxCanvas.style.cursor = 'grabbing';
  event.preventDefault();
});

window.addEventListener('pointermove', (event) => {
  if (!lightboxPanState.active) return;
  lightboxCanvas.scrollLeft = lightboxPanState.scrollLeft - (event.clientX - lightboxPanState.startX);
  lightboxCanvas.scrollTop = lightboxPanState.scrollTop - (event.clientY - lightboxPanState.startY);
});

window.addEventListener('pointerup', () => {
  if (lightboxPanState.active) {
    lightboxPanState.active = false;
    lightboxCanvas.style.cursor = 'grab';
  }
});
window.addEventListener('keydown', (event) => {
  if (lightbox.hidden) {
    return;
  }
  if (event.key === 'Escape') {
    closeLightbox();
  } else if (event.key === 'ArrowLeft') {
    cycleLightbox(-1);
  } else if (event.key === 'ArrowRight') {
    cycleLightbox(1);
  }
});

const makeWindowDraggable = () => {
  let active = false;
  let offsetX = 0;
  let offsetY = 0;

  dragHandle.addEventListener('pointerdown', (event) => {
    if (window.innerWidth <= 900) return;
    active = true;
    const rect = windowShell.getBoundingClientRect();
    offsetX = event.clientX - rect.left;
    offsetY = event.clientY - rect.top;
    windowShell.style.position = 'fixed';
    windowShell.style.margin = '0';
    dragHandle.style.cursor = 'grabbing';
  });

  window.addEventListener('pointermove', (event) => {
    if (!active) {
      return;
    }

    const x = Math.max(8, Math.min(window.innerWidth - windowShell.offsetWidth - 8, event.clientX - offsetX));
    const y = Math.max(8, Math.min(window.innerHeight - 40, event.clientY - offsetY));

    windowShell.style.left = `${x}px`;
    windowShell.style.top = `${y}px`;
  });

  window.addEventListener('pointerup', () => {
    active = false;
    dragHandle.style.cursor = 'grab';
  });
};

const loadProviders = async () => {
  const response = await fetch('/api/providers');
  if (!response.ok) {
    throw new Error('Failed to load providers');
  }

  const payload = await response.json();
  providers = payload.providers || {};

  providerSelect.innerHTML = '';
  Object.entries(providers).forEach(([id, provider]) => {
    const option = toOptionMarkup(id);
    option.textContent = provider.hasKey ? provider.label : `${provider.label} (no API key)`;
    providerSelect.appendChild(option);
  });

  updateProviderOptions();
};

const start = async () => {
  try {
    setAiNarrationEnabled(true);
    await loadProviders();
    makeWindowDraggable();
    renderReferenceGallery();
    setStatus('Ready. Upload references or generate directly.');
  } catch (error) {
    setStatus(error.message || 'Could not initialize app');
  }
};

window.addEventListener('beforeunload', clearCachedAiReplyAudio);

start();
