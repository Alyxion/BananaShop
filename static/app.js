const form = document.getElementById('generatorForm');
const controlsPanel = document.querySelector('.controls');
const providerGroup = document.getElementById('providerGroup');
const qualityPills = document.getElementById('qualityPills');
const ratioPills = document.getElementById('ratioPills');
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
const clearDescriptionButton = document.getElementById('clearDescriptionButton');
const cameraInput = document.getElementById('cameraInput');
const recentGallery = document.getElementById('recentGallery');
const recentButton = document.getElementById('recentButton');
const recentPopup = document.getElementById('recentPopup');
const recentCloseButton = document.getElementById('recentCloseButton');
const voicePopup = document.getElementById('voicePopup');
const voiceCancelButton = document.getElementById('voiceCancelButton');
const voiceSendButton = document.getElementById('voiceSendButton');
const voicePopupNote = document.getElementById('voicePopupNote');
const voiceWaveCanvas = document.getElementById('voiceWaveCanvas');
const voiceBandsCanvas = document.getElementById('voiceBandsCanvas');
const voiceLoudnessCanvas = document.getElementById('voiceLoudnessCanvas');

const formatGroup = document.getElementById('formatGroup');
const formatPills = document.getElementById('formatPills');
const downloadFormatPopup = document.getElementById('downloadFormatPopup');
const downloadFormatCloseButton = document.getElementById('downloadFormatCloseButton');
const downloadSvgButton = document.getElementById('downloadSvgButton');
const downloadPngButton = document.getElementById('downloadPngButton');

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
const mobileTabs = document.getElementById('mobileTabs');
const layoutGrid = document.querySelector('.layout-grid');

const GENERATION_TIMEOUT_MS = 120000;

const isMobileLayout = () => window.innerWidth <= 900;

const switchMobileTab = (tab) => {
  layoutGrid.dataset.activeTab = tab;
  mobileTabs.querySelectorAll('.mobile-tab').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.tab === tab);
  });
  if (tab === 'result') {
    const badge = mobileTabs.querySelector('.tab-badge');
    if (badge) badge.remove();
  }
};

const notifyResultTab = () => {
  if (!isMobileLayout() || layoutGrid.dataset.activeTab === 'result') return;
  const resultBtn = mobileTabs.querySelector('[data-tab="result"]');
  if (resultBtn && !resultBtn.querySelector('.tab-badge')) {
    const dot = document.createElement('span');
    dot.className = 'tab-badge';
    resultBtn.appendChild(dot);
  }
};

mobileTabs.addEventListener('click', (e) => {
  const tab = e.target.closest('.mobile-tab');
  if (!tab) return;
  switchMobileTab(tab.dataset.tab);
});

let providers = {};

const fallbackFilenameFromDescription = (description, ext = 'png') => {
  const fallback = description
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .trim()
    .replace(/[-\s]+/g, '-')
    .slice(0, 80);
  return `${fallback || 'generated-image'}.${ext}`;
};
let isGenerating = false;
let generatedImageDataUrl = '';
let generatedFilename = 'generated-image.png';
let generatedFormat = 'Photo';
let generatedSvgRaw = '';
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

let recentDB = null;
const RECENT_DB_NAME = 'BananaStoreDB';
const RECENT_STORE = 'recentImages';
const RECENT_LIMIT = 9;

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
  const icon = aiNarrationToggle.querySelector('i');
  if (icon) icon.className = aiNarrationEnabled ? 'ph ph-speaker-high' : 'ph ph-speaker-slash';
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

const setAiReplyPlaying = (playing) => {
  const icon = aiReplySpeakButton.querySelector('i');
  const bars = aiReplySpeakButton.querySelector('.audio-bars');
  if (playing) {
    aiReplySpeakButton.classList.add('is-playing');
    if (icon) icon.className = 'ph ph-stop';
    if (!bars) {
      const barsEl = document.createElement('span');
      barsEl.className = 'audio-bars';
      barsEl.innerHTML = '<span></span><span></span><span></span><span></span>';
      aiReplySpeakButton.appendChild(barsEl);
    }
  } else {
    aiReplySpeakButton.classList.remove('is-playing');
    if (icon) icon.className = 'ph ph-speaker-high';
    if (bars) bars.remove();
  }
};

const stopAiReplyPlayback = () => {
  if (aiReplyAudio) {
    aiReplyAudio.pause();
    aiReplyAudio.currentTime = 0;
  }
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
  setAiReplyPlaying(false);
};

const playAndCacheAiReplyAudio = async (audioBlob) => {
  if (!audioBlob || !audioBlob.size) {
    throw new Error('No narration audio returned');
  }
  clearCachedAiReplyAudio();
  aiReplyAudioUrl = URL.createObjectURL(audioBlob);
  aiReplyAudio = new Audio(aiReplyAudioUrl);
  aiReplyAudio.preload = 'auto';
  aiReplyAudio.addEventListener('ended', () => setAiReplyPlaying(false));
  aiReplyAudio.addEventListener('pause', () => {
    if (aiReplyAudio?.ended || aiReplyAudio?.currentTime === 0) setAiReplyPlaying(false);
  });
  setAiReplyPlaying(true);
  await aiReplyAudio.play();
};

const replayCachedAiReplyAudio = async () => {
  if (!aiReplyAudio) {
    return false;
  }
  aiReplyAudio.currentTime = 0;
  setAiReplyPlaying(true);
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

  utterance.addEventListener('end', () => setAiReplyPlaying(false));
  utterance.addEventListener('error', () => setAiReplyPlaying(false));
  setAiReplyPlaying(true);
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

const PROVIDER_LOGOS = {
  openai: '<svg class="provider-logo" viewBox="0 0 24 24" fill="currentColor"><path d="M22.28 9.37a6.3 6.3 0 0 0-.54-5.2 6.37 6.37 0 0 0-6.86-3.1A6.3 6.3 0 0 0 10.13 0a6.37 6.37 0 0 0-6.07 4.42 6.3 6.3 0 0 0-4.23 3.07 6.37 6.37 0 0 0 .79 7.47 6.3 6.3 0 0 0 .54 5.2 6.37 6.37 0 0 0 6.86 3.1A6.3 6.3 0 0 0 12.77 24a6.37 6.37 0 0 0 6.07-4.42 6.3 6.3 0 0 0 4.23-3.07 6.37 6.37 0 0 0-.79-7.14zM12.77 22.66a4.75 4.75 0 0 1-3.05-1.1l.15-.09 5.07-2.93a.82.82 0 0 0 .42-.72v-7.15l2.14 1.24a.08.08 0 0 1 .04.06v5.92a4.77 4.77 0 0 1-4.77 4.77zM3.67 18.5a4.74 4.74 0 0 1-.57-3.2l.15.09 5.07 2.93a.83.83 0 0 0 .83 0l6.19-3.57v2.47a.07.07 0 0 1-.03.06l-5.12 2.96a4.77 4.77 0 0 1-6.52-1.74zM2.34 7.9A4.74 4.74 0 0 1 4.82 5.8v6.03a.82.82 0 0 0 .41.71l6.19 3.57-2.14 1.24a.08.08 0 0 1-.07 0L4.09 14.4A4.77 4.77 0 0 1 2.34 7.9zm17.13 3.98-6.19-3.57 2.14-1.24a.08.08 0 0 1 .07 0l5.12 2.96a4.77 4.77 0 0 1-.74 8.6v-6.03a.83.83 0 0 0-.4-.72zm2.13-3.2-.15-.1-5.07-2.93a.83.83 0 0 0-.83 0l-6.19 3.58V6.75a.07.07 0 0 1 .03-.06l5.12-2.96a4.77 4.77 0 0 1 7.09 4.95zM9.47 13.37l-2.14-1.24a.08.08 0 0 1-.04-.06V6.15a4.77 4.77 0 0 1 7.82-3.67l-.15.09-5.07 2.93a.82.82 0 0 0-.42.72zm1.16-2.5L12.77 9.8l2.13 1.24v2.47l-2.13 1.24-2.14-1.24z"/></svg>',
  google: '<svg class="provider-logo" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>',
  anthropic: '<svg class="provider-logo" viewBox="0 0 24 24" fill="currentColor"><path d="M17.304 3.541h-3.672l6.696 16.918H24Zm-10.608 0L0 20.459h3.744l1.37-3.553h7.005l1.369 3.553h3.744L10.536 3.541Zm-.371 10.223L8.616 7.82l2.291 5.945Z"/></svg>',
};

const QUALITY_LABELS = {
  auto: 'Auto',
  low: 'Lo',
  medium: 'Med',
  high: 'Hi',
  standard: 'Std',
  hd: 'HD',
};

const renderPills = (container, values, selectedIndex = 0, labels = null) => {
  container.innerHTML = '';
  values.forEach((value, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `option-pill${i === selectedIndex ? ' active' : ''}`;
    btn.dataset.value = value;
    const label = labels?.[value];
    if (label) {
      btn.innerHTML = label;
    } else {
      btn.textContent = value;
    }
    btn.addEventListener('click', () => {
      container.querySelectorAll('.option-pill').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
    });
    container.appendChild(btn);
  });
};

const setGenerating = (value) => {
  isGenerating = value;
  controlsPanel.classList.toggle('locked', value);
  generateButton.disabled = value;
  progressOverlay.hidden = !value;
  if (value) {
    emptyPreview.hidden = true;
  }
  if (!value) {
    generationAbortController = null;
    if (!generatedImageDataUrl) {
      emptyPreview.hidden = false;
    }
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

const getActiveProvider = () => providerGroup.querySelector('.provider-btn.active')?.dataset.provider || '';
const getActivePill = (container) => container.querySelector('.option-pill.active')?.dataset.value || '';

const updateQualityForFormat = () => {
  const provider = providers[getActiveProvider()];
  if (!provider) return;
  const activeFormat = getActivePill(formatPills);
  const fq = provider.formatQualities;
  const qualities = (fq && activeFormat && fq[activeFormat]) || provider.qualities;
  renderPills(qualityPills, qualities, 0, QUALITY_LABELS);
};

const updateProviderOptions = () => {
  const provider = providers[getActiveProvider()];
  if (!provider) return;
  renderPills(ratioPills, provider.ratios);
  const formats = provider.formats || ['Photo'];
  renderPills(formatPills, formats);
  formatGroup.hidden = formats.length <= 1;
  updateQualityForFormat();
};

formatPills.addEventListener('click', (e) => {
  if (e.target.closest('.option-pill')) updateQualityForFormat();
});

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

const makeUploadTile = (iconClass, handler) => {
  const tile = document.createElement('div');
  tile.className = 'upload-tile';
  tile.innerHTML = `<i class="ph ${iconClass}"></i>`;
  tile.addEventListener('click', handler);
  ['dragenter', 'dragover'].forEach((evt) => {
    tile.addEventListener(evt, (e) => { e.preventDefault(); tile.classList.add('drag-over'); });
  });
  ['dragleave', 'drop'].forEach((evt) => {
    tile.addEventListener(evt, (e) => { e.preventDefault(); tile.classList.remove('drag-over'); });
  });
  tile.addEventListener('drop', (e) => {
    const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith('image/'));
    if (files.length) addReferenceFiles(files, true);
  });
  return tile;
};

const renderReferenceGallery = () => {
  referenceGallery.innerHTML = '';

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

  referenceGallery.appendChild(makeUploadTile('ph-plus', () => {
    referenceInput.click();
  }));
  referenceGallery.appendChild(makeUploadTile('ph-camera', () => {
    if (isTouchDevice()) cameraInput.click(); else captureFromWebcam();
  }));

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

const fileToDataUrl = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(reader.result);
  reader.onerror = () => reject(reader.error);
  reader.readAsDataURL(file);
});

const svgDataUrlToPngDataUrl = (svgDataUrl, targetWidth = 2000) => new Promise((resolve, reject) => {
  let aspectRatio = 1;
  try {
    const svgText = atob(svgDataUrl.split(',')[1]);
    const vbMatch = svgText.match(/viewBox=["']\s*[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)/);
    if (vbMatch) {
      const vbW = parseFloat(vbMatch[1]);
      const vbH = parseFloat(vbMatch[2]);
      if (vbW > 0 && vbH > 0) aspectRatio = vbH / vbW;
    }
  } catch { /* ignore */ }
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = Math.round(targetWidth * aspectRatio);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    resolve(canvas.toDataURL('image/png'));
  };
  img.onerror = () => reject(new Error('Failed to rasterize SVG'));
  img.src = svgDataUrl;
});

const hashImageData = async (dataUrl) => {
  const base64 = dataUrl.split(',')[1] || '';
  if (crypto.subtle) {
    const data = new TextEncoder().encode(base64);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    return Array.from(new Uint8Array(hashBuffer)).map((b) => b.toString(16).padStart(2, '0')).join('');
  }
  let h = 0;
  for (let i = 0; i < base64.length; i += 1) {
    h = ((h << 5) - h + base64.charCodeAt(i)) | 0;
  }
  return h.toString(16);
};

const createThumbnail = (dataUrl) => new Promise((resolve) => {
  const img = new Image();
  img.onload = () => {
    const size = 82;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    const min = Math.min(img.width, img.height);
    const sx = (img.width - min) / 2;
    const sy = (img.height - min) / 2;
    ctx.drawImage(img, sx, sy, min, min, 0, 0, size, size);
    resolve(canvas.toDataURL('image/jpeg', 0.75));
  };
  img.src = dataUrl;
});

const initRecentDB = () => new Promise((resolve, reject) => {
  const request = indexedDB.open(RECENT_DB_NAME, 1);
  request.onupgradeneeded = (e) => {
    const db = e.target.result;
    if (!db.objectStoreNames.contains(RECENT_STORE)) {
      const store = db.createObjectStore(RECENT_STORE, { keyPath: 'id', autoIncrement: true });
      store.createIndex('hash', 'hash', { unique: false });
      store.createIndex('timestamp', 'timestamp', { unique: false });
    }
  };
  request.onsuccess = (e) => {
    recentDB = e.target.result;
    resolve(recentDB);
  };
  request.onerror = (e) => reject(e.target.error);
});

const getAllRecent = () => new Promise((resolve) => {
  if (!recentDB) { resolve([]); return; }
  const tx = recentDB.transaction(RECENT_STORE, 'readonly');
  const store = tx.objectStore(RECENT_STORE);
  const req = store.index('timestamp').getAll();
  req.onsuccess = (e) => {
    const items = e.target.result || [];
    items.sort((a, b) => b.timestamp - a.timestamp);
    resolve(items);
  };
  req.onerror = () => resolve([]);
});

const enforceRecentLimit = () => new Promise((resolve) => {
  if (!recentDB) { resolve(); return; }
  const tx = recentDB.transaction(RECENT_STORE, 'readwrite');
  const store = tx.objectStore(RECENT_STORE);
  store.index('timestamp').getAll().onsuccess = (e) => {
    const all = e.target.result || [];
    if (all.length > RECENT_LIMIT) {
      all.sort((a, b) => a.timestamp - b.timestamp);
      for (const item of all.slice(0, all.length - RECENT_LIMIT)) {
        store.delete(item.id);
      }
    }
  };
  tx.oncomplete = () => resolve();
  tx.onerror = () => resolve();
});

const addRecentImage = async (dataUrl, filename) => {
  if (!recentDB) return;
  try {
    const hash = await hashImageData(dataUrl);
    const thumbnail = await createThumbnail(dataUrl);
    await new Promise((resolve, reject) => {
      const tx = recentDB.transaction(RECENT_STORE, 'readwrite');
      const store = tx.objectStore(RECENT_STORE);
      store.index('hash').openCursor(IDBKeyRange.only(hash)).onsuccess = (e) => {
        const cursor = e.target.result;
        if (cursor) {
          const record = cursor.value;
          record.timestamp = Date.now();
          store.put(record);
        } else {
          store.add({ hash, thumbnail, dataUrl, filename: filename || 'image.png', timestamp: Date.now() });
        }
      };
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    await enforceRecentLimit();
    renderRecentGallery();
  } catch (err) {
    console.warn('Failed to add recent image:', err);
  }
};

const renderRecentGallery = async () => {
  const items = await getAllRecent();
  recentGallery.innerHTML = '';
  recentButton.hidden = !items.length;
  if (!items.length) {
    recentPopup.hidden = true;
    return;
  }
  items.forEach((item) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'recent-thumb';
    button.title = item.filename;
    const img = document.createElement('img');
    img.src = item.thumbnail;
    img.alt = item.filename;
    button.appendChild(img);
    button.addEventListener('click', async () => {
      const file = await dataUrlToFile(item.dataUrl, item.filename);
      addReferenceFiles([file], true);
      recentPopup.hidden = true;
    });
    recentGallery.appendChild(button);
  });
};

const suggestFilename = async (description, ext = 'png') => {
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
    return `${stem}.${ext}`;
  } catch {
    const fallback = description
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .trim()
      .replace(/[-\s]+/g, '-')
      .slice(0, 80);
    return `${fallback || 'generated-image'}.${ext}`;
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
  const activeFormat = getActivePill(formatPills) || 'Photo';
  setGenerating(true);
  const loaderText = progressOverlay.querySelector('.loader-text');
  if (loaderText) loaderText.textContent = activeFormat === 'Vector' ? 'Generating vector...' : 'Generating image...';
  setStatus('Generating...');
  const fileExt = activeFormat === 'Vector' ? 'svg' : 'png';
  const filenamePromise = suggestFilename(description, fileExt);

  try {
    const fd = new FormData();
    fd.append('provider', getActiveProvider());
    fd.append('description', description);
    fd.append('quality', getActivePill(qualityPills));
    fd.append('ratio', getActivePill(ratioPills));
    fd.append('format', activeFormat);
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

    generatedFormat = data.format || 'Photo';
    generatedSvgRaw = '';
    if (generatedFormat === 'Vector' && data.image_data_url.startsWith('data:image/svg+xml;base64,')) {
      try {
        generatedSvgRaw = atob(data.image_data_url.split(',')[1]);
      } catch { /* ignore decode errors */ }
    }

    generatedFilename = fallbackFilenameFromDescription(description, fileExt);
    const suggestedFilename = await Promise.race([
      filenamePromise,
      new Promise((resolve) => window.setTimeout(() => resolve(generatedFilename), 2500)),
    ]);
    generatedFilename = suggestedFilename || generatedFilename;
    setGeneratedPreview(data.image_data_url);
    setImageHint(`${description.slice(0, 120)}${description.length > 120 ? '...' : ''} — ${data.provider}, ${data.size}`);

    if (isMobileLayout()) {
      switchMobileTab('result');
    }

    const generatedFile = await dataUrlToFile(data.image_data_url, generatedFilename);
    addReferenceFiles([generatedFile], true);
    addRecentImage(data.image_data_url, generatedFilename);

    let narrationErrorMessage = '';
    const language = inferSpeechLanguageFromText(description);
    try {
      let narrationImageUrl = data.image_data_url;
      if (generatedFormat === 'Vector') {
        try {
          narrationImageUrl = await svgDataUrlToPngDataUrl(data.image_data_url);
        } catch { /* fall back to original data URL */ }
      }
      const reply = await describeGeneratedImage(narrationImageUrl, description);
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
    files.forEach((file) => {
      fileToDataUrl(file).then((dataUrl) => addRecentImage(dataUrl, file.name));
    });
  }
  referenceInput.value = '';
});

form.addEventListener('submit', (event) => {
  event.preventDefault();
  generateImage();
});

aiNarrationToggle.addEventListener('click', () => {
  setAiNarrationEnabled(!aiNarrationEnabled);
});

aiReplySpeakButton.addEventListener('click', () => {
  if (aiReplySpeakButton.classList.contains('is-playing')) {
    stopAiReplyPlayback();
    return;
  }
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
      setAiReplyPlaying(true);
      return undefined;
    })
    .catch((error) => {
      setAiReplyPlaying(false);
      setStatus(error.message || 'Replay failed.');
    });
});

previewImage.addEventListener('click', () => openLightbox('generated'));

const showDownloadFormatPopup = () => {
  downloadFormatPopup.hidden = false;
};

const hideDownloadFormatPopup = () => {
  downloadFormatPopup.hidden = true;
};

const downloadAsSvg = (svgRaw, filename) => {
  const blob = new Blob([svgRaw], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename.endsWith('.svg') ? filename : filename.replace(/\.\w+$/, '.svg');
  link.click();
  URL.revokeObjectURL(url);
};

const downloadAsPng = async (svgDataUrl, filename) => {
  try {
    const pngDataUrl = await svgDataUrlToPngDataUrl(svgDataUrl);
    const link = document.createElement('a');
    link.href = pngDataUrl;
    link.download = filename.replace(/\.\w+$/, '.png');
    link.click();
  } catch (err) {
    setStatus(`PNG conversion failed: ${err.message}`);
  }
};

downloadButton.addEventListener('click', () => {
  if (!generatedImageDataUrl) {
    return;
  }
  if (generatedFormat === 'Vector') {
    showDownloadFormatPopup();
    return;
  }
  const link = document.createElement('a');
  link.href = generatedImageDataUrl;
  link.download = generatedFilename;
  link.click();
});

downloadFormatCloseButton.addEventListener('click', hideDownloadFormatPopup);

downloadSvgButton.addEventListener('click', () => {
  hideDownloadFormatPopup();
  if (generatedSvgRaw) {
    downloadAsSvg(generatedSvgRaw, generatedFilename);
  }
});

downloadPngButton.addEventListener('click', () => {
  hideDownloadFormatPopup();
  if (generatedImageDataUrl) {
    downloadAsPng(generatedImageDataUrl, generatedFilename);
  }
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
    const captureDataUrl = canvas.toDataURL('image/png');
    canvas.toBlob((blob) => {
      if (blob) {
        const fileName = `camera-${Date.now()}.png`;
        const file = new File([blob], fileName, { type: 'image/png' });
        addReferenceFiles([file], true);
        addRecentImage(captureDataUrl, fileName);
        setStatus('Photo captured and added to references.');
      }
    }, 'image/png');
  });
};

cameraInput.addEventListener('change', () => {
  const files = Array.from(cameraInput.files || []);
  if (files.length) {
    addReferenceFiles(files, true);
    files.forEach((file) => {
      fileToDataUrl(file).then((dataUrl) => addRecentImage(dataUrl, file.name));
    });
  }
  cameraInput.value = '';
});

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
  voiceButton.classList.remove('recording');
  voiceButton.title = 'Talk me through your image idea';
  voicePopup.hidden = true;
};

const stopOpenAiRecording = () => {
  if (openAiRecorder && openAiRecorder.state !== 'inactive') {
    openAiRecorder.stop();
  }
  clearOpenAiTimers();
};

const sendNowOpenAiRecording = () => {
  if (!openAiRecorder || openAiRecorder.state === 'inactive') {
    return;
  }
  voicePopupNote.textContent = 'Sending...';
  openAiRecorder.requestData();
  stopOpenAiRecording();
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
  voiceButton.classList.add('recording');
  voiceButton.title = 'Listening...';
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
    updateClearButton();
    setStatus('Voice transcript added.');
    closeOpenAiVoicePopup();
  } catch (error) {
    if (error.name === 'AbortError') {
      return;
    }
    openAiTranscribeAbortController = null;
    voicePopupNote.textContent = error.message || 'Transcription failed.';
  }
};

const updateClearButton = () => {
  clearDescriptionButton.hidden = !descriptionInput.value.trim();
};

descriptionInput.addEventListener('input', updateClearButton);

clearDescriptionButton.addEventListener('click', () => {
  descriptionInput.value = '';
  descriptionInput.focus();
  updateClearButton();
});

voiceButton.addEventListener('click', () => {
  voiceButton.classList.remove('recording');
  voicePopup.hidden = false;
  voicePopupNote.textContent = 'Listening...';
  startOpenAiRecording();
});
voiceCancelButton.addEventListener('click', closeOpenAiVoicePopup);
voiceSendButton.addEventListener('click', sendNowOpenAiRecording);

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
  if (lightboxIndex === 0 && generatedFormat === 'Vector' && generatedSvgRaw) {
    showDownloadFormatPopup();
    return;
  }
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
  if (event.key === 'Escape' && !downloadFormatPopup.hidden) {
    hideDownloadFormatPopup();
    return;
  }
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

  providerGroup.innerHTML = '';
  let first = true;
  Object.entries(providers).forEach(([id, provider]) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `provider-btn${first ? ' active' : ''}`;
    btn.dataset.provider = id;
    const logo = PROVIDER_LOGOS[id] || '<i class="ph ph-robot"></i>';
    const keyHint = provider.hasKey ? '' : ' <span class="no-key">(no key)</span>';
    btn.innerHTML = `${logo}<span>${provider.label}${keyHint}</span>`;
    btn.addEventListener('click', () => {
      providerGroup.querySelectorAll('.provider-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      updateProviderOptions();
    });
    providerGroup.appendChild(btn);
    first = false;
  });

  updateProviderOptions();
};

recentButton.addEventListener('click', () => {
  recentPopup.hidden = !recentPopup.hidden;
});

recentCloseButton.addEventListener('click', () => {
  recentPopup.hidden = true;
});

document.addEventListener('pointerdown', (e) => {
  if (!recentPopup.hidden && !recentPopup.contains(e.target) && e.target !== recentButton && !recentButton.contains(e.target)) {
    recentPopup.hidden = true;
  }
  if (!downloadFormatPopup.hidden && e.target === downloadFormatPopup) {
    hideDownloadFormatPopup();
  }
});

const start = async () => {
  try {
    setAiNarrationEnabled(true);
    await loadProviders();
    makeWindowDraggable();
    renderReferenceGallery();
    try { await initRecentDB(); renderRecentGallery(); } catch (e) { console.warn('Recent images DB unavailable:', e); }
    setStatus('Ready. Upload references or generate directly.');
  } catch (error) {
    setStatus(error.message || 'Could not initialize app');
  }
};

window.addEventListener('beforeunload', clearCachedAiReplyAudio);

start();
