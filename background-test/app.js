const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 25;
const OVERLAP_SECONDS = 1.5;

const $ = (id) => document.getElementById(id);
const fileInput = $('file');
const preview = $('preview');
const filename = $('filename');
const filemeta = $('filemeta');
const startButton = $('start');
const statusBox = $('status');
const progress = $('progress');
const percent = $('percent');
const detail = $('detail');
const srtOutput = $('srt');
const txtOutput = $('txt');
const saveSrt = $('saveSrt');
const saveTxt = $('saveTxt');

let selectedFile = null;
let previewUrl = null;
let audioDuration = 0;
let requestId = 0;
const pending = new Map();

const worker = new Worker('./worker.js', { type: 'module' });

worker.addEventListener('message', (event) => {
  const message = event.data ?? {};

  if (message.type === 'model-progress') {
    setProgress(message.progress, 'Stahuji model', `${Math.round(message.progress)} % • ${String(message.device).toUpperCase()}`);
    return;
  }

  if (!message.id || !pending.has(message.id)) return;
  const waiter = pending.get(message.id);
  pending.delete(message.id);

  if (message.type === 'error') waiter.reject(new Error(message.error || 'Worker selhal.'));
  else waiter.resolve(message.result);
});

worker.addEventListener('error', (event) => {
  for (const waiter of pending.values()) waiter.reject(new Error(event.message || 'Worker selhal.'));
  pending.clear();
});

function workerRequest(type, payload = {}, transfer = []) {
  const id = ++requestId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ type, id, ...payload }, transfer);
  });
}

function prettyTime(seconds) {
  if (!Number.isFinite(seconds)) return '0:00';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`;
}

function prettyBytes(bytes) {
  const units = ['B', 'kB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit++;
  }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function setStatus(text, kind = '') {
  statusBox.textContent = text;
  statusBox.dataset.kind = kind;
}

function setProgress(value, stage, text) {
  const safe = Math.max(0, Math.min(100, Number(value) || 0));
  progress.value = safe;
  percent.textContent = `${Math.round(safe)} %`;
  detail.textContent = `${stage} • ${text}`;
}

function requestedDevice() {
  const value = $('device').value;
  if (value === 'webgpu') {
    if (!navigator.gpu) throw new Error('WebGPU není v tomto prohlížeči dostupné.');
    return 'webgpu';
  }
  return 'wasm';
}

function cleanText(value) {
  return String(value ?? '').replace(/\s+/g, ' ').replace(/\s+([,.;:!?])/g, '$1').trim();
}

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith('audio/') && !/\.(mp3|wav|m4a|ogg|flac)$/i.test(file.name)) {
    setStatus('Soubor nevypadá jako podporovaný zvuk.', 'error');
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  filename.textContent = file.name;
  filemeta.textContent = prettyBytes(file.size);
  startButton.disabled = false;
  setStatus('Soubor je připraven. Spusť test a pak klidně přepni na jinou záložku.', 'ok');
}

fileInput.addEventListener('change', () => setFile(fileInput.files?.[0]));
preview.addEventListener('loadedmetadata', () => {
  audioDuration = preview.duration;
  filemeta.textContent = `${prettyBytes(selectedFile.size)} • ${prettyTime(audioDuration)}`;
});

function resampleLinear(input, inputRate, outputRate) {
  if (inputRate === outputRate) return input;
  const ratio = inputRate / outputRate;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i++) {
    const position = i * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const fraction = position - left;
    output[i] = input[left] * (1 - fraction) + input[right] * fraction;
  }
  return output;
}

async function decodeAudio(file) {
  setProgress(0, 'Připravuji zvuk', prettyBytes(file.size));
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error('Prohlížeč nepodporuje Web Audio API.');

  let context;
  try {
    context = new AudioContextClass({ sampleRate: SAMPLE_RATE });
  } catch (_) {
    context = new AudioContextClass();
  }

  try {
    const buffer = await file.arrayBuffer();
    setProgress(25, 'Připravuji zvuk', 'Dekóduji MP3…');
    const decoded = await context.decodeAudioData(buffer.slice(0));
    const mono = new Float32Array(decoded.length);

    for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
      const data = decoded.getChannelData(channel);
      for (let i = 0; i < data.length; i++) mono[i] += data[i] / decoded.numberOfChannels;
    }

    setProgress(70, 'Připravuji zvuk', 'Převádím na 16 kHz…');
    const output = decoded.sampleRate === SAMPLE_RATE ? mono : resampleLinear(mono, decoded.sampleRate, SAMPLE_RATE);
    audioDuration = output.length / SAMPLE_RATE;
    setProgress(100, 'Zvuk připraven', prettyTime(audioDuration));
    return output;
  } finally {
    await context.close();
  }
}

function removeRepeatedBoundary(previousText, currentText) {
  const previous = cleanText(previousText).split(' ').filter(Boolean);
  const current = cleanText(currentText).split(' ').filter(Boolean);
  const maximum = Math.min(10, previous.length, current.length);
  for (let count = maximum; count >= 2; count--) {
    if (previous.slice(-count).join(' ').toLowerCase() === current.slice(0, count).join(' ').toLowerCase()) {
      return current.slice(count).join(' ');
    }
  }
  return current.join(' ');
}

function normalizeResult(result, chunkStart, chunkEnd, acceptFrom) {
  const segments = [];
  const chunks = result?.chunks ?? [];

  for (const chunk of chunks) {
    const text = cleanText(chunk.text);
    if (!text) continue;
    const timestamp = Array.isArray(chunk.timestamp) ? chunk.timestamp : [0, null];
    let start = chunkStart + (Number.isFinite(timestamp[0]) ? timestamp[0] : 0);
    let end = Number.isFinite(timestamp[1]) ? chunkStart + timestamp[1] : Math.min(chunkEnd, start + Math.max(0.8, text.length / 13));
    if (end <= acceptFrom) continue;
    start = Math.max(start, acceptFrom);
    end = Math.max(start + 0.1, Math.min(end, chunkEnd));
    segments.push({ text, start, end });
  }

  if (!segments.length && cleanText(result?.text)) {
    segments.push({ text: cleanText(result.text), start: acceptFrom, end: chunkEnd });
  }
  return segments;
}

function mergeSegments(target, additions) {
  for (const item of additions) {
    const next = { ...item };
    const previous = target[target.length - 1];
    if (previous) next.text = removeRepeatedBoundary(previous.text, next.text);
    next.text = cleanText(next.text);
    if (next.text) target.push(next);
  }
}

function segmentsToWords(segments) {
  const words = [];
  for (const segment of segments) {
    const parts = cleanText(segment.text).split(/\s+/).filter(Boolean);
    if (!parts.length) continue;
    const duration = Math.max(0.4, segment.end - segment.start);
    const each = duration / parts.length;
    parts.forEach((text, index) => {
      const start = segment.start + each * index;
      const end = index === parts.length - 1 ? segment.end : start + each;
      words.push({ text, start, end });
    });
  }
  return words;
}

function wrapText(text, charsPerLine, lines) {
  if (lines <= 1 || text.length <= charsPerLine) return text;
  const words = text.split(' ');
  const result = [];
  let current = '';
  const target = Math.ceil(text.length / lines);
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (result.length < lines - 1 && current && candidate.length > target) {
      result.push(current);
      current = word;
    } else current = candidate;
  }
  if (current) result.push(current);
  return result.join('\n');
}

function buildCaptions(segments) {
  const maxDuration = Number($('maxdur').value);
  const charsPerLine = Number($('chars').value);
  const lines = Number($('lines').value);
  const maxChars = charsPerLine * lines;
  const words = segmentsToWords(segments);
  const captions = [];
  let current = null;

  const flush = () => {
    if (!current) return;
    captions.push({ ...current, text: wrapText(current.text, charsPerLine, lines) });
    current = null;
  };

  for (const word of words) {
    if (!current) {
      current = { ...word };
      continue;
    }
    const candidate = `${current.text} ${word.text}`;
    if (candidate.length > maxChars || word.end - current.start > maxDuration) flush();
    if (!current) current = { ...word };
    else {
      current.text = candidate;
      current.end = word.end;
    }
    if (/[.!?…][”"')\]]?$/.test(current.text) && current.end - current.start >= 1.1) flush();
  }
  flush();
  return captions;
}

function formatSrtTime(seconds) {
  const totalMs = Math.max(0, Math.round(seconds * 1000));
  const ms = totalMs % 1000;
  const totalSeconds = Math.floor(totalMs / 1000);
  const s = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const m = totalMinutes % 60;
  const h = Math.floor(totalMinutes / 60);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
}

function captionsToSrt(captions) {
  return captions.map((caption, index) => `${index + 1}\n${formatSrtTime(caption.start)} --> ${formatSrtTime(caption.end)}\n${caption.text}`).join('\n\n');
}

function renderOutputs(segments, chunkIndex, totalChunks) {
  const text = cleanText(segments.map((segment) => segment.text).join(' '));
  const captions = buildCaptions(segments);
  txtOutput.value = text;
  srtOutput.value = captionsToSrt(captions);
  setStatus(`Průběžně hotovo ${chunkIndex}/${totalChunks} částí. Worker stále běží i při neaktivní záložce.`, 'warn');
  return captions.length;
}

async function runTranscription(audio, modelId, language, device) {
  await workerRequest('load', { modelId, device });

  const chunkSamples = Math.round(CHUNK_SECONDS * SAMPLE_RATE);
  const overlapSamples = Math.round(OVERLAP_SECONDS * SAMPLE_RATE);
  const stepSamples = chunkSamples - overlapSamples;
  const totalChunks = Math.max(1, Math.ceil(Math.max(1, audio.length - overlapSamples) / stepSamples));
  const segments = [];

  for (let index = 0; index < totalChunks; index++) {
    const startSample = index * stepSamples;
    if (startSample >= audio.length) break;
    const endSample = Math.min(audio.length, startSample + chunkSamples);
    const chunkStart = startSample / SAMPLE_RATE;
    const chunkEnd = endSample / SAMPLE_RATE;
    const chunk = audio.slice(startSample, endSample);

    setProgress((chunkStart / audioDuration) * 100, 'Přepisuji ve workeru', `${prettyTime(chunkStart)} / ${prettyTime(audioDuration)} • ${index + 1}/${totalChunks}`);

    const result = await workerRequest('transcribe', {
      modelId,
      language,
      device,
      audio: chunk,
    }, [chunk.buffer]);

    const acceptFrom = index === 0 ? chunkStart : chunkStart + OVERLAP_SECONDS;
    mergeSegments(segments, normalizeResult(result, chunkStart, chunkEnd, acceptFrom));
    renderOutputs(segments, index + 1, totalChunks);
    setProgress((chunkEnd / audioDuration) * 100, 'Přepisuji ve workeru', `${prettyTime(chunkEnd)} / ${prettyTime(audioDuration)} • ${index + 1}/${totalChunks}`);
  }

  return segments;
}

function setBusy(busy) {
  startButton.disabled = busy || !selectedFile;
  fileInput.disabled = busy;
  for (const id of ['model', 'language', 'device', 'maxdur', 'chars', 'lines']) $(id).disabled = busy;
}

startButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  setBusy(true);
  srtOutput.value = '';
  txtOutput.value = '';
  saveSrt.disabled = true;
  saveTxt.disabled = true;

  try {
    const modelId = $('model').value;
    const language = $('language').value;
    const device = requestedDevice();
    const audio = await decodeAudio(selectedFile);
    setStatus('Zvuk je připraven. Model a přepis teď běží ve Web Workeru.', 'warn');
    const segments = await runTranscription(audio, modelId, language, device);
    const captions = buildCaptions(segments);
    if (!segments.length || !captions.length) throw new Error('Model nevrátil použitelný přepis.');
    renderOutputs(segments, 1, 1);
    setProgress(100, 'Hotovo', `${captions.length} titulků • ${device.toUpperCase()}`);
    setStatus(`Hotovo. Vytvořeno ${captions.length} titulků.`, 'ok');
    saveSrt.disabled = false;
    saveTxt.disabled = false;
  } catch (error) {
    console.error(error);
    setStatus(`Přepis se nepodařil: ${error instanceof Error ? error.message : String(error)}`, 'error');
  } finally {
    setBusy(false);
  }
});

function downloadText(text, extension, mime) {
  const base = (selectedFile?.name ?? 'prepis').replace(/\.[^.]+$/, '').replace(/[\\/:*?"<>|]+/g, '_');
  const blob = new Blob(['\ufeff', text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${base}.${extension}`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

saveSrt.addEventListener('click', () => downloadText(srtOutput.value, 'srt', 'application/x-subrip'));
saveTxt.addEventListener('click', () => downloadText(txtOutput.value, 'txt', 'text/plain'));

window.addEventListener('beforeunload', () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  worker.terminate();
});
