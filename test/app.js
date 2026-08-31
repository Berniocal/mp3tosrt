import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

env.allowLocalModels = false;
env.useBrowserCache = true;
if ('useWasmCache' in env) env.useWasmCache = true;

const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 25;
const OVERLAP_SECONDS = 1.5;
const MIN_SPLIT_SECONDS = 5.5;
const MAX_SPLIT_DEPTH = 2;

const $ = (id) => document.getElementById(id);
const fileInput = $('file');
const drop = $('drop');
const fileBox = $('filebox');
const fileName = $('filename');
const fileMeta = $('filemeta');
const preview = $('preview');
const startButton = $('start');
const statusBox = $('status');
const progressBox = $('progressbox');
const progress = $('progress');
const stage = $('stage');
const percent = $('percent');
const detail = $('detail');
const srtOutput = $('srt');
const txtOutput = $('txt');
const count = $('count');
const saveSrt = $('saveSrt');
const saveTxt = $('saveTxt');
const copyButton = $('copy');

let selectedFile = null;
let previewUrl = null;
let audioDuration = 0;
let currentPipeline = null;
let currentPipelineKey = '';
let activeTab = 'srt';

class DegenerateOutputError extends Error {
  constructor(message) {
    super(message);
    this.name = 'DegenerateOutputError';
  }
}

class BackendRetryError extends Error {
  constructor(message, cause) {
    super(message);
    this.name = 'BackendRetryError';
    this.cause = cause;
  }
}

function cleanText(value) {
  return String(value ?? '')
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.;:!?])/g, '$1')
    .trim();
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function setStatus(message, kind = '') {
  statusBox.textContent = message;
  statusBox.dataset.kind = kind;
}

function setProgress(value, stageText = 'Pracuji', detailText = '') {
  const safeValue = Math.max(0, Math.min(100, Number(value) || 0));
  progressBox.classList.remove('hidden');
  progress.value = safeValue;
  stage.textContent = stageText;
  percent.textContent = `${Math.round(safeValue)} %`;
  detail.textContent = detailText || 'Probíhá zpracování…';
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

function prettyTime(seconds) {
  if (!Number.isFinite(seconds)) return 'neznámá délka';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return hours
    ? `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
    : `${minutes}:${String(secs).padStart(2, '0')}`;
}

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith('audio/') && !/\.(mp3|wav|m4a|ogg|flac)$/i.test(file.name)) {
    setStatus('Soubor nevypadá jako zvuk.', 'error');
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  fileName.textContent = file.name;
  fileMeta.textContent = prettyBytes(file.size);
  fileBox.style.display = 'block';
  startButton.disabled = false;
  setStatus('Soubor je připraven. Spusť přepis.');
}

fileInput.addEventListener('change', () => setFile(fileInput.files?.[0]));

for (const eventName of ['dragenter', 'dragover']) {
  drop.addEventListener(eventName, (event) => {
    event.preventDefault();
    drop.classList.add('drag');
  });
}

for (const eventName of ['dragleave', 'drop']) {
  drop.addEventListener(eventName, (event) => {
    event.preventDefault();
    drop.classList.remove('drag');
  });
}

drop.addEventListener('drop', (event) => setFile(event.dataTransfer.files?.[0]));

preview.addEventListener('loadedmetadata', () => {
  audioDuration = preview.duration;
  fileMeta.textContent = `${prettyBytes(selectedFile.size)} • ${prettyTime(audioDuration)}`;
});

function requestedDevice() {
  const selected = $('device').value;
  if (selected === 'webgpu') {
    if (!navigator.gpu) {
      throw new Error('WebGPU není v tomto prohlížeči dostupné. Zvol Automaticky nebo WASM.');
    }
    return 'webgpu';
  }
  return 'wasm';
}

async function disposePipeline() {
  if (currentPipeline?.dispose) {
    try { await currentPipeline.dispose(); } catch (_) {}
  }
  currentPipeline = null;
  currentPipelineKey = '';
}

async function loadTranscriber(modelId, device) {
  const key = `${modelId}|${device}`;
  if (currentPipeline && currentPipelineKey === key) return currentPipeline;

  await disposePipeline();
  setProgress(0, 'Načítám model', `Backend: ${device.toUpperCase()}`);
  setStatus(`Načítám model přes ${device.toUpperCase()}…`);

  const progressByFile = new Map();
  const progressCallback = (data) => {
    if ((data.status === 'progress' || data.status === 'progress_total') && typeof data.progress === 'number') {
      const keyName = data.file ?? data.name ?? `část-${progressByFile.size}`;
      progressByFile.set(keyName, data.progress);
      const values = [...progressByFile.values()];
      const average = values.reduce((sum, item) => sum + item, 0) / Math.max(1, values.length);
      setProgress(average, 'Stahuji model', `${Math.round(average)} % • ${device.toUpperCase()}`);
    }
  };

  currentPipeline = await pipeline('automatic-speech-recognition', modelId, {
    device,
    progress_callback: progressCallback,
  });
  currentPipelineKey = key;
  setProgress(100, 'Model je připraven', `Backend: ${device.toUpperCase()}`);
  return currentPipeline;
}

async function decodeAudio(file) {
  setProgress(0, 'Připravuji zvuk', `Načítám ${prettyBytes(file.size)}.`);
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error('Prohlížeč nepodporuje Web Audio API.');

  let context;
  try {
    context = new AudioContextClass({ sampleRate: SAMPLE_RATE });
  } catch (_) {
    context = new AudioContextClass();
  }

  try {
    const arrayBuffer = await file.arrayBuffer();
    setProgress(35, 'Připravuji zvuk', 'Dekóduji zvuk…');
    const decoded = await context.decodeAudioData(arrayBuffer.slice(0));
    const mono = new Float32Array(decoded.length);

    setProgress(65, 'Připravuji zvuk', 'Převádím zvuk do jednoho kanálu.');
    for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
      const channelData = decoded.getChannelData(channel);
      for (let index = 0; index < channelData.length; index++) {
        mono[index] += channelData[index] / decoded.numberOfChannels;
      }
    }

    let output = mono;
    if (decoded.sampleRate !== SAMPLE_RATE) {
      setProgress(82, 'Připravuji zvuk', `Převádím ${decoded.sampleRate} Hz na ${SAMPLE_RATE} Hz.`);
      output = resampleLinear(mono, decoded.sampleRate, SAMPLE_RATE);
    }

    audioDuration = output.length / SAMPLE_RATE;
    setProgress(100, 'Zvuk je připraven', `${prettyTime(audioDuration)} zvuku.`);
    return output;
  } finally {
    await context.close();
  }
}

function resampleLinear(input, inputRate, outputRate) {
  if (inputRate === outputRate) return input;
  const ratio = inputRate / outputRate;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(outputLength);

  for (let index = 0; index < outputLength; index++) {
    const position = index * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const fraction = position - left;
    output[index] = input[left] * (1 - fraction) + input[right] * fraction;
  }
  return output;
}

function audioLevel(audio) {
  if (!audio.length) return { rms: 0, peak: 0 };
  const step = Math.max(1, Math.floor(audio.length / 6000));
  let sum = 0;
  let peak = 0;
  let samples = 0;

  for (let index = 0; index < audio.length; index += step) {
    const value = Math.abs(audio[index]);
    sum += value * value;
    peak = Math.max(peak, value);
    samples++;
  }

  return { rms: Math.sqrt(sum / Math.max(1, samples)), peak };
}

function isSilent(audio) {
  const level = audioLevel(audio);
  return level.rms < 0.0008 && level.peak < 0.008;
}

function normalizedTokens(text) {
  return cleanText(text)
    .toLocaleLowerCase('cs')
    .match(/[\p{L}\p{N}]+/gu) ?? [];
}

function qualityReport(text) {
  const tokens = normalizedTokens(text);
  if (tokens.length < 8) return { bad: false, reason: '', tokens: tokens.length };

  const counts = new Map();
  let longestRun = 1;
  let currentRun = 1;
  let singleCharacter = 0;

  tokens.forEach((token, index) => {
    counts.set(token, (counts.get(token) ?? 0) + 1);
    if ([...token].length === 1) singleCharacter++;
    if (index > 0 && token === tokens[index - 1]) {
      currentRun++;
      longestRun = Math.max(longestRun, currentRun);
    } else {
      currentRun = 1;
    }
  });

  const dominantCount = Math.max(...counts.values());
  const dominantRatio = dominantCount / tokens.length;
  const uniqueRatio = counts.size / tokens.length;
  const singleRatio = singleCharacter / tokens.length;

  if (longestRun >= 6) {
    return { bad: true, reason: 'stejný výraz se opakuje mnohokrát za sebou', tokens: tokens.length };
  }
  if (tokens.length >= 16 && dominantRatio >= 0.42 && uniqueRatio <= 0.32) {
    return { bad: true, reason: 'jeden výraz tvoří nepřirozeně velkou část přepisu', tokens: tokens.length };
  }
  if (tokens.length >= 20 && singleRatio >= 0.62 && uniqueRatio <= 0.35) {
    return { bad: true, reason: 'výstup tvoří převážně jednotlivá opakovaná písmena', tokens: tokens.length };
  }
  if (tokens.length >= 25 && singleRatio >= 0.75) {
    return { bad: true, reason: 'výstup tvoří téměř jen jednotlivá písmena', tokens: tokens.length };
  }
  return { bad: false, reason: '', tokens: tokens.length };
}

function assertSaneText(text, label = 'Výstup') {
  const report = qualityReport(text);
  if (report.bad) {
    throw new DegenerateOutputError(`${label} byl odmítnut: ${report.reason}.`);
  }
}

function isRecoverableDecodeError(error) {
  return /token_ids must be a non-empty array|timestamp|cross attentions?|ending timestamp|empty array/i.test(errorMessage(error));
}

function transcriberOptions(language, timestamps) {
  const options = {
    task: 'transcribe',
    force_full_sequences: false,
  };
  if (timestamps) options.return_timestamps = true;
  if (language) options.language = language;
  return options;
}

async function transcribePiece(transcriber, audio, language, device, depth = 0) {
  if (isSilent(audio)) return { text: '', chunks: [], silence: true };

  let timestampError = null;
  try {
    const result = await transcriber(audio, transcriberOptions(language, true));
    const text = cleanText(result?.text);
    if (text) {
      assertSaneText(text, 'Část přepisu');
      return result;
    }
  } catch (error) {
    if (error instanceof DegenerateOutputError) {
      if (device === 'webgpu') throw new BackendRetryError(error.message, error);
      throw error;
    }
    if (!isRecoverableDecodeError(error)) throw error;
    timestampError = error;
  }

  if (device === 'webgpu' && timestampError && /token_ids must be a non-empty array/i.test(errorMessage(timestampError))) {
    throw new BackendRetryError('WebGPU vrátil prázdné tokeny.', timestampError);
  }

  try {
    const result = await transcriber(audio, transcriberOptions(language, false));
    const text = cleanText(result?.text);
    if (text) {
      assertSaneText(text, 'Část přepisu');
      return result;
    }
  } catch (error) {
    if (error instanceof DegenerateOutputError) {
      if (device === 'webgpu') throw new BackendRetryError(error.message, error);
      throw error;
    }
    if (!isRecoverableDecodeError(error)) throw error;
    if (device === 'webgpu') throw new BackendRetryError('WebGPU nedokázal dekódovat část zvuku.', error);
  }

  const minimumSamples = Math.round(MIN_SPLIT_SECONDS * SAMPLE_RATE);
  if (depth < MAX_SPLIT_DEPTH && audio.length >= minimumSamples * 2) {
    const midpoint = Math.floor(audio.length / 2);
    const left = await transcribePiece(transcriber, audio.slice(0, midpoint), language, device, depth + 1);
    const right = await transcribePiece(transcriber, audio.slice(midpoint), language, device, depth + 1);
    const leftText = cleanText(left.text);
    let rightText = cleanText(right.text);
    if (leftText && rightText) rightText = removeRepeatedBoundary(leftText, rightText);
    const combinedText = cleanText(`${leftText} ${rightText}`);
    const shiftedRightChunks = (right.chunks ?? []).map((chunk) => ({
      ...chunk,
      timestamp: Array.isArray(chunk.timestamp)
        ? [
            Number.isFinite(chunk.timestamp[0]) ? chunk.timestamp[0] + midpoint / SAMPLE_RATE : chunk.timestamp[0],
            Number.isFinite(chunk.timestamp[1]) ? chunk.timestamp[1] + midpoint / SAMPLE_RATE : chunk.timestamp[1],
          ]
        : chunk.timestamp,
    }));

    if (combinedText) {
      assertSaneText(combinedText, 'Rozdělená část přepisu');
      return {
        text: combinedText,
        chunks: [...(left.chunks ?? []), ...shiftedRightChunks],
      };
    }
  }

  if (isSilent(audio)) return { text: '', chunks: [], silence: true };
  throw new Error('Model nedokázal přepsat slyšitelný úsek. Zkus model Base nebo jiný backend.');
}

function normalizeChunks(chunks, chunkStart, chunkEnd, acceptFrom, fallbackText) {
  const normalized = [];
  let lastEnd = chunkStart;

  for (const chunk of chunks ?? []) {
    const text = cleanText(chunk.text);
    if (!text) continue;

    const timestamp = Array.isArray(chunk.timestamp) ? chunk.timestamp : [0, null];
    let start = Number.isFinite(timestamp[0]) ? chunkStart + timestamp[0] : lastEnd;
    let end = Number.isFinite(timestamp[1])
      ? chunkStart + timestamp[1]
      : Math.min(chunkEnd, start + Math.max(0.8, text.length / 13));

    start = Math.max(chunkStart, start);
    end = Math.min(chunkEnd, Math.max(start + 0.08, end));
    if (end <= acceptFrom || (start < acceptFrom && (start + end) / 2 <= acceptFrom)) continue;

    start = Math.max(start, acceptFrom);
    lastEnd = end;
    normalized.push({ text, start, end });
  }

  if (!normalized.length && fallbackText) {
    normalized.push({
      text: cleanText(fallbackText),
      start: acceptFrom,
      end: chunkEnd,
    });
  }
  return normalized;
}

function removeRepeatedBoundary(previousText, currentText) {
  const previousWords = cleanText(previousText).split(' ').filter(Boolean);
  const currentWords = cleanText(currentText).split(' ').filter(Boolean);
  const maximum = Math.min(12, previousWords.length, currentWords.length);

  for (let count = maximum; count >= 2; count--) {
    const suffix = previousWords.slice(-count).join(' ').toLocaleLowerCase('cs');
    const prefix = currentWords.slice(0, count).join(' ').toLocaleLowerCase('cs');
    if (suffix === prefix) return currentWords.slice(count).join(' ');
  }
  return currentWords.join(' ');
}

function mergeSegments(target, additions) {
  for (const original of additions) {
    const addition = { ...original };
    const previous = target[target.length - 1];
    if (previous && addition.start <= previous.end + 1.5) {
      addition.text = removeRepeatedBoundary(previous.text, addition.text);
    }
    addition.text = cleanText(addition.text);
    if (!addition.text) continue;
    if (previous && addition.start < previous.end) {
      addition.start = Math.max(previous.start + 0.05, previous.end - 0.05);
    }
    if (addition.end <= addition.start) addition.end = addition.start + 0.5;
    target.push(addition);
  }
}

function settingsFromForm() {
  return {
    maxDuration: Number($('maxdur').value),
    charsPerLine: Number($('chars').value),
    lineCount: Number($('lines').value),
  };
}

async function transcribeAudio(transcriber, audio, language, device, onUpdate) {
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
    const completedBefore = (chunkStart / audioDuration) * 100;

    setProgress(
      completedBefore,
      'Přepisuji zvuk',
      `${prettyTime(chunkStart)} z ${prettyTime(audioDuration)} • část ${index + 1} z ${totalChunks} • ${device.toUpperCase()}`,
    );
    setStatus(`Přepisuji část ${index + 1} z ${totalChunks}. Text se průběžně doplňuje.`);

    const result = await transcribePiece(
      transcriber,
      audio.slice(startSample, endSample),
      language,
      device,
    );

    const resultText = cleanText(result?.text);
    const acceptFrom = index === 0 ? chunkStart : chunkStart + OVERLAP_SECONDS;
    const chunkSegments = normalizeChunks(
      result?.chunks,
      chunkStart,
      chunkEnd,
      acceptFrom,
      resultText,
    );
    mergeSegments(segments, chunkSegments);

    const partialText = cleanText(segments.map((segment) => segment.text).join(' '));
    assertSaneText(partialText, 'Průběžný přepis');

    const completedAfter = (chunkEnd / audioDuration) * 100;
    onUpdate({
      text: partialText,
      segments: [...segments],
      completedSeconds: chunkEnd,
      completedPercent: completedAfter,
      chunkIndex: index + 1,
      totalChunks,
    });

    setProgress(
      completedAfter,
      'Přepisuji zvuk',
      `${prettyTime(chunkEnd)} z ${prettyTime(audioDuration)} • hotovo ${index + 1} z ${totalChunks} částí • ${device.toUpperCase()}`,
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  const text = cleanText(segments.map((segment) => segment.text).join(' '));
  assertSaneText(text, 'Celý přepis');
  return { text, segments };
}

function wrapCaption(text, charsPerLine, lineCount) {
  const cleaned = cleanText(text);
  if (lineCount === 1 || cleaned.length <= charsPerLine) return cleaned;

  const words = cleaned.split(' ');
  const targetLength = Math.ceil(cleaned.length / lineCount);
  const lines = [];
  let current = '';

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (lines.length < lineCount - 1 && current && candidate.length > targetLength) {
      lines.push(current);
      current = word;
    } else {
      current = candidate;
    }
  }
  if (current) lines.push(current);
  return lines.join('\n');
}

function segmentsToWords(segments) {
  const words = [];
  for (const segment of segments) {
    const pieces = cleanText(segment.text).split(/\s+/).filter(Boolean);
    if (!pieces.length) continue;

    const duration = Math.max(0.4, segment.end - segment.start);
    const weights = pieces.map((word) => Math.max(1, word.replace(/[^\p{L}\p{N}]/gu, '').length));
    const totalWeight = weights.reduce((sum, value) => sum + value, 0);
    let cursor = segment.start;

    pieces.forEach((word, index) => {
      const isLast = index === pieces.length - 1;
      const wordDuration = duration * (weights[index] / totalWeight);
      const end = isLast ? segment.end : Math.min(segment.end, cursor + wordDuration);
      words.push({ text: word, start: cursor, end: Math.max(cursor + 0.04, end) });
      cursor = end;
    });
  }
  return words;
}

function buildCaptions(segments, settings) {
  const words = segmentsToWords(segments);
  const captions = [];
  const maxChars = settings.charsPerLine * settings.lineCount;
  let current = null;

  const flush = () => {
    if (!current) return;
    captions.push({
      start: current.start,
      end: Math.max(current.end, current.start + 0.65),
      text: wrapCaption(current.text, settings.charsPerLine, settings.lineCount),
    });
    current = null;
  };

  for (const word of words) {
    if (!current) {
      current = { ...word };
      continue;
    }

    const candidate = cleanText(`${current.text} ${word.text}`);
    const tooLong = candidate.length > maxChars || word.end - current.start > settings.maxDuration;
    if (tooLong) flush();

    if (!current) {
      current = { ...word };
    } else {
      current.text = candidate;
      current.end = word.end;
    }

    if (/[.!?…][”"')\]]?$/.test(current.text) && current.end - current.start >= 1.1) flush();
  }
  flush();

  for (let index = 0; index < captions.length - 1; index++) {
    captions[index].end = Math.min(
      captions[index].end,
      Math.max(captions[index].start + 0.35, captions[index + 1].start - 0.04),
    );
  }
  return captions;
}

function formatSrtTime(seconds) {
  const totalMs = Math.max(0, Math.round(seconds * 1000));
  const milliseconds = totalMs % 1000;
  const totalSeconds = Math.floor(totalMs / 1000);
  const secs = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const minutes = totalMinutes % 60;
  const hours = Math.floor(totalMinutes / 60);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')},${String(milliseconds).padStart(3, '0')}`;
}

function captionsToSrt(captions) {
  return captions.map((caption, index) =>
    `${index + 1}\n${formatSrtTime(caption.start)} --> ${formatSrtTime(caption.end)}\n${caption.text}`,
  ).join('\n\n');
}

function renderPartial({ text, segments, completedSeconds, chunkIndex, totalChunks }) {
  const captions = buildCaptions(segments, settingsFromForm());
  txtOutput.value = text;
  srtOutput.value = captionsToSrt(captions);
  count.textContent = `${captions.length} titulků • průběžně ${chunkIndex}/${totalChunks}`;

  txtOutput.scrollTop = txtOutput.scrollHeight;
  srtOutput.scrollTop = srtOutput.scrollHeight;
  setStatus(`Průběžný přepis je zobrazen do času ${prettyTime(completedSeconds)}.`, 'warn');
}

function clearOutput(message = '0 titulků') {
  srtOutput.value = '';
  txtOutput.value = '';
  count.textContent = message;
  saveSrt.disabled = true;
  saveTxt.disabled = true;
  copyButton.disabled = true;
}

function setBusy(busy) {
  startButton.disabled = busy || !selectedFile;
  fileInput.disabled = busy;
  for (const id of ['model', 'language', 'device', 'maxdur', 'chars', 'lines']) {
    $(id).disabled = busy;
  }
}

async function runWithDevice(audio, modelId, language, device) {
  const transcriber = await loadTranscriber(modelId, device);
  return transcribeAudio(transcriber, audio, language, device, renderPartial);
}

startButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  setBusy(true);
  clearOutput();

  try {
    const modelId = $('model').value;
    const language = $('language').value;
    const firstDevice = requestedDevice();
    const audio = await decodeAudio(selectedFile);
    let result;
    let finalDevice = firstDevice;

    try {
      result = await runWithDevice(audio, modelId, language, firstDevice);
    } catch (error) {
      if (firstDevice !== 'webgpu') throw error;

      console.warn('WebGPU přepis selhal, opakuji přes WASM:', error);
      setStatus('WebGPU vytvořilo chybu nebo vadný text. Přepis začíná znovu přes stabilní WASM.', 'warn');
      setProgress(0, 'Přepínám na WASM', errorMessage(error));
      clearOutput('Přepínám na WASM…');
      await disposePipeline();
      finalDevice = 'wasm';
      result = await runWithDevice(audio, modelId, language, 'wasm');
    }

    assertSaneText(result.text, 'Výsledný přepis');
    const captions = buildCaptions(result.segments, settingsFromForm());
    if (!result.text || !captions.length) {
      throw new Error('Model nevrátil použitelný přepis. Zkus model Base nebo zkontroluj hlasitost.');
    }

    txtOutput.value = result.text;
    srtOutput.value = captionsToSrt(captions);
    count.textContent = `${captions.length} titulků`;
    saveSrt.disabled = false;
    saveTxt.disabled = false;
    copyButton.disabled = false;
    setProgress(100, 'Hotovo', `Vytvořeno ${captions.length} titulků • ${finalDevice.toUpperCase()}`);
    setStatus(`Hotovo. Vytvořeno ${captions.length} titulků.`, 'ok');
  } catch (error) {
    console.error(error);
    stage.textContent = 'Přepis selhal';
    detail.textContent = errorMessage(error);
    clearOutput('Výstup byl odmítnut');
    setStatus(`Přepis se nepodařil: ${errorMessage(error)}`, 'error');
  } finally {
    setBusy(false);
  }
});

function baseFileName() {
  return (selectedFile?.name ?? 'prepis')
    .replace(/\.[^.]+$/, '')
    .replace(/[\\/:*?"<>|]+/g, '_');
}

function downloadText(content, extension, mime) {
  const blob = new Blob(['\ufeff', content], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${baseFileName()}.${extension}`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

saveSrt.addEventListener('click', () => downloadText(srtOutput.value, 'srt', 'application/x-subrip'));
saveTxt.addEventListener('click', () => downloadText(txtOutput.value, 'txt', 'text/plain'));

copyButton.addEventListener('click', async () => {
  const value = activeTab === 'srt' ? srtOutput.value : txtOutput.value;
  try {
    await navigator.clipboard.writeText(value);
  } catch (_) {
    const area = activeTab === 'srt' ? srtOutput : txtOutput;
    area.focus();
    area.select();
    document.execCommand('copy');
  }
  setStatus('Text byl zkopírován.', 'ok');
});

function switchTab(tab) {
  activeTab = tab;
  const isSrt = tab === 'srt';
  srtOutput.classList.toggle('hidden', !isSrt);
  txtOutput.classList.toggle('hidden', isSrt);
  $('srtTab').classList.toggle('active', isSrt);
  $('txtTab').classList.toggle('active', !isSrt);
}

$('srtTab').addEventListener('click', () => switchTab('srt'));
$('txtTab').addEventListener('click', () => switchTab('txt'));

window.addEventListener('beforeunload', () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
