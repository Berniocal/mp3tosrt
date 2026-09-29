import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

env.allowLocalModels = false;
env.useBrowserCache = true;
if ('useWasmCache' in env) env.useWasmCache = true;

let currentPipeline = null;
let currentKey = '';

function cleanText(value) {
  return String(value ?? '')
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.;:!?])/g, '$1')
    .trim();
}

async function disposePipeline() {
  if (currentPipeline?.dispose) {
    try { await currentPipeline.dispose(); } catch (_) {}
  }
  currentPipeline = null;
  currentKey = '';
}

async function ensurePipeline(modelId, device) {
  const key = `${modelId}|${device}`;
  if (currentPipeline && currentKey === key) return currentPipeline;

  await disposePipeline();
  const progressByFile = new Map();

  currentPipeline = await pipeline('automatic-speech-recognition', modelId, {
    device,
    progress_callback: (data) => {
      if ((data.status === 'progress' || data.status === 'progress_total') && typeof data.progress === 'number') {
        const name = data.file ?? data.name ?? `část-${progressByFile.size}`;
        progressByFile.set(name, data.progress);
        const values = [...progressByFile.values()];
        const average = values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
        self.postMessage({ type: 'model-progress', progress: average, device });
      }
    },
  });

  currentKey = key;
  return currentPipeline;
}

function serializableResult(result) {
  return {
    text: cleanText(result?.text),
    chunks: (result?.chunks ?? []).map((chunk) => ({
      text: cleanText(chunk?.text),
      timestamp: Array.isArray(chunk?.timestamp)
        ? [
            Number.isFinite(chunk.timestamp[0]) ? chunk.timestamp[0] : 0,
            Number.isFinite(chunk.timestamp[1]) ? chunk.timestamp[1] : null,
          ]
        : [0, null],
    })),
  };
}

self.addEventListener('message', async (event) => {
  const message = event.data ?? {};
  const { id, type } = message;

  try {
    if (type === 'load') {
      await ensurePipeline(message.modelId, message.device);
      self.postMessage({ type: 'result', id, result: { ready: true } });
      return;
    }

    if (type === 'dispose') {
      await disposePipeline();
      self.postMessage({ type: 'result', id, result: { disposed: true } });
      return;
    }

    if (type === 'transcribe') {
      const transcriber = await ensurePipeline(message.modelId, message.device);
      const audio = message.audio instanceof Float32Array
        ? message.audio
        : new Float32Array(message.audio);

      const options = {
        task: 'transcribe',
        force_full_sequences: false,
        return_timestamps: true,
      };
      if (message.language) options.language = message.language;

      const result = await transcriber(audio, options);
      self.postMessage({ type: 'result', id, result: serializableResult(result) });
      return;
    }

    throw new Error(`Neznámý příkaz workeru: ${type}`);
  } catch (error) {
    self.postMessage({
      type: 'error',
      id,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});
