<script setup>
import { computed, onBeforeUnmount, onMounted } from 'vue';
import { api } from './api';
import { useLocale } from './i18n';
import { store } from './store';
import WorkflowStepper from './components/WorkflowStepper.vue';
import DatasetStep from './components/steps/DatasetStep.vue';
import VectorStep from './components/steps/VectorStep.vue';
import LlmStep from './components/steps/LlmStep.vue';
import RetrievalStep from './components/steps/RetrievalStep.vue';
import MetricsStep from './components/steps/MetricsStep.vue';
import ResultsStep from './components/steps/ResultsStep.vue';
import logoUrl from '../../static/logo-mark.png';

const { lang, t, format } = useLocale();
const stepComponents = [DatasetStep, VectorStep, LlmStep, RetrievalStep, MetricsStep, ResultsStep];
const currentStep = computed(() => stepComponents[store.step - 1]);
const canGotoNext = computed(() => {
  if (store.step === 1) {
    if (!store.dataset) return false;
    return !store.presetDataset || (
      store.dataset.source === 'preset' && store.dataset.name === store.presetDataset
    );
  }
  if (store.step === 5) return store.selectedMetrics.length > 0;
  return store.step >= 2 && store.step <= 4;
});

let toastTimer = null;
function dismissToast() {
  store.toastError = '';
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = null;
}
function setError(message) {
  store.error = message;
  store.notice = '';
  dismissToast();
  if (!message) return;
  store.toastError = message;
  toastTimer = setTimeout(dismissToast, 5000);
}
function setNotice(message) { store.notice = message; store.error = ''; }
function gotoStep(step) { if (step >= 1 && step <= 6) store.step = step; }

function datasetDisplayName(name) {
  return store.options?.preset_datasets?.find(item => item.id === name)?.label || name;
}

function datasetErrorMessage(error, fallbackName) {
  const name = datasetDisplayName(error.details?.dataset_name || fallbackName);
  if (error.code === 'dataset_not_found') return format(t.value.messages.datasetNotFound, { name });
  if (error.code === 'dataset_load_failed') return format(t.value.messages.datasetLoadFailed, { name });
  return error.message || t.value.messages.datasetLoadFailed;
}

async function init() {
  try {
    const [capabilities, config] = await Promise.all([api.capabilities(), api.config()]);
    const options = capabilities.options;
    store.options = options;
    store.capabilities = capabilities;
    store.config = config;
    const llm = config.llm;
    const embedding = config.embedding;
    const chunk = config.chunk;
    const retrieval = config.retrieval;
    Object.assign(store, {
      llm: llm.llm || 'openai', apiKey: llm.api_key || '', apiBase: llm.api_base || '',
      apiName: llm.api_name || '', authToken: llm.auth_token || '',
      hfModel: llm.huggingface_model || 'llama', ollamaModel: llm.ollama_model || '',
      ollamaTimeout: llm.ollama_request_timeout || 60, temperature: llm.temperature || 0,
      embeddings: embedding.embeddings || options.embeddings[0], splitType: chunk.split_type || 'sentence',
      chunkSize: chunk.chunk_size || 512, chunkOverlap: chunk.chunk_overlap ?? 20,
      windowSize: chunk.window_size || 3, chunkSizes: (chunk.chunk_sizes || [2048, 512, 128]).join(', '),
      persistDir: chunk.persist_dir || 'storage',
      orchestrator: retrieval.orchestrator || 'default', retriever: retrieval.retriever || 'BM25', retrieverMode: retrieval.retriever_mode || 0,
      queryTransform: retrieval.query_transform || 'none', postprocessRerank: retrieval.postprocess_rerank || 'none',
      responseSynthesizer: retrieval.response_synthesizer || 'refine', similarityTopK: retrieval.similarity_top_k || 3,
      textQaTemplate: config.templates?.text_qa_template || options.default_templates.text_qa_template,
      refineTemplate: config.templates?.refine_template || options.default_templates.refine_template,
    });
    const quickPreset = options.metric_presets?.find(preset => preset.id === 'quick');
    store.selectedMetrics = quickPreset ? [...quickPreset.metric_ids] : [];
    store.metricPreset = quickPreset ? 'quick' : '';
  } catch (error) {
    setError(`${t.value.messages.initFailed} ${error.message}`);
  }
}

async function updateVector() {
  const chunkSizes = store.chunkSizes.split(',').map(value => Number(value.trim())).filter(Number.isFinite);
  await api.updateVector({ embeddings: store.embeddings, split_type: store.splitType, chunk_size: store.chunkSize, chunk_overlap: store.chunkOverlap, window_size: store.windowSize, chunk_sizes: chunkSizes, persist_dir: store.persistDir });
}
async function updateLlm() {
  await api.updateLLM({ llm: store.llm, api_key: store.apiKey || undefined, api_base: store.apiBase || undefined, api_name: store.apiName || undefined, auth_token: store.authToken || undefined, huggingface_model: store.hfModel || undefined, ollama_model: store.ollamaModel || undefined, ollama_request_timeout: store.ollamaTimeout || undefined, temperature: store.temperature });
}
async function updateRetrieval() {
  await api.updateRetrieval({ orchestrator: store.orchestrator, retriever: store.retriever, retriever_mode: store.retrieverMode, similarity_top_k: store.similarityTopK, query_transform: store.queryTransform, postprocess_rerank: store.postprocessRerank, response_synthesizer: store.responseSynthesizer, text_qa_template: store.textQaTemplate, refine_template: store.refineTemplate });
}

async function next() {
  store.error = '';
  try {
    if (store.step === 1 && !store.dataset) return setError(t.value.messages.chooseDataset);
    if (store.step === 2) { await updateVector(); setNotice(t.value.messages.buildingIndex); await api.buildIndex(); setNotice(t.value.messages.indexBuilt); }
    if (store.step === 3) { await updateLlm(); setNotice(t.value.messages.llmSaved); }
    if (store.step === 4) { await updateRetrieval(); setNotice(t.value.messages.buildingEngine); await api.buildQueryEngine(); setNotice(t.value.messages.engineReady); }
    if (store.step === 5) { if (!store.selectedMetrics.length) return setError(t.value.messages.chooseMetric); await startEvaluation(); return; }
    gotoStep(store.step + 1);
  } catch (error) { setError(error.message || String(error)); }
}

async function loadPreset(name) {
  store.loading = true; store.presetLoading = true; setError('');
  try { const result = await api.presetDataset(name); store.dataset = result.dataset; setNotice(format(t.value.messages.loaded, { name: result.dataset.display })); }
  catch (error) { setError(datasetErrorMessage(error, name)); }
  finally { store.loading = false; store.presetLoading = false; }
}
async function uploadJson() {
  if (!store.uploadFile) return setError(t.value.messages.chooseJson);
  store.loading = true; setError('');
  try { const result = await api.uploadJson(store.uploadFile); store.dataset = result.dataset; store.presetDataset = ''; setNotice(format(t.value.messages.uploaded, { name: store.uploadFile.name })); }
  catch (error) { setError(error.message); }
  finally { store.loading = false; }
}
async function generateFromFolder() {
  store.loading = true; setError('');
  try {
    const result = await api.fromFolder({ folder_path: store.folderPath, output_json: store.folderOutput, num_questions: store.folderNum, sentence_length: store.folderSentenceLen });
    store.dataset = result.dataset;
    store.presetDataset = '';
    setNotice(format(t.value.messages.generated, { count: result.num_generated_qa_pairs }));
  } catch (error) { setError(error.message); }
  finally { store.loading = false; }
}

function closeEvalStream() { store.evalStream?.close(); store.evalStream = null; }
function handleEvalEvent(data) {
  if (data.event === 'started') { store.evalTotal = data.total; store.evalCompleted = 0; }
  if (data.event === 'progress') { Object.assign(store, { evalCompleted: data.completed, evalTotal: data.total, evalProgress: data.progress }); if (data.sample) store.evalSamples.push(data.sample); if (data.summary) store.evalSummary = data.summary; }
  if (data.event === 'sample_error') store.evalError = data.error || t.value.messages.sampleFailed;
  if (data.event === 'done') { store.evalStatus = 'done'; store.evalProgress = 1; if (data.summary) store.evalSummary = data.summary; closeEvalStream(); }
  if (data.event === 'error') { store.evalStatus = 'error'; store.evalError = data.error; closeEvalStream(); }
  if (data.event === 'cancelled') { store.evalStatus = 'cancelled'; closeEvalStream(); }
}
function subscribeToEvaluation(taskId) {
  closeEvalStream();
  const stream = new EventSource(`/api/evaluate/${taskId}/stream`);
  store.evalStream = stream;
  stream.onmessage = event => { try { handleEvalEvent(JSON.parse(event.data)); } catch { /* Ignore malformed events. */ } };
}
async function startEvaluation() {
  Object.assign(store, { evalStatus: 'running', evalProgress: 0, evalCompleted: 0, evalTotal: 0, evalSamples: [], evalSummary: null, evalError: '' });
  try { const result = await api.startEvaluation({ metrics: store.selectedMetrics, num_samples: store.numSamples, experiment_1: store.experiment1 }); store.experimentId = result.experiment_id; store.evalTaskId = result.task_id; subscribeToEvaluation(result.task_id); gotoStep(6); }
  catch (error) { store.evalStatus = 'error'; store.evalError = error.message; setError(error.message); }
}
async function cancelEvaluation() { if (store.evalTaskId) await api.cancelEvaluation(store.evalTaskId); }

onMounted(init);
onBeforeUnmount(() => { closeEvalStream(); dismissToast(); });
</script>

<template>
  <div class="app-shell">
    <Transition name="toast">
      <div v-if="store.toastError" class="toast toast-error" role="alert" aria-live="assertive">
        <span class="toast-icon" aria-hidden="true">!</span>
        <span class="toast-message">{{ store.toastError }}</span>
        <button class="toast-close" type="button" :aria-label="t.common.close" @click="dismissToast">×</button>
      </div>
    </Transition>
    <header class="app-header"><div class="container">
        <div class="brand" lang="en"><img class="brand-logo" :src="logoUrl" alt="XRAG" />XRAG</div>
        <nav class="nav">
          <a href="#workflow">{{ t.nav.workflow }}</a>
          <!--
          <a href="#features">{{ t.nav.features }}</a>
          <a href="#demo">{{ t.nav.demo }}</a>
          -->
          <a href="https://github.com/DocAILab/XRAG" target="_blank" rel="noopener">{{ t.nav.github }}</a>
          <div class="lang-toggle">
            <button :class="{ active: lang === 'zh' }" @click="lang = 'zh'">中</button>
            <button :class="{ active: lang === 'en' }" @click="lang = 'en'">EN</button>
          </div>
        </nav>
      </div></header>
    <main id="workflow" class="container">
      <div class="hero">
        <h1 lang="en">{{ t.hero.title }}</h1>
        <p class="subtitle">{{ t.hero.subtitle }}</p>
      </div>
      <h2 class="section-heading">{{ t.hero.demo }}</h2>
      <WorkflowStepper :step="store.step" :steps="t.steps" @goto="gotoStep" />
      <div v-if="store.error" class="alert alert-error">{{ store.error }}</div>
      <div v-if="store.notice" class="alert alert-info">{{ store.notice }}</div>
      <component :is="currentStep" :store="store" :options="store.options" :t="t" @load-preset="loadPreset"
        @upload-json="uploadJson" @generate-folder="generateFromFolder" @restart="gotoStep(1)"
        @cancel="cancelEvaluation" @error="setError" @notice="setNotice" />
      <div v-if="store.step <= 5" class="nav-buttons"><button class="button button-secondary"
          :disabled="store.step === 1 || store.loading" @click="gotoStep(store.step - 1)">{{ t.common.previous
          }}</button><button class="button button-primary" :disabled="!canGotoNext || store.loading" @click="next"><span
            v-if="store.loading">{{ t.common.loading }}</span><span v-else-if="store.step === 5">{{
              t.common.runEvaluation }}</span><span v-else>{{ t.common.next }}</span></button></div>
    </main>
    <footer class="app-footer">
      <div class="container"><span>{{ t.footer.copyright }}</span>
        <div><a href="https://github.com/DocAILab/XRAG" target="_blank" rel="noopener">{{ t.footer.github }}</a><a
            href="https://docailab.github.io/XRAG/" target="_blank" rel="noopener">{{ t.footer.docs }}</a></div>
      </div>
    </footer>
  </div>
</template>
