<script setup>
import { ref } from 'vue';
import { api } from '../../api';
import ModelCombobox from '../ModelCombobox.vue';
import chevronUrl from '../../../../static/chevron-down.svg';
const props = defineProps({ store: Object, t: Object });
const emit = defineEmits(['build-index', 'load-index', 'error', 'notice']);
const buildExpanded = ref(true);
const loadExpanded = ref(false);
const localModels = ref([]);
const loadingLocalModels = ref(false);
const apiModels = ref([]);
const loadingApiModels = ref(false);
async function queryApiModels() {
  loadingApiModels.value = true;
  try {
    const apiKey = props.store.embeddingApiKey.trim();
    if (!apiKey) throw new Error('Missing embedding API key');
    const result = await api.listLLMModels({
      api_key: apiKey,
      api_base: props.store.embeddingApiBase.trim() || 'https://api.openai.com/v1',
    });
    apiModels.value = result.models;
    emit('notice', props.t.llm.modelsFound.replace('{count}', result.models.length));
  } catch {
    apiModels.value = [];
    emit('error', props.t.llm.modelsQueryFailed);
  } finally {
    loadingApiModels.value = false;
  }
}
async function queryLocalModels() {
  loadingLocalModels.value = true;
  try {
    const result = await api.listLocalEmbeddingModels();
    localModels.value = result.models || [];
    emit('notice', props.t.vector.localModelsFound.replace('{count}', localModels.value.length));
  } catch {
    localModels.value = [];
    emit('error', props.t.vector.localModelsQueryFailed);
  } finally {
    loadingLocalModels.value = false;
  }
}
</script>
<template>
  <section class="card">
    <h2>{{ t.vector.title }}</h2>
    <div v-if="store.indexStatus === 'done'" class="alert alert-success step-ready-summary" role="status">
      {{ t.vector.ready }}
    </div>
    <button class="custom-dataset-toggle" type="button" :aria-expanded="buildExpanded"
      aria-controls="vector-build-panel" @click="buildExpanded = !buildExpanded">
      <span>{{ t.vector.buildTitle }}</span>
      <span class="collapse-chevron" :class="{ 'is-expanded': buildExpanded }" aria-hidden="true"><img class="icon" :src="chevronUrl" alt="" /></span>
    </button>
    <div v-if="buildExpanded" id="vector-build-panel" class="vector-collapse-panel">
      <div class="form-row"><label class="field-label">{{ t.vector.backend }}</label><select v-model="store.embeddingType"><option v-for="opt in store.options?.embedding_backends || ['local', 'openai']" :key="opt" :value="opt">{{ t.vector.backends[opt] || opt }}</option></select></div>
      <div class="form-row">
        <label class="field-label" for="embedding-model">{{ t.vector.embedding }}</label>
        <div v-if="store.embeddingType === 'local'" class="llm-model-control">
          <ModelCombobox id="embedding-model" v-model="store.embeddings" :options="localModels.length ? localModels : (store.options?.embeddings || [])" :label="t.vector.embedding" placeholder="BAAI/bge-large-en-v1.5" />
          <button class="button button-primary" type="button" :disabled="loadingLocalModels" @click="queryLocalModels">{{ loadingLocalModels ? t.common.loading : t.vector.queryLocalModels }}</button>
        </div>
        <div v-else class="llm-model-control">
          <ModelCombobox id="embedding-model" v-model="store.embeddings" :options="apiModels" :label="t.vector.embedding" placeholder="text-embedding-3-small" />
          <button class="button button-primary" type="button" :disabled="loadingApiModels" @click="queryApiModels">{{ loadingApiModels ? t.common.loading : t.llm.queryModels }}</button>
        </div>
      </div>
      <template v-if="store.embeddingType === 'openai'">
        <div class="form-row"><label class="field-label">{{ t.vector.apiBase }}</label><input v-model="store.embeddingApiBase" type="text" placeholder="https://api.openai.com/v1" /></div>
        <div class="form-row"><label class="field-label">{{ t.vector.apiKey }}</label><input v-model="store.embeddingApiKey" type="password" autocomplete="off" placeholder="sk-..." style="font-family: monospace;"/></div>
      </template>
      <div class="form-row"><label class="field-label">{{ t.vector.batchSize }}</label><input v-model.number="store.embedBatchSize" type="number" min="1" /></div>
      <div class="form-row"><label class="field-label">{{ t.vector.splitType }}</label><select v-model="store.splitType"><option v-for="opt in store.options?.split_types || []" :key="opt" :value="opt">{{ t.vector.splitTypes[opt] || opt }}</option></select></div>
      <div class="form-row"><label class="field-label">{{ t.vector.chunkSize }}</label><input v-model.number="store.chunkSize" type="number" min="1" /></div>
      <div class="form-row"><label class="field-label">{{ t.vector.chunkOverlap }}</label><input v-model.number="store.chunkOverlap" type="number" min="0" /></div>
      <div v-if="store.splitType === 'sentence_window'" class="form-row"><label class="field-label">{{ t.vector.windowSize }}</label><input v-model.number="store.windowSize" type="number" min="1" /></div>
      <div v-if="store.splitType === 'hierarchical'" class="form-row"><label class="field-label">{{ t.vector.chunkSizes }}</label><input v-model="store.chunkSizes" type="text" placeholder="2048, 512, 128" /></div>
      <div class="form-row">
        <label class="field-label" for="persist-directory">{{ t.vector.persistDir }}</label>
        <div class="persist-generate-control">
          <input id="persist-directory" v-model="store.persistDir" type="text" :placeholder="t.vector.persistDirPlaceholder" />
          <button class="button button-primary" type="button"
            :disabled="store.loading || !store.persistDir.trim()" @click="$emit('build-index')">
            {{ store.loading ? t.common.loading : t.vector.generate }}
          </button>
        </div>
      </div>
      <div v-if="store.indexStatus === 'running'" class="index-build-progress" role="status" aria-live="polite">
        <div class="index-build-status">
          <span>{{ t.vector.phases[store.indexPhase] || t.vector.building }}</span>
          <strong>{{ Math.round(store.indexProgress * 100) }}%</strong>
        </div>
        <div class="progress-bar" role="progressbar" :aria-valuenow="Math.round(store.indexProgress * 100)" aria-valuemin="0" aria-valuemax="100">
          <div class="fill" :style="{ width: `${store.indexProgress * 100}%` }" />
        </div>
        <div v-if="store.indexPhase === 'embedding' && store.indexTotal" class="index-build-count">
          {{ t.vector.embeddedNodes }} {{ store.indexCompleted.toLocaleString() }} / {{ store.indexTotal.toLocaleString() }}
        </div>
      </div>
    </div>
    <button class="custom-dataset-toggle vector-load-toggle" type="button" :aria-expanded="loadExpanded"
      aria-controls="existing-index-panel" @click="loadExpanded = !loadExpanded">
      <span>{{ t.vector.loadExistingTitle }}</span>
      <span class="collapse-chevron" :class="{ 'is-expanded': loadExpanded }" aria-hidden="true"><img class="icon" :src="chevronUrl" alt="" /></span>
    </button>
    <section v-if="loadExpanded" id="existing-index-panel" class="existing-index-loader vector-collapse-panel">
      <div class="index-load-control">
        <div>
          <label class="field-label" for="existing-index-dir">{{ t.vector.existingDir }}</label>
          <input id="existing-index-dir" v-model="store.existingIndexDir" type="text" :placeholder="t.vector.existingDirPlaceholder" />
        </div>
        <button class="button button-ghost" type="button" :disabled="store.loading || !store.existingIndexDir.trim()" @click="$emit('load-index')">{{ store.loading ? t.common.loading : t.vector.loadExisting }}</button>
      </div>
    </section>
  </section>
</template>
