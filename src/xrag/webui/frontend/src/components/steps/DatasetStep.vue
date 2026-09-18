<script setup>
import { ref } from 'vue';

defineProps({ store: Object, t: Object });
const emit = defineEmits(['load-preset', 'upload-json', 'generate-folder']);
const customExpanded = ref(false);
function onFileChange(event, store) { store.uploadFile = event.target.files[0] || null; }
function choosePreset(name, store) { if (!store.presetLoading) store.presetDataset = name; }
function loadSelectedPreset(store) { if (store.presetDataset && !store.presetLoading) emit('load-preset', store.presetDataset); }
</script>

<template>
  <section class="card">
    <h2>{{ t.dataset.title }}</h2>
    <h3>{{ t.dataset.preset }}</h3>
    <div class="preset-cards">
      <button v-for="d in store.options?.preset_datasets || []" :key="d.id"
        type="button" class="preset-card" :class="{ 'is-selected': store.presetDataset === d.id }"
        :aria-pressed="store.presetDataset === d.id" :disabled="store.presetLoading"
        @click="choosePreset(d.id, store)">
        <span class="preset-title">{{ d.label }}</span>
        <span class="preset-desc">{{ t.dataset.descriptions[d.id] || d.id }}</span>
      </button>
    </div>
    <div v-if="store.presetLoading" class="dataset-load-progress" role="status" aria-live="polite">
      <div class="dataset-load-status">
        <span>{{ t.dataset.loadingPreset }}</span>
        <strong>{{ store.options?.preset_datasets?.find(item => item.id === store.presetDataset)?.label || store.presetDataset }}</strong>
      </div>
      <div class="progress-bar progress-indeterminate" :aria-label="t.dataset.loadingPreset">
        <div class="fill" />
      </div>
    </div>
    <div class="dataset-load-actions">
      <button class="button button-primary" type="button"
        :disabled="!store.presetDataset || store.presetLoading"
        @click="loadSelectedPreset(store)">
        <span v-if="store.presetLoading">{{ t.common.loading }}</span>
        <span v-else>{{ t.dataset.loadSelected }}</span>
      </button>
    </div>

    <button class="custom-dataset-toggle" type="button"
      :aria-expanded="customExpanded"
      aria-controls="custom-dataset-panel"
      @click="customExpanded = !customExpanded"
    >
      <span>{{ t.dataset.custom }}</span>
      <span class="collapse-chevron" :class="{ 'is-expanded': customExpanded }" aria-hidden="true"><svg focusable="false" aria-hidden="true" xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" viewBox="0 0 24 24"><path d="M7 10l5 5 5-5z"></path></svg></span>
    </button>
    <div v-if="customExpanded" id="custom-dataset-panel" class="upload-grid custom-dataset-panel">
      <div class="upload-card">
        <div class="upload-title" lang="en">{{ t.dataset.folderTitle }}</div>
        <div class="upload-desc">{{ t.dataset.folderDesc }}</div>
        <div class="form-row"><label class="field-label">{{ t.dataset.folderPath }}</label><input v-model="store.folderPath" type="text" :placeholder="t.dataset.folderPlaceholder" /></div>
        <div class="form-row"><label class="field-label">{{ t.dataset.questionsPerFile }}</label><input v-model.number="store.folderNum" type="number" min="1" /></div>
        <div class="form-row"><label class="field-label">{{ t.dataset.sentenceLength }}</label><input v-model.number="store.folderSentenceLen" type="number" /></div>
        <div class="form-row"><label class="field-label">{{ t.dataset.outputPath }}</label><input v-model="store.folderOutput" type="text" /></div>
        <button class="button button-ghost" type="button" @click="$emit('generate-folder')">{{ t.dataset.chooseFolder }}</button>
        <ul><li v-for="item in t.dataset.folderFeatures" :key="item">{{ item }}</li></ul>
      </div>
      <div class="upload-card">
        <div class="upload-title">{{ t.dataset.jsonTitle }}</div>
        <div class="upload-desc">{{ t.dataset.jsonDesc }}</div>
        <div class="upload-area">
          <div>{{ t.dataset.dropJson }}</div>
          <div class="file-input-row"><input type="file" accept="application/json,.json" @change="onFileChange($event, store)" /></div>
          <div v-if="store.uploadFile" class="selected-file">{{ store.uploadFile.name }} ({{ Math.round(store.uploadFile.size / 1024) }} KB)</div>
        </div>
        <button class="button button-ghost upload-button" type="button" @click="$emit('upload-json')">{{ t.dataset.chooseJson }}</button>
        <ul><li v-for="item in t.dataset.jsonFeatures" :key="item">{{ item }}</li></ul>
      </div>
    </div>
    <div v-if="store.dataset" class="alert alert-success dataset-summary">
      {{ t.dataset.loaded }} <strong>{{ store.dataset.display }}</strong> — {{ store.dataset.num_documents }} {{ t.dataset.documents }}, {{ store.dataset.num_test_questions }} {{ t.dataset.questions }} ({{ t.dataset.source }}: {{ t.dataset.sources[store.dataset.source] || store.dataset.source }}).
    </div>
  </section>
</template>
