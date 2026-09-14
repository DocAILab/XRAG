<script setup>
defineProps({ store: Object, t: Object });
const emit = defineEmits(['load-preset', 'upload-json', 'generate-folder']);
function onFileChange(event, store) { store.uploadFile = event.target.files[0] || null; }
function choosePreset(name, store) { store.presetDataset = name; emit('load-preset', name); }
</script>

<template>
  <section class="card">
    <h2>{{ t.dataset.title }}</h2>
    <h3>{{ t.dataset.preset }}</h3>
    <div class="preset-cards">
      <button v-for="d in store.options?.preset_datasets || []" :key="d.id" type="button"
              class="preset-card" :class="{ 'is-selected': store.presetDataset === d.id }"
              @click="choosePreset(d.id, store)">
        <span class="preset-title">{{ d.label }}</span>
        <span class="preset-desc">{{ t.dataset.descriptions[d.id] || d.id }}</span>
      </button>
    </div>

    <h3 class="section-subheading">{{ t.dataset.custom }}</h3>
    <div class="upload-grid">
      <div class="upload-card">
        <div class="upload-title">{{ t.dataset.folderTitle }}</div>
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
