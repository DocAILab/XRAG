<script setup>
defineProps({ store: Object, t: Object });
defineEmits(['load-index']);
</script>
<template>
  <section class="card">
    <h2>{{ t.vector.title }}</h2>
    <div v-if="store.indexStatus === 'done'" class="alert alert-success step-ready-summary" role="status">
      {{ t.vector.ready }}
    </div>
    <div class="form-row"><label class="field-label">{{ t.vector.backend }}</label><select v-model="store.embeddingType"><option v-for="opt in store.options?.embedding_backends || ['local', 'openai']" :key="opt" :value="opt">{{ t.vector.backends[opt] || opt }}</option></select></div>
    <div class="form-row"><label class="field-label">{{ t.vector.embedding }}</label><select v-if="store.embeddingType === 'local'" v-model="store.embeddings"><option v-for="opt in store.options?.embeddings || []" :key="opt" :value="opt">{{ opt }}</option></select><input v-else v-model="store.embeddings" type="text" placeholder="text-embedding-3-small" /></div>
    <template v-if="store.embeddingType === 'openai'">
      <div class="form-row"><label class="field-label">{{ t.vector.apiBase }}</label><input v-model="store.embeddingApiBase" type="url" placeholder="https://api.openai.com/v1" /></div>
      <div class="form-row"><label class="field-label">{{ t.vector.apiKey }}</label><input v-model="store.embeddingApiKey" type="password" autocomplete="off" placeholder="sk-..." /></div>
    </template>
    <div class="form-row"><label class="field-label">{{ t.vector.batchSize }}</label><input v-model.number="store.embedBatchSize" type="number" min="1" /></div>
    <div class="form-row"><label class="field-label">{{ t.vector.splitType }}</label><select v-model="store.splitType"><option v-for="opt in store.options?.split_types || []" :key="opt" :value="opt">{{ t.vector.splitTypes[opt] || opt }}</option></select></div>
    <div class="form-row"><label class="field-label">{{ t.vector.chunkSize }}</label><input v-model.number="store.chunkSize" type="number" min="1" /></div>
    <div class="form-row"><label class="field-label">{{ t.vector.chunkOverlap }}</label><input v-model.number="store.chunkOverlap" type="number" min="0" /></div>
    <div v-if="store.splitType === 'sentence_window'" class="form-row"><label class="field-label">{{ t.vector.windowSize }}</label><input v-model.number="store.windowSize" type="number" min="1" /></div>
    <div v-if="store.splitType === 'hierarchical'" class="form-row"><label class="field-label">{{ t.vector.chunkSizes }}</label><input v-model="store.chunkSizes" type="text" placeholder="2048, 512, 128" /></div>
    <div class="form-row"><label class="field-label">{{ t.vector.persistDir }}</label><input v-model="store.persistDir" type="text" /></div>
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
    <section class="existing-index-loader" :aria-labelledby="'existing-index-title'">
      <h3 id="existing-index-title">{{ t.vector.loadExistingTitle }}</h3>
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
