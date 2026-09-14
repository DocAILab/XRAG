<script setup>defineProps({ store: Object, t: Object });</script>
<template>
  <section class="card">
    <h2>{{ t.vector.title }}</h2>
    <div class="form-row"><label class="field-label">{{ t.vector.embedding }}</label><select v-model="store.embeddings"><option v-for="opt in store.options?.embeddings || []" :key="opt" :value="opt">{{ opt }}</option></select></div>
    <div class="form-row"><label class="field-label">{{ t.vector.splitType }}</label><select v-model="store.splitType"><option v-for="opt in store.options?.split_types || []" :key="opt" :value="opt">{{ t.vector.splitTypes[opt] || opt }}</option></select></div>
    <div class="form-row"><label class="field-label">{{ t.vector.chunkSize }}</label><input v-model.number="store.chunkSize" type="number" min="1" /></div>
    <div class="form-row"><label class="field-label">{{ t.vector.chunkOverlap }}</label><input v-model.number="store.chunkOverlap" type="number" min="0" /></div>
    <div v-if="store.splitType === 'sentence_window'" class="form-row"><label class="field-label">{{ t.vector.windowSize }}</label><input v-model.number="store.windowSize" type="number" min="1" /></div>
    <div v-if="store.splitType === 'hierarchical'" class="form-row"><label class="field-label">{{ t.vector.chunkSizes }}</label><input v-model="store.chunkSizes" type="text" placeholder="2048, 512, 128" /></div>
    <div class="form-row"><label class="field-label">{{ t.vector.persistDir }}</label><input v-model="store.persistDir" type="text" /></div>
  </section>
</template>
