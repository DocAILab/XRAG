<script setup>defineProps({ store: Object, options: Object, t: Object });</script>
<template>
  <section class="card">
    <h2>{{ t.retrieval.title }}</h2>
    <div class="form-row"><label class="field-label">{{ t.retrieval.type }}</label><select v-model="store.retriever"><option v-for="opt in options?.retrievers || []" :key="opt" :value="opt">{{ opt }} {{ t.retrieval.retrieval }}</option></select></div>
    <div class="form-row"><label class="field-label">{{ t.retrieval.mode }}</label><select v-model.number="store.retrieverMode"><option v-for="opt in options?.retriever_modes || []" :key="opt" :value="opt">{{ t.retrieval.modeValue }} {{ opt }}</option></select></div>
    <div class="form-row"><label class="field-label">{{ t.retrieval.topK }}</label><input v-model.number="store.similarityTopK" type="number" min="1" /></div>
    <div class="form-row"><label class="field-label">{{ t.retrieval.pre }}</label><select v-model="store.queryTransform"><option v-for="opt in options?.query_transform || []" :key="opt" :value="opt">{{ opt === 'none' ? t.retrieval.noneProcess : (t.retrieval.transforms?.[opt] || opt) }}</option></select></div>
    <div class="form-row"><label class="field-label">{{ t.retrieval.rerank }}</label><select v-model="store.postprocessRerank"><option v-for="opt in options?.postprocess_rerank || []" :key="opt" :value="opt">{{ opt === 'none' ? t.retrieval.noneRerank : (t.retrieval.rerankers?.[opt] || opt) }}</option></select></div>
    <div class="form-row"><label class="field-label">{{ t.retrieval.orchestrator }}</label><select v-model="store.orchestrator"><option v-for="item in options?.orchestrators || []" :key="item.id" :value="item.id">{{ item.label }}</option></select></div>
    <div class="form-row"><label class="field-label">{{ t.retrieval.synthesizer }}</label><select v-model="store.responseSynthesizer"><option v-for="item in options?.response_synthesizers || []" :key="item" :value="item">{{ item }}</option></select></div>
    <div class="form-row"><label class="field-label">{{ t.retrieval.qaTemplate }}</label><textarea v-model="store.textQaTemplate" rows="6" /></div>
    <div class="form-row"><label class="field-label">{{ t.retrieval.refineTemplate }}</label><textarea v-model="store.refineTemplate" rows="6" /></div>
  </section>
</template>
