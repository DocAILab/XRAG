<script setup>
import ModelCombobox from '../ModelCombobox.vue';

const props = defineProps({ store: Object, options: Object, t: Object });
function rerankLabel(value) {
  return value === 'none' ? props.t.retrieval.noneRerank : (props.t.retrieval.rerankers?.[value] || value);
}
function synthesizerLabel(value) {
  return value.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}
</script>
<template>
  <section class="card">
    <h2>{{ t.retrieval.title }}</h2>
    <div class="form-row">
      <div class="retriever-label-row">
        <label class="field-label" for="retriever-type">{{ t.retrieval.type }}</label>
        <a class="retriever-help" href="https://developers.llamaindex.ai/python/framework/integrations/retrievers/auto_merging_retriever/" target="_blank" rel="noopener noreferrer" :aria-label="`${t.retrieval.type} — LlamaIndex`">
          <span aria-hidden="true">ⓘ</span>
        </a>
      </div>
      <select id="retriever-type" v-model="store.retriever"><option v-for="opt in options?.retrievers || []" :key="opt" :value="opt">{{ opt }}</option></select>
    </div>
    <!--
    <div class="form-row"><label class="field-label">{{ t.retrieval.mode }}</label><select v-model.number="store.retrieverMode"><option v-for="opt in options?.retriever_modes || []" :key="opt" :value="opt">{{ t.retrieval.modeValue }} {{ opt }}</option></select></div>
    -->
    <div class="form-row"><label class="field-label">{{ t.retrieval.topK }}</label><input v-model.number="store.similarityTopK" type="number" min="1" /></div>
    <div class="form-row"><label class="field-label">{{ t.retrieval.pre }}</label><select v-model="store.queryTransform"><option v-for="opt in options?.query_transform || []" :key="opt" :value="opt">{{ opt === 'none' ? t.retrieval.noneProcess : (t.retrieval.transforms?.[opt] || opt) }}</option></select></div>
    <div class="form-row">
      <div class="retriever-label-row">
        <label class="field-label" for="postprocess-rerank">{{ t.retrieval.rerank }}</label>
        <a class="retriever-help" href="https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/node_postprocessors/" target="_blank" rel="noopener noreferrer" :aria-label="`${t.retrieval.rerank} — LlamaIndex`">
          <span aria-hidden="true">ⓘ</span>
        </a>
      </div>
      <ModelCombobox id="postprocess-rerank" v-model="store.postprocessRerank"
        :options="options?.postprocess_rerank || []" :label="t.retrieval.rerank" :option-label="rerankLabel" />
    </div>
    <div class="form-row"><label class="field-label">{{ t.retrieval.orchestrator }}</label><select v-model="store.orchestrator"><option v-for="item in options?.orchestrators || []" :key="item.id" :value="item.id">{{ item.label }}</option></select></div>
    <div class="form-row">
      <div class="retriever-label-row">
        <label class="field-label" for="response-synthesizer">{{ t.retrieval.synthesizer }}</label>
        <a class="retriever-help" href="https://developers.llamaindex.ai/python/framework/module_guides/querying/response_synthesizers/#configuring-the-response-mode" target="_blank" rel="noopener noreferrer" :aria-label="`${t.retrieval.synthesizer} — LlamaIndex`">
          <span aria-hidden="true">ⓘ</span>
        </a>
      </div>
      <select id="response-synthesizer" v-model="store.responseSynthesizer"><option v-for="item in options?.response_synthesizers || []" :key="item" :value="item">{{ synthesizerLabel(item) }}</option></select>
    </div>
    <div class="form-row"><label class="field-label">{{ t.retrieval.qaTemplate }}</label><textarea v-model="store.textQaTemplate" rows="6" style="font-family: monospace;"/></div>
    <div class="form-row"><label class="field-label">{{ t.retrieval.refineTemplate }}</label><textarea v-model="store.refineTemplate" rows="6" style="font-family: monospace;"/></div>
  </section>
</template>

<style scoped>
.retriever-label-row {
  margin-bottom: 6px;
}
.retriever-label-row .field-label {
  display: inline;
  margin-bottom: 0;
}
.retriever-help {
  margin-left: 8px;
  color: inherit;
  text-decoration: none;
}
</style>
