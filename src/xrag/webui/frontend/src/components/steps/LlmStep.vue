<script setup>
import { ref } from 'vue';
import { api } from '../../api';

const props = defineProps({ store: Object, t: Object });
const emit = defineEmits(['error', 'notice']);
const models = ref([]);
const loadingModels = ref(false);

async function queryModels() {
  loadingModels.value = true;
  try {
    const result = await api.listLLMModels({ api_key: props.store.apiKey || undefined, api_base: props.store.apiBase || undefined });
    models.value = result.models;
    if (result.models.length && !result.models.includes(props.store.apiName)) props.store.apiName = result.models[0];
    emit('notice', props.t.llm.modelsFound.replace('{count}', result.models.length));
  } catch {
    models.value = [];
    emit('error', props.t.llm.modelsQueryFailed);
  } finally {
    loadingModels.value = false;
  }
}
</script>
<template>
  <section class="card">
    <h2>{{ t.llm.title }}</h2>
    <div class="form-row"><label class="field-label">{{ t.llm.select }}</label><select v-model="store.llm"><option v-for="opt in store.options?.llms || []" :key="opt" :value="opt">{{ opt === 'openai' ? 'OpenAI Models' : opt === 'huggingface' ? 'HuggingFace Models' : 'Ollama Models' }}</option></select></div>
    <template v-if="store.llm === 'openai'">
      <div class="form-row"><label class="field-label">{{ t.llm.apiKey }}</label><input v-model="store.apiKey" type="password" placeholder="sk-..." /></div>
      <div class="form-row"><label class="field-label">{{ t.llm.apiBase }}</label><input v-model="store.apiBase" type="text" placeholder="https://api.openai.com/v1" /></div>
      <div class="form-row llm-model-row">
        <label class="field-label">{{ t.llm.modelName }}</label>
        <div class="llm-model-control">
          <select v-if="models.length" v-model="store.apiName" :aria-label="t.llm.modelName">
            <option v-for="model in models" :key="model" :value="model">{{ model }}</option>
          </select>
          <input v-else v-model="store.apiName" type="text" placeholder="gpt-4o" />
          <button class="button button-primary" type="button" :disabled="loadingModels" @click="queryModels">
            {{ loadingModels ? t.common.loading : t.llm.queryModels }}
          </button>
        </div>
      </div>
    </template>
    <template v-else-if="store.llm === 'huggingface'">
      <div class="form-row"><label class="field-label">{{ t.llm.hfModel }}</label><select v-model="store.hfModel"><option v-for="opt in store.options?.hf_models || []" :key="opt" :value="opt">{{ opt }}</option></select></div>
      <div class="form-row"><label class="field-label">{{ t.llm.authToken }}</label><input v-model="store.authToken" type="password" placeholder="hf_..." /></div>
    </template>
    <template v-else>
      <div class="form-row"><label class="field-label">{{ t.llm.ollamaModel }}</label><input v-model="store.ollamaModel" type="text" placeholder="llama2:7b" /></div>
      <div class="form-row"><label class="field-label">{{ t.llm.timeout }}</label><input v-model.number="store.ollamaTimeout" type="number" min="1" /></div>
    </template>
    <div class="form-row"><label class="field-label">{{ t.llm.temperature }}</label><input v-model.number="store.temperature" type="number" step="0.1" /></div>
  </section>
</template>
