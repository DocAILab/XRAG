<script setup>
import { ref } from 'vue';
import { api } from '../../api';
import ModelCombobox from '../ModelCombobox.vue';
import eyeUrl from '../../../../static/eye.svg';
import eyeSlashUrl from '../../../../static/eye-slash.svg';

const props = defineProps({ store: Object, t: Object });
const emit = defineEmits(['error', 'notice']);
const models = ref([]);
const loadingModels = ref(false);
const ollamaModels = ref([]);
const loadingOllamaModels = ref(false);
const localHfModels = ref([]);
const loadingLocalHfModels = ref(false);
const showApiKey = ref(false);

function selectHfModel(value) {
  props.store.hfModel = value;
  const model = value.trim().toLowerCase();
  const match = Object.entries(props.store.options?.hf_model_templates || {})
    .find(([name]) => name.toLowerCase() === model);
  props.store.hfTemplate = match?.[1] || 'auto';
}

async function queryModels() {
  loadingModels.value = true;
  try {
    const result = await api.listLLMModels({ api_key: props.store.apiKey || undefined, api_base: props.store.apiBase || undefined });
    models.value = result.models;
    emit('notice', props.t.llm.modelsFound.replace('{count}', result.models.length));
  } catch {
    models.value = [];
    emit('error', props.t.llm.modelsQueryFailed);
  } finally {
    loadingModels.value = false;
  }
}

async function queryOllamaModels() {
  loadingOllamaModels.value = true;
  try {
    const result = await api.listOllamaModels({
      ollama_url: props.store.ollamaUrl || undefined,
      ollama_port: props.store.ollamaPort || undefined,
    });
    ollamaModels.value = result.models;
    emit('notice', props.t.llm.modelsFound.replace('{count}', result.models.length));
  } catch {
    ollamaModels.value = [];
    emit('error', props.t.llm.ollamaModelsQueryFailed);
  } finally {
    loadingOllamaModels.value = false;
  }
}

async function queryLocalHfModels() {
  loadingLocalHfModels.value = true;
  try {
    const result = await api.listLocalHuggingFaceModels();
    localHfModels.value = result.models;
    emit('notice', props.t.llm.modelsFound.replace('{count}', result.models.length));
  } catch {
    localHfModels.value = [];
    emit('error', props.t.llm.localModelsQueryFailed);
  } finally {
    loadingLocalHfModels.value = false;
  }
}
</script>
<template>
  <section class="card">
    <h2>{{ t.llm.title }}</h2>
    <div class="form-row"><label class="field-label">{{ t.llm.select }}</label><select v-model="store.llm"><option v-for="opt in store.options?.llms || []" :key="opt" :value="opt">{{ opt === 'openai' ? 'OpenAI Models' : opt === 'huggingface' ? 'HuggingFace Models' : 'Ollama Models' }}</option></select></div>
    <template v-if="store.llm === 'openai'">
      <div class="form-row">
        <label class="field-label" for="llm-api-key">{{ t.llm.apiKey }}</label>
        <div class="secret-input">
          <input id="llm-api-key" v-model="store.apiKey" :type="showApiKey ? 'text' : 'password'" autocomplete="off" placeholder="sk-..." style="font-family: monospace;"/>
          <button class="secret-toggle" type="button" :aria-label="showApiKey ? t.llm.hideApiKey : t.llm.showApiKey" :title="showApiKey ? t.llm.hideApiKey : t.llm.showApiKey" :aria-pressed="showApiKey" @click="showApiKey = !showApiKey">
            <img :src="showApiKey ? eyeSlashUrl : eyeUrl" alt="" aria-hidden="true" />
          </button>
        </div>
      </div>
      <div class="form-row"><label class="field-label">{{ t.llm.apiBase }}</label><input v-model="store.apiBase" type="text" placeholder="https://api.openai.com/v1" /></div>
      <div class="form-row llm-model-row">
        <label class="field-label">{{ t.llm.modelName }}</label>
        <div class="llm-model-control">
          <ModelCombobox id="llm-model-name" v-model="store.apiName" :options="models"
            :label="t.llm.modelName" placeholder="gpt-4o" />
          <button class="button button-primary" type="button" :disabled="loadingModels" @click="queryModels">
            {{ loadingModels ? t.common.loading : t.llm.queryModels }}
          </button>
        </div>
      </div>
    </template>
    <template v-else-if="store.llm === 'huggingface'">
      <div class="form-row">
        <label class="field-label" for="hf-model-name">{{ t.llm.hfModel }}</label>
        <div class="llm-model-control">
          <ModelCombobox id="hf-model-name" :model-value="store.hfModel" @update:model-value="selectHfModel"
            :options="localHfModels"
            :label="t.llm.hfModel" placeholder="Qwen/Qwen2.5-7B-Instruct" />
          <button class="button button-primary" type="button" :disabled="loadingLocalHfModels" @click="queryLocalHfModels">
            {{ loadingLocalHfModels ? t.common.loading : t.llm.queryLocalModels }}
          </button>
        </div>
      </div>
      <!-- <div class="form-row">
        <label class="field-label" for="hf-template">
          {{ t.llm.hfTemplate }}
          <a class="inline-help" href="https://github.com/hiyouga/LLaMA-Factory" target="_blank" rel="noopener noreferrer" :title="`${t.llm.hfTemplateHelp}\n${t.llm.hfTemplateCredit}`" :aria-label="`${t.llm.hfTemplateHelp} ${t.llm.hfTemplateCredit}`">ⓘ</a>
        </label>
        <select id="hf-template" v-model="store.hfTemplate">
          <option v-for="template in store.options?.hf_templates || ['auto']" :key="template" :value="template">
            {{ template === 'auto' ? t.llm.hfTemplateAuto : template }}
          </option>
        </select>
      </div>
      <div class="hf-number-grid">
        <div class="form-row">
          <label class="field-label" for="hf-context-window">
            {{ t.llm.hfContextWindow }}
            <span class="inline-help" :title="t.llm.hfContextWindowHelp" :aria-label="t.llm.hfContextWindowHelp">ⓘ</span>
          </label>
          <input id="hf-context-window" v-model.number="store.hfContextWindow" type="number" min="0" step="1" />
        </div>
        <div class="form-row">
          <label class="field-label" for="hf-max-new-tokens">{{ t.llm.hfMaxNewTokens }}</label>
          <input id="hf-max-new-tokens" v-model.number="store.hfMaxNewTokens" type="number" min="1" step="1" />
        </div>
      </div>
      <label class="hf-trust-control">
        <input v-model="store.hfTrustRemoteCode" type="checkbox" />
        <span>{{ t.llm.hfTrustRemoteCode }} <small>{{ t.llm.hfTrustWarning }}</small></span>
      </label> -->
    </template>
    <template v-else>
      <div class="ollama-server-control">
        <div class="form-row">
          <label class="field-label" for="ollama-url">{{ t.llm.ollamaUrl }}</label><input id="ollama-url" v-model="store.ollamaUrl" type="text" placeholder="http://localhost" />
        </div>
        <div class="form-row">
          <label class="field-label" for="ollama-port">{{ t.llm.ollamaPort }}</label><input id="ollama-port" v-model.number="store.ollamaPort" type="text" />
        </div>
      </div>
      <div class="form-row">
        <label class="field-label">{{ t.llm.ollamaModel }}</label>
        <div class="llm-model-control">
          <ModelCombobox id="ollama-model-name" v-model="store.ollamaModel" :options="ollamaModels"
            :label="t.llm.ollamaModel" placeholder="llama2:7b" />
          <button class="button button-primary" type="button" :disabled="loadingOllamaModels" @click="queryOllamaModels">
            {{ loadingOllamaModels ? t.common.loading : t.llm.queryOllamaModels }}
          </button>
        </div>
      </div>
      <div class="form-row"><label class="field-label">{{ t.llm.timeout }}</label><input v-model.number="store.ollamaTimeout" type="number" min="1" /></div>
    </template>
    <div class="form-row"><label class="field-label">{{ t.llm.temperature }}</label><input v-model.number="store.temperature" type="number" step="0.1" max="2.0" min="0.0"/></div>
  </section>
</template>
