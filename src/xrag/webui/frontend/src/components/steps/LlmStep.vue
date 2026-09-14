<script setup>defineProps({ store: Object, t: Object });</script>
<template>
  <section class="card">
    <h2>{{ t.llm.title }}</h2>
    <div class="form-row"><label class="field-label">{{ t.llm.select }}</label><select v-model="store.llm"><option v-for="opt in store.options?.llms || []" :key="opt" :value="opt">{{ opt === 'openai' ? 'OpenAI Models' : opt === 'huggingface' ? 'HuggingFace Models' : 'Ollama Models' }}</option></select></div>
    <template v-if="store.llm === 'openai'">
      <div class="form-row"><label class="field-label">{{ t.llm.apiKey }}</label><input v-model="store.apiKey" type="password" placeholder="sk-..." /></div>
      <div class="form-row"><label class="field-label">{{ t.llm.apiBase }}</label><input v-model="store.apiBase" type="text" placeholder="https://api.openai.com/v1" /></div>
      <div class="form-row"><label class="field-label">{{ t.llm.modelName }}</label><input v-model="store.apiName" type="text" placeholder="gpt-4o" /></div>
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
