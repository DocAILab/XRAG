<script setup>
import { computed } from 'vue';

const props = defineProps({ store: Object, options: Object, t: Object });
defineEmits(['restart', 'cancel']);

const metricLabels = computed(() => {
  const labels = {};
  for (const group of props.options?.metric_groups || []) {
    for (const section of group.sections || []) {
      for (const metric of section.metrics || []) {
        for (const id of metric.metric_ids || [metric.id]) {
          labels[id] = props.t.metrics.metricLabels?.[id]
            || `${metric.label}${metric.provider ? ` (${metric.provider})` : ''}`;
        }
      }
    }
  }
  return labels;
});

function label(id) { return metricLabels.value[id] || id; }
function score(value) { return Number(value).toFixed(4); }
</script>

<template>
  <section class="card progress-card">
    <h2>{{ t.results.title }}</h2>
    <div v-if="store.experimentId" class="experiment-id">{{ t.results.experimentId }} <code>{{ store.experimentId }}</code></div>
    <div v-if="store.evalStatus === 'running'">
      <div>{{ t.results.running }} - {{ store.evalCompleted }} / {{ store.evalTotal }} {{ t.results.complete }}</div>
      <div class="progress-bar"><div class="fill" :style="{ width: `${store.evalProgress * 100}%` }" /></div>
      <button class="button button-secondary result-action" type="button" @click="$emit('cancel')">{{ t.common.cancel }}</button>
    </div>
    <div v-else-if="store.evalStatus === 'error'" class="alert alert-error">{{ t.results.failed }} {{ store.evalError }}</div>
    <div v-else-if="store.evalStatus === 'cancelled'" class="alert alert-info">{{ t.results.cancelled }}</div>
    <div v-else-if="store.evalStatus === 'done'" class="alert alert-success">{{ t.results.finished }} - {{ store.evalSummary?.n || 0 }} {{ t.results.processed }}</div>
    <div v-if="store.evalSummary && Object.keys(store.evalSummary.global || {}).length" class="result-section">
      <h3>{{ t.results.retrievalAggregate }}</h3>
      <table class="summary-table"><thead><tr><th>{{ t.results.metric }}</th><th>{{ t.results.score }}</th></tr></thead><tbody><tr v-for="(value, key) in store.evalSummary.global" :key="key"><td>{{ label(key) }}</td><td>{{ score(value) }}</td></tr></tbody></table>
    </div>
    <div v-if="store.evalSummary && Object.keys(store.evalSummary.metrics || {}).length" class="result-section">
      <h3>{{ t.results.aggregate }}</h3>
      <table class="summary-table"><thead><tr><th>{{ t.results.metric }}</th><th>{{ t.results.score }}</th><th>{{ t.results.valid }}</th></tr></thead><tbody><tr v-for="(value, key) in store.evalSummary.metrics" :key="key"><td>{{ label(key) }}</td><td>{{ value.valid_count ? score(value.score) : t.results.unavailable }}</td><td>{{ value.valid_count }}</td></tr></tbody></table>
    </div>
    <div v-if="store.evalSamples.length" class="result-section">
      <h3>{{ t.results.perSample }}</h3>
      <div class="results-list"><div v-for="sample in store.evalSamples" :key="sample.index" class="result-item"><div class="q">{{ sample.index + 1 }}. {{ sample.question }}</div><div class="a">{{ t.results.answer }} {{ sample.actual_response }}</div><div class="a">{{ t.results.expected }} {{ sample.expected_answer }}</div><div class="meta">{{ t.results.retrieved }} {{ sample.retrieval_context.length }} {{ t.results.chunks }}</div></div></div>
    </div>
    <div class="result-section"><button class="button button-ghost" type="button" @click="$emit('restart')">{{ t.common.startOver }}</button></div>
  </section>
</template>
