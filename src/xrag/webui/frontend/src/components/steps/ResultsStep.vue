<script setup>
import { computed } from 'vue';

const props = defineProps({ store: Object, options: Object, t: Object });
defineEmits(['restart', 'cancel']);
const displayedSamples = computed(() => props.store.evalSamples.slice(0, 5));

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

const quickMetricIds = ['F1', 'mrr', 'hit10', 'NDCG', 'NLG_chrf_pp', 'NLG_rouge_rougeL', 'NLG_wer'];
const metricRows = computed(() => quickMetricIds.flatMap(id => {
  const globalValue = props.store.evalSummary?.global?.[id];
  if (globalValue != null) return [{ id, value: globalValue, group: 'ConR', validCount: props.store.evalSummary?.n }];
  const metric = props.store.evalSummary?.metrics?.[id];
  if (!metric) return [];
  return [{ id, value: metric.score, group: 'ConG', validCount: metric.valid_count }];
}));

function label(id) { return metricLabels.value[id] || id; }
function score(value) { return Number(value).toFixed(4); }
</script>

<template>
  <section class="card progress-card">
    <div class="results-heading">
      <div>
        <h2>{{ t.results.title }}</h2>
        <div v-if="store.experimentId" class="experiment-id">{{ t.results.experimentId }} <code>{{ store.experimentId }}</code></div>
      </div>
      <span v-if="store.evalMeta?.preset" class="result-preset">{{ t.results.quickPreset }}</span>
    </div>
    <div v-if="store.evalStatus === 'running'">
      <div>{{ t.results.running }} - {{ store.evalCompleted }} / {{ store.evalTotal }} {{ t.results.complete }}</div>
      <div class="progress-bar"><div class="fill" :style="{ width: `${store.evalProgress * 100}%` }" /></div>
      <button class="button button-secondary result-action" type="button" @click="$emit('cancel')">{{ t.common.cancel }}</button>
    </div>
    <div v-else-if="store.evalStatus === 'error'" class="alert alert-error">{{ t.results.failed }} {{ store.evalError }}</div>
    <div v-else-if="store.evalStatus === 'cancelled'" class="alert alert-info">{{ t.results.cancelled }}</div>
    <div v-else-if="store.evalStatus === 'done'" class="result-complete">
      <span class="result-complete-mark" aria-hidden="true">✓</span>
      <span><strong>{{ t.results.finished }}</strong><small>{{ store.evalSummary?.n || 0 }} {{ t.results.processed }}</small></span>
    </div>
    <dl v-if="store.evalMeta" class="run-summary">
      <div><dt>{{ t.results.dataset }}</dt><dd>{{ store.evalMeta.dataset }}</dd></div>
      <div><dt>{{ t.results.model }}</dt><dd>{{ store.evalMeta.model }}</dd></div>
      <div><dt>{{ t.results.samples }}</dt><dd>{{ store.evalCompleted }} / {{ store.evalTotal }}</dd></div>
      <div><dt>{{ t.results.duration }}</dt><dd>{{ store.evalMeta.duration }}</dd></div>
    </dl>
    <div v-if="metricRows.length" class="result-section">
      <div class="result-section-heading">
        <h3>{{ t.results.quickOverview }}</h3>
        <span>{{ metricRows.length }} {{ t.results.metricsCount }}</span>
      </div>
      <!-- <div class="metric-results-grid">
        <div v-for="metric in metricRows" :key="metric.id" class="metric-result">
          <div class="metric-result-label"><span>{{ label(metric.id) }}</span><code>{{ metric.group }}</code></div>
          <strong>{{ metric.validCount ? score(metric.value) : t.results.unavailable }}</strong>
        </div>
      </div> -->
    </div>
    <div v-if="store.evalSummary && Object.keys(store.evalSummary.global || {}).length" class="result-section">
      <div class="result-section-heading"><h3>{{ t.results.retrievalAggregate }}</h3><span>ConR</span></div>
      <table class="summary-table"><thead><tr><th>{{ t.results.metric }}</th><th>{{ t.results.score }}</th><th>{{ t.results.valid }}</th></tr></thead><tbody><tr v-for="(value, key) in store.evalSummary.global" :key="key"><td :title="store.evalSummary.metric_errors?.[key]?.last_error">{{ label(key) }}</td><td>{{ value == null ? t.results.unavailable : score(value) }}</td><td>{{ store.evalSummary.global_valid_counts?.[key] ?? store.evalSummary.n }}</td></tr></tbody></table>
    </div>
    <div v-if="store.evalSummary && Object.keys(store.evalSummary.metrics || {}).length" class="result-section">
      <div class="result-section-heading"><h3>{{ t.results.aggregate }}</h3><span>ConG</span></div>
      <table class="summary-table"><thead><tr><th>{{ t.results.metric }}</th><th>{{ t.results.score }}</th><th>{{ t.results.valid }}</th></tr></thead><tbody><tr v-for="(value, key) in store.evalSummary.metrics" :key="key"><td :title="store.evalSummary.metric_errors?.[key]?.last_error">{{ label(key) }}</td><td>{{ value.valid_count ? score(value.score) : t.results.unavailable }}</td><td>{{ value.valid_count }}</td></tr></tbody></table>
    </div>
    <div v-if="displayedSamples.length" class="result-section">
      <div class="result-section-heading"><h3>{{ t.results.perSample }}</h3><span>{{ displayedSamples.length }} {{ t.results.shown }}</span></div>
      <div class="results-list"><details v-for="sample in displayedSamples" :key="sample.index" class="result-item" :open="sample.index === 0"><summary><span class="sample-index">{{ sample.index + 1 }}</span><span class="q">{{ sample.question }}</span><span class="sample-context-count">{{ sample.retrieval_context.length }} {{ t.results.chunks }}</span></summary><div class="sample-detail"><div class="sample-answer"><span>{{ t.results.answer }}</span><p>{{ sample.actual_response }}</p></div><div class="sample-answer"><span>{{ t.results.expected }}</span><p>{{ sample.expected_answer }}</p></div></div></details></div>
    </div>
    <div class="result-section"><button class="button button-ghost" type="button" @click="$emit('restart')">{{ t.common.startOver }}</button></div>
  </section>
</template>
