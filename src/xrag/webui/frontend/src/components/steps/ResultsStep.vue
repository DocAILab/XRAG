<script setup>
import { computed } from 'vue';

const props = defineProps({ store: Object, options: Object, t: Object });
defineEmits(['restart', 'cancel']);
const displayedSamples = computed(() => props.store.evalSamples.slice(0, 5));

function metricIds(metric) { return metric.metric_ids || [metric.id]; }
function metricLabel(metric, id) { return props.t.metrics.metricLabels?.[id] || metric.label; }
function metricHelp(metric) {
  const details = [metric.full_name || metric.label];
  if (metric.provider) details.push(metric.provider);
  if (metric.inputs) details.push(`Inputs: ${metric.inputs}`);
  if (metric.cost === 'high') details.push(props.t.metrics.highCost);
  if (metric.experimental) details.push(props.t.metrics.experimental);
  return details.join('\n');
}
function hasMetricHelp(metric) {
  return Boolean(metric.full_name || metric.provider || metric.inputs || metric.cost === 'high' || metric.experimental);
}

const resultGroups = computed(() => {
  const summary = props.store.evalSummary;
  if (!summary) return [];

  return (props.options?.metric_groups || []).map(group => {
    const sections = (group.sections || []).map(section => {
      const rows = (section.metrics || []).flatMap(metric => metricIds(metric).flatMap(id => {
        if (Object.prototype.hasOwnProperty.call(summary.global || {}, id)) {
          return [{
            id,
            label: metricLabel(metric, id),
            value: summary.global[id],
            validCount: summary.global_valid_counts?.[id] ?? summary.n,
            help: hasMetricHelp(metric) ? metricHelp(metric) : '',
            error: summary.metric_errors?.[id]?.last_error || '',
          }];
        }
        const result = summary.metrics?.[id];
        if (!result) return [];
        return [{
          id,
          label: metricLabel(metric, id),
          value: result.valid_count ? result.score : null,
          validCount: result.valid_count,
          help: hasMetricHelp(metric) ? metricHelp(metric) : '',
          error: summary.metric_errors?.[id]?.last_error || '',
        }];
      }));
      return { id: section.id, rows };
    }).filter(section => section.rows.length);
    return { ...group, sections };
  }).filter(group => group.sections.length);
});

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
    <div v-for="group in resultGroups" :key="group.id" class="result-section">
      <div class="result-section-heading">
        <h3>{{ t.metrics.groups[group.id] || group.label }}</h3>
        <span v-if="group.code">{{ group.code }}</span>
      </div>
      <div v-for="section in group.sections" :key="section.id" class="result-metric-section">
        <h4 v-if="group.sections.length > 1">{{ t.metrics.sections[section.id] || section.id }}</h4>
        <table class="summary-table">
          <thead><tr><th>{{ t.results.metric }}</th><th>{{ t.results.score }}</th><th>{{ t.results.valid }}</th></tr></thead>
          <tbody>
            <tr v-for="row in section.rows" :key="row.id">
              <td :title="row.error">
                <abbr v-if="row.help" class="metric-name metric-help" :title="row.help">{{ row.label }}</abbr>
                <span v-else>{{ row.label }}</span>
              </td>
              <td>{{ row.value == null ? t.results.unavailable : score(row.value) }}</td>
              <td>{{ row.validCount }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
    <div v-if="displayedSamples.length" class="result-section">
      <div class="result-section-heading"><h3>{{ t.results.perSample }}</h3><span>{{ displayedSamples.length }} {{ t.results.shown }}</span></div>
      <div class="results-list"><details v-for="sample in displayedSamples" :key="sample.index" class="result-item" :open="sample.index === 0"><summary><span class="sample-index">{{ sample.index + 1 }}</span><span class="q">{{ sample.question }}</span><span class="sample-context-count">{{ sample.retrieval_context.length }} {{ t.results.chunks }}</span></summary><div class="sample-detail"><div class="sample-answer"><span>{{ t.results.answer }}</span><p>{{ sample.actual_response }}</p></div><div class="sample-answer"><span>{{ t.results.expected }}</span><p>{{ sample.expected_answer }}</p></div></div></details></div>
    </div>
    <div class="result-section"><button class="button button-ghost" type="button" @click="$emit('restart')">{{ t.common.startOver }}</button></div>
  </section>
</template>
