<script setup>
import { computed } from 'vue';

const props = defineProps({ store: Object, options: Object, t: Object });
const groups = computed(() => props.options?.metric_groups || []);
const presets = computed(() => props.options?.metric_presets || []);

function metricIds(metric) { return metric.metric_ids || [metric.id]; }
function sectionIds(section) { return [...new Set(section.metrics.filter(metric => metric.available !== false).flatMap(metricIds))]; }
function groupIds(group) { return [...new Set(group.sections.flatMap(sectionIds))]; }
function isSelected(metric) { return metricIds(metric).every(id => props.store.selectedMetrics.includes(id)); }

function toggleIds(ids, checked) {
  const selected = new Set(props.store.selectedMetrics);
  ids.forEach(id => checked ? selected.add(id) : selected.delete(id));
  props.store.selectedMetrics = [...selected];
  props.store.metricPreset = 'custom';
}

function applyPreset(preset) {
  props.store.selectedMetrics = [...preset.metric_ids];
  props.store.metricPreset = preset.id;
}
</script>

<template>
  <section class="card metrics-card">
    <h2>{{ t.metrics.title }}</h2>
    <div class="metric-presets" role="group" :aria-label="t.metrics.presets">
      <span class="metric-presets-label">{{ t.metrics.presets }}</span>
      <button v-for="preset in presets" :key="preset.id" type="button" class="metric-preset"
        :class="{ active: store.metricPreset === preset.id }" @click="applyPreset(preset)">
        {{ t.metrics.presetNames[preset.id] || preset.id }}
      </button>
      <button type="button" class="metric-preset" :class="{ active: store.metricPreset === 'custom' }"
        @click="store.metricPreset = 'custom'">{{ t.metrics.presetNames.custom }}</button>
    </div>
    <div class="metric-selection-summary">{{ store.selectedMetrics.length }} {{ t.metrics.selected }}</div>

    <div v-for="group in groups" :key="group.id" class="metric-group">
      <div class="metric-group-title">
        <label class="metric-group-header">
          <input type="checkbox" :checked="groupIds(group).every(id => store.selectedMetrics.includes(id))"
            @change="toggleIds(groupIds(group), $event.target.checked)" />
          <span>{{ t.metrics.groups[group.id] || group.label }}</span>
        </label>
        <code v-if="group.code" class="metric-code">{{ group.code }}</code>
      </div>
      <div v-for="section in group.sections" :key="section.id" class="metric-section">
        <label v-if="group.sections.length > 1" class="metric-section-header">
          <input type="checkbox" :checked="sectionIds(section).length > 0 && sectionIds(section).every(id => store.selectedMetrics.includes(id))"
            :disabled="sectionIds(section).length === 0"
            @change="toggleIds(sectionIds(section), $event.target.checked)" />
          <span>{{ t.metrics.sections[section.id] || section.id }}</span>
        </label>
        <div class="metric-grid">
          <label v-for="metric in section.metrics" :key="metric.id" class="metric-check"
            :class="{ disabled: metric.available === false }">
            <input type="checkbox" :checked="isSelected(metric)" :disabled="metric.available === false"
              @change="toggleIds(metricIds(metric), $event.target.checked)" />
            <span class="metric-name">{{ metric.label }}</span>
            <span v-if="metric.direction" class="metric-flair">{{ metric.direction === 'lower' ? '↓' : '↑' }}</span>
            <span v-if="metric.provider" class="metric-flair">{{ metric.provider }}</span>
            <span v-if="metric.cost === 'high'" class="metric-flair">{{ t.metrics.highCost }}</span>
            <span v-if="metric.experimental" class="metric-flair">{{ t.metrics.experimental }}</span>
          </label>
        </div>
        <div v-if="section.id === 'retrieval_utility'" class="metric-note">
          <span>{{ t.metrics.seperOutputs }}</span>
          <span v-if="section.metrics[0]?.requirements">{{ t.metrics.requires }} {{ section.metrics[0].requirements.generation_model }}, {{ section.metrics[0].requirements.entailment_model }}, {{ section.metrics[0].requirements.device }}</span>
          <span v-if="section.metrics[0]?.available === false" class="metric-unavailable">{{ t.metrics.seperUnavailable }}</span>
        </div>
        <div v-if="section.id === 'golden_context'" class="metric-note">{{ t.metrics.goldenContextNote }}</div>
      </div>
    </div>
    <div class="form-row"><label class="field-label">{{ t.metrics.samples }}</label><input v-model.number="store.numSamples" type="number" min="1" /></div>
  </section>
</template>
